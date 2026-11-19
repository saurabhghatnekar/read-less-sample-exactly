"""Strict count of dithered codes that leave their range, on the real heads (clip_check.json). CPU only.

  modal run clip_check_modal.py [--models ...]
Writes logs/clip_check.json-style results to logs/clip_check_<model>.json (numbers only).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
KER = os.path.join(HERE, "..", "kernel")
RULE = os.path.join(HERE, "clip_check.json")
app = modal.App("exp01-clip-check")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"})
         .add_local_file(os.path.join(KER, "common.py"), "/root/common.py"))

HEADS = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218", head="lm_head.weight"),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight"),
    "sarvam_30b": dict(repo="sarvamai/sarvam-30b", rev="071ae95e933605ca1104a6b4524a36a98488efa4", head="lm_head.weight"),
    "qwen3_1.7b": dict(repo="Qwen/Qwen3-1.7B", rev="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", head="lm_head.weight"),
    "qwen3_0.6b": dict(saved=True),
    "sarvam_1": dict(saved=True),
    "smollm3_3b": dict(saved=True),
    "phi4_mini": dict(saved=True),
}
SHIFTS = (0.0, 0.5, 1.0, 1.5)
G, CHUNK = 128, 8192
FP16_MIN_NORMAL = 6.103515625e-05


@app.function(image=image, volumes={VOL: vol}, cpu=16, memory=64 * 1024, timeout=2 * 3600, retries=0)
def run(model: str, rule_sha: str):
    import torch

    from common import load_head

    torch.set_num_threads(16)
    c = HEADS[model]
    W = (torch.load(f"{VOL}/states/{model}/W.pt", map_location="cpu") if c.get("saved") else load_head(c)).to(torch.bfloat16)
    V, d = W.shape
    ng = d // G
    assert d % G == 0, d
    mu64 = W.double().mean(0)                                         # a3_lib.mean_row
    out = dict(model=model, V=V, d=d, rule_sha=rule_sha, torch=torch.__version__, settings=[])
    seeds = {t: (1234,) for t in SHIFTS}
    if model == "sarvam_30b":
        seeds[0.5] = (1234, 1, 2, 3)
    for t in SHIFTS:
        us = {}
        for seed in seeds[t]:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            us[seed] = (torch.rand(d, generator=gen, dtype=torch.float64) - 0.5).float().view(1, ng, G)
        rec = {seed: dict(shift=t, seed=seed, int4_above_7=0, int4_below_minus_8=0, int4_equal_minus_8=0, int4_max_abs_x=0.0,
                          int8_out_of_range=0, int8_max_abs_x=0.0) for seed in seeds[t]}
        zero_scale = sub_scale = sub_nonzero = 0
        min_nonzero_amax = float("inf")
        worst = []
        for r0 in range(0, V, CHUNK):
            w = W[r0:r0 + CHUNK].float()
            wt = (w.double() - t * mu64).float() if t != 0 else w     # a3_lib.shifted
            wg = wt.view(-1, ng, G)
            amax = wg.abs().amax(-1, keepdim=True)
            nz = amax[amax > 0]
            if nz.numel():
                min_nonzero_amax = min(min_nonzero_amax, float(nz.min()))
            s4h, s8h = (amax / 6.5).half(), (amax / 126.5).half()
            zero_scale += int((s4h == 0).sum())
            sub = (s4h != 0) & (s4h.float() < FP16_MIN_NORMAL)
            sub_scale += int(sub.sum())
            sub_nonzero += int((sub & (amax > 0)).sum())
            s4, s8 = s4h.float().clamp_min(1e-12), s8h.float().clamp_min(1e-12)
            for seed, ug in us.items():
                x = wg / s4 + ug
                q = torch.round(x)
                a = rec[seed]
                a["int4_above_7"] += int((q > 7).sum())
                a["int4_below_minus_8"] += int((q < -8).sum())
                a["int4_equal_minus_8"] += int((q == -8).sum())
                a["int4_max_abs_x"] = max(a["int4_max_abs_x"], float(x.abs().max()))
                bad = (q > 7) | (q < -8)
                if bad.any() and len(worst) < 50:
                    for i, gi, j in bad.nonzero()[:50 - len(worst)].tolist():
                        worst.append(dict(seed=seed, row=r0 + i, group=gi, coord=j, x=float(x[i, gi, j]),
                                          amax=float(amax[i, gi, 0]), scale=float(s4[i, gi, 0])))
                x8 = wg / s8 + ug
                q8 = torch.round(x8)
                a["int8_out_of_range"] += int((q8.abs() > 127).sum())
                a["int8_max_abs_x"] = max(a["int8_max_abs_x"], float(x8.abs().max()))
        for seed in seeds[t]:
            out["settings"].append(dict(rec[seed], groups=V * ng, zero_scale_groups=zero_scale, subnormal_scale_groups=sub_scale,
                                        subnormal_scale_nonzero_groups=sub_nonzero, min_nonzero_group_max=min_nonzero_amax,
                                        examples=[w_ for w_ in worst if w_["seed"] == seed]))
            print(model, json.dumps({k: v for k, v in out["settings"][-1].items() if k != "examples"}), flush=True)
    return out


@app.local_entrypoint()
def main(models: str = ",".join(HEADS)):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"clip_check.json sha256 {sha[:12]} is not logged")
    names = models.split(",")
    for name, res in zip(names, run.starmap([(m, sha) for m in names], return_exceptions=True)):
        if not isinstance(res, dict):
            print(name, "FAILED", f"{res!r}"[:1200], flush=True)
            continue
        json.dump(res, open(os.path.join(HERE, "logs", f"clip_check_{name}.json"), "w"), indent=1)
        bad = sum(s["int4_above_7"] + s["int4_below_minus_8"] + s["int8_out_of_range"] for s in res["settings"])
        print(name, f"V {res['V']} d {res['d']}: out-of-range codes over {len(res['settings'])} settings = {bad}; largest |x| 4-bit",
              round(max(s["int4_max_abs_x"] for s in res["settings"]), 4), "8-bit",
              round(max(s["int8_max_abs_x"] for s in res["settings"]), 3), flush=True)
