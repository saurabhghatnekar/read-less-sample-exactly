"""Three of the lab's controls that our first run never checked, on the saved states (numbers only, no text leaves Modal).

  C7 language: per Indian-language group and mode, the median over prompts of the share of letters in the target script.
  C8 sampling: on T=1 states, mean NLL of the emitted token / mean entropy (0.9 to 1.1 passes; catches top-k/top-p leaks).

  C6 planted faults (controls_lib.py): modal run controls_modal.py --model gemma4_e4b --step c6

  modal run controls_modal.py --model gemma4_e4b
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("exp01-controls")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(os.path.join(HERE, "cert_core.py"), "/root/cert_core.py")
    .add_local_file(os.path.join(HERE, "controls_lib.py"), "/root/controls_lib.py")
)
RULESETS = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_0.6": ("t1", dict(rule="gumbel", T=0.6)),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
}
MODELS = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218",
                     head="lm_head.weight", cap=None),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0),
}


def _load_head(cfg):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    try:
        idx = hf_hub_download(cfg["repo"], "model.safetensors.index.json", revision=cfg["rev"])
        shard = json.load(open(idx))["weight_map"][cfg["head"]]
    except Exception:
        shard = "model.safetensors"
    f = hf_hub_download(cfg["repo"], shard, revision=cfg["rev"])
    with safe_open(f, "pt") as s:
        return s.get_tensor(cfg["head"]).to(torch.bfloat16)
SCRIPT = {"hindi": (0x0900, 0x097F), "bengali": (0x0980, 0x09FF), "tamil": (0x0B80, 0x0BFF), "telugu": (0x0C00, 0x0C7F)}


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=1200, retries=0)
def controls(model: str):
    import statistics

    import torch
    from transformers import AutoTokenizer

    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["rev"])
    out = {"model": model, "c7": {}, "c8": {}}
    for mode in ("greedy", "t1"):
        rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")]
        byp = {}
        for r in rows:
            byp.setdefault((r["group"], r["pid"]), []).append(r["tok"])
        for grp, (lo, hi) in SCRIPT.items():
            shares = []
            for (g, _), ids in byp.items():
                if g != grp:
                    continue
                letters = [c for c in tok.decode(ids, skip_special_tokens=True) if c.isalpha() or lo <= ord(c) <= hi]
                shares.append(sum(lo <= ord(c) <= hi for c in letters) / max(1, len(letters)))
            out["c7"][f"{mode}/{grp}"] = dict(median=statistics.median(shares), min=min(shares), prompts=len(shares))
        out[f"n_{mode}"] = len(rows)

    torch.backends.cuda.matmul.allow_tf32 = False
    W = _load_head(cfg).cuda().float()
    H = torch.load(f"{sdir}/h_t1.pt")
    rows = [json.loads(l) for l in open(f"{sdir}/rows_t1.jsonl")]
    nll, ent, groups = [], [], [r["group"] for r in rows]
    for c0 in range(0, len(rows), 256):
        z = H[c0:c0 + 256].cuda().float() @ W.T
        if cfg["cap"]:
            z = cfg["cap"] * torch.tanh(z / cfg["cap"])
        lp = torch.log_softmax(z.double(), 1)
        t = torch.tensor([r["tok"] for r in rows[c0:c0 + 256]], device="cuda")
        nll.append(-lp.gather(1, t[:, None]).squeeze(1).cpu())
        ent.append(-(lp.exp() * lp).sum(1).cpu())
    nll, ent = torch.cat(nll), torch.cat(ent)
    out["c8"]["all"] = dict(ratio=(nll.mean() / ent.mean()).item(), mean_nll=nll.mean().item(), mean_entropy=ent.mean().item())
    for g in sorted(set(groups)):
        m = torch.tensor([x == g for x in groups])
        out["c8"][g] = dict(ratio=(nll[m].mean() / ent[m].mean()).item(), n=int(m.sum()))
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=1800, retries=0)
def planted(model: str, n: int = 1024, n_fp64: int = 64):
    """C6 on the first n states of each mode (FP64 comparison on the first n_fp64)."""
    import torch

    import cert_core as cc
    import controls_lib as cl

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    W = _load_head(cfg).cuda().float()
    w_norm = W.norm(dim=1)
    variants = {v.name: cc.prepare(v) for v in (cc.make_int8_row(W), cc.make_int4_g(W), cc.make_int4_dither(W),
                                                cc.make_int4_rot(W), cc.make_nvfp4(W))}
    out = {"model": model}
    for mode in ("greedy", "t1"):
        H = torch.load(f"{sdir}/h_{mode}.pt")[:n]
        rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][:n]
        rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in RULESETS.items() if md == mode}
        tot = {}
        for c0 in range(0, len(rows), 128):
            h = H[c0:c0 + 128].cuda().float()
            keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows[c0:c0 + 128]],
                                dtype=torch.int64, device="cuda")
            cl.merge(tot, cl.planted_chunk(cc, variants, W, w_norm, h, keys, rules, n_fp64 if c0 == 0 else 0))
            torch.cuda.empty_cache()
        checks, ok = cl.verdict(tot, list(rules), list(variants))
        out[mode] = dict(passed=ok, checks=checks, counters=tot)
    return out


@app.local_entrypoint()
def main(model: str = "gemma4_e4b", step: str = "c78"):
    """--step c78 (language and sampling) or --step c6 (planted faults)."""
    res = (controls if step == "c78" else planted).remote(model)
    name = f"controls_{model}.json" if step == "c78" else f"controls_c6_{model}.json"
    json.dump(res, open(os.path.join(HERE, "results", name), "w"), indent=1)
    print(json.dumps(res, indent=1))
