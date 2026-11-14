"""Ablation A3 on Modal, on the exp01 states already on the exp01-repro volume.

  modal run a3_modal.py --model qwen3_8b --smoke      # first 512 states per mode, all controls
  modal run a3_modal.py --model qwen3_8b
  modal volume get exp01-repro a3/qwen3_8b ./results/a3/
  python3 -I a3_summarize.py results/a3/qwen3_8b results/qwen3_8b

Refuses to run unless a3_decision_rule.json matches the hash in logs/a3_decision_rule.log.
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "a3_decision_rule.json")

app = modal.App("exp01-a3")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_file(os.path.join(HERE, "cert_core.py"), "/root/cert_core.py")
    .add_local_file(os.path.join(HERE, "a3_lib.py"), "/root/a3_lib.py")
)

MODELS = {   # same pins as exp01_modal.py
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218",
                     head="lm_head.weight", cap=None),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0),
    "sarvam_30b": dict(repo="sarvamai/sarvam-30b", rev="071ae95e933605ca1104a6b4524a36a98488efa4",
                       head="lm_head.weight", cap=None),
}
RULESETS = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_0.6": ("t1", dict(rule="gumbel", T=0.6)),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
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


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def a3_chunk(model: str, mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import numpy as np
    import torch

    import a3_lib

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    mu64 = a3_lib.mean_row(W)
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in RULESETS.items() if md == mode}
    out, stats = {}, None
    CH = 128
    for c0 in range(0, len(rows), CH):
        h = H[c0:c0 + CH].cuda().float()
        rec, st = a3_lib.run_chunk(W, mu64, h, rows[c0:c0 + CH], rules, controls=(start == 0 and c0 == 0))
        for k, x in rec.items():
            out.setdefault(k, []).append(x)
        if stats is None:
            stats = st
        else:
            for f in ("c2_viol", "c3_mismatch", "dither_se", "dither_n"):
                for k, v in st[f].items():
                    stats[f][k] = stats[f].get(k, 0) + v
        if st["c6_shift_states"] is not None:
            for k in ("c6_shift_states", "c6_eligible", "c6_eligible_missed"):
                stats[k] = st[k]
            stats["c6_shift_n"] = h.shape[0]
    rdir = f"{VOL}/a3/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: torch.cat(v).numpy() for k, v in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, mu_norm=float(mu64.norm()),
                   rows=[(r["pid"], r["group"], r["step"]) for r in rows]),
              open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(mode=mode, start=start, n=len(rows), c2=sum(stats["c2_viol"].values()),
                c3=sum(v for k, v in stats["c3_mismatch"].items() if "/hoeff/" not in k),
                c6=stats.get("c6_shift_states"))


@app.local_entrypoint()
def main(model: str = "qwen3_8b", smoke: bool = False, chunk: int = 4096):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    logged = open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read()
    if sha not in logged:
        raise SystemExit(f"a3_decision_rule.json sha256 {sha[:12]} is not in logs/a3_decision_rule.log")
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{model}/rows_{mode}.jsonl")).count(b"\n")
        n = min(n, 512) if smoke else n
        jobs += [(model, mode, s, min(s + chunk, n), smoke, sha) for s in range(0, n, chunk)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, c2=0, c3=0)
    for r in a3_chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)
