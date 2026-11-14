"""B1 on Modal: sub-Gaussian dither bound on the exp01 states (subg_rule.json).

  modal run subg_modal.py --model sarvam_30b      # also qwen3_8b, gemma4_e4b
Results go to a volume folder subg/<model>; summarize with subg_summarize.py after downloading.
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "subg_rule.json")
app = modal.App("exp01-subg")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def chunk(model: str, mode: str, start: int, stop: int, rule_sha: str):
    import numpy as np
    import torch

    import a3_lib
    import subg_lib
    from a3_modal import MODELS, RULESETS, _load_head

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
    for c0 in range(0, len(rows), 128):
        rec, st = subg_lib.run_chunk(W, mu64, H[c0:c0 + 128].cuda().float(), rows[c0:c0 + 128], rules)
        for k, x in rec.items():
            out.setdefault(k, []).append(x)
        if stats is None:
            stats = st
        else:
            for f in ("viol", "mismatch", "dither_se", "dither_n"):
                for k, v in st[f].items():
                    stats[f][k] = stats[f].get(k, 0) + v
            for k, v in st["viol_excess_max"].items():
                stats["viol_excess_max"][k] = max(stats["viol_excess_max"][k], v)
    rdir = f"{VOL}/subg/{model}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: torch.cat(v).numpy() for k, v in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, rows=[(r["pid"], r["group"], r["step"]) for r in rows]),
              open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(mode=mode, start=start, n=len(rows), viol=sum(stats["viol"].values()),
                mism=sum(stats["mismatch"].values()))


@app.local_entrypoint()
def main(model: str = "sarvam_30b", chunk_size: int = 4096):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"subg_rule.json sha256 {sha[:12]} is not logged")
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{model}/rows_{mode}.jsonl")).count(b"\n")
        jobs += [(model, mode, s, min(s + chunk_size, n), sha) for s in range(0, n, chunk_size)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, viol=0, mism=0)
    for r in chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)
