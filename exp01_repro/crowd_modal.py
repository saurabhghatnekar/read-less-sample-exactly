"""K1 on Modal: crowding of the top reference logits on the greedy exp01 states (crowd_rule.json).

  modal run crowd_modal.py --model qwen3_8b --smoke     # first 512 states
  modal run crowd_modal.py --model sarvam_30b           # also qwen3_8b, gemma4_e4b
  modal volume get exp01-repro crowd/sarvam_30b ./results/crowd/
  python3 -I crowd_summarize.py results/crowd results/flip_bf16
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "crowd_rule.json")
app = modal.App("exp01-crowd")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "crowd_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

T_STAR = {"qwen3_8b": 0.5, "gemma4_e4b": 1.5, "sarvam_30b": 0.5}   # the B1 greedy shifts
CHUNK = 4096                                                         # the A3 and B1 job size, for control K0


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def chunk(model: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import numpy as np
    import torch

    import a3_lib
    import crowd_lib
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_greedy.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_greedy.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    mu64 = a3_lib.mean_row(W)
    w_norm = W.norm(dim=1)
    copies = {}
    for tag, t in (("t0", 0.0), ("ts", T_STAR[model])):
        vs = a3_lib.build(W, mu64, t)
        copies[tag] = vs["int4d_g128"]
        del vs
        torch.cuda.empty_cache()
    out = {}
    for c0 in range(0, len(rows), 128):
        rec = crowd_lib.run_chunk(W, mu64, w_norm, copies, H[c0:c0 + 128].cuda().float(), cfg["cap"])
        for k, x in rec.items():
            out.setdefault(k, []).append(x)
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    n = len(rows)
    mine = [(r["pid"], r["group"], r["step"]) for r in rows]

    # control K0: the shortlist equals what A3 (Hoeffding) and B1 (sub-Gaussian) stored at t = 0
    k0 = {}
    for src, key, mykey in (("a3", "t0/int4d_g128/hoeff/casc/greedy/S", "t0/hoeff/need_actual"),
                            ("subg", "t0/int4d_g128/subg/casc/greedy/S", "t0/subg/need_actual")):
        f = f"{VOL}/{src}/{model}/greedy_{start:06d}"
        assert [tuple(r) for r in json.load(open(f + ".json"))["rows"]][:n] == mine, "state order differs"
        k0[src] = int((np.load(f + ".npz")[key][:n] != out[mykey]).sum())
    rdir = f"{VOL}/crowd/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/greedy_{start:06d}.npz", **{k: v.astype(np.float32) for k, v in out.items()})
    json.dump(dict(stats=dict(k0=k0, n=n), rule_sha=rule_sha, t_star=T_STAR[model], rows=mine),
              open(f"{rdir}/greedy_{start:06d}.json", "w"))
    vol.commit()
    return dict(start=start, n=n, k0=k0, need_median=float(np.median(out["t0/hoeff/need_actual"])),
                n16_median=float(np.median(out["n_within_16"])))


@app.local_entrypoint()
def main(model: str = "qwen3_8b", smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"crowd_rule.json sha256 {sha[:12]} is not logged")
    n = b"".join(vol.read_file(f"states/{model}/rows_greedy.jsonl")).count(b"\n")
    n = min(n, 512) if smoke else n
    jobs = [(model, s, min(s + CHUNK, n), smoke, sha) for s in range(0, n, CHUNK)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    k0 = dict(a3=0, subg=0)
    tot = 0
    for r in chunk.starmap(jobs, order_outputs=False):
        tot += r["n"]
        for k in k0:
            k0[k] += r["k0"][k]
        print(r, flush=True)
    print("TOTAL", tot, "K0 differences", k0)
