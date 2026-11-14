"""F1 on Modal: flip rate of uncertified low-bit heads on the exp01 states (flip_rule.json).

  modal run flip_modal.py --model qwen3_8b --smoke     # first 512 states per mode
  modal run flip_modal.py --model qwen3_8b             # also gemma4_e4b, sarvam_30b
  modal volume get exp01-repro flip/qwen3_8b ./results/flip/
  python3 -I flip_summarize.py results/flip/qwen3_8b "Qwen3-8B"

Refuses to run unless flip_rule.json matches a hash in logs/a3_decision_rule.log.
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "flip_rule.json")
app = modal.App("exp01-flip")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "flip_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

T_STAR = {"qwen3_8b": 0.5, "gemma4_e4b": 1.5, "sarvam_30b": 0.5}   # the B1 greedy shifts (l4_protocol.json)
A3_CHUNK = 4096                                                      # a3_modal.py's job size, for control F0


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def chunk(model: str, mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import numpy as np
    import torch

    import a3_lib
    import flip_lib
    from a3_modal import MODELS, RULESETS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert start % A3_CHUNK == 0
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    mu64 = a3_lib.mean_row(W)
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in RULESETS.items() if md == mode}
    out = {}
    for t in (0.0, T_STAR[model]):
        vs = flip_lib.build_copies(W, mu64, t)
        for c0 in range(0, len(rows), 128):                          # 128 states at a time, as in A3
            rec = flip_lib.run_chunk(W, mu64, vs, H[c0:c0 + 128].cuda().float(), rows[c0:c0 + 128], rules)
            for k, x in rec.items():
                if k.startswith("ref/") and t != 0.0:
                    continue
                out.setdefault(k, []).append(x)
        del vs
        torch.cuda.empty_cache()
    out = {k: torch.cat(v).numpy() for k, v in out.items()}

    # control F0: the largest error per state equals what A3 stored for the same state, copy and shift
    a3, a3rows = {}, []
    for s in range(start, stop, A3_CHUNK):
        f = f"{VOL}/a3/{model}/{mode}_{s:06d}"
        z = np.load(f + ".npz")
        for k in out:
            if k.endswith("/maxerr"):
                a3.setdefault(k, []).append(z[k])
        a3rows += json.load(open(f + ".json"))["rows"]
    n = len(rows)
    mine = [(r["pid"], r["group"], r["step"]) for r in rows]
    assert [tuple(r) for r in a3rows[:n]] == mine, "state order differs from A3"
    f0_diff, f0_max = {}, {}
    for k, v in a3.items():
        d = np.abs(out[k].astype(np.float64) - np.concatenate(v)[:n].astype(np.float64))
        f0_diff[k], f0_max[k] = int((d != 0).sum()), float(d.max())
    emitted = np.array([r["tok"] for r in rows])
    stats = dict(f0_diff=f0_diff, f0_max=f0_max, n=n,
                 ref_eq_emitted={k[4:]: int((out[k] == emitted).sum()) for k in out if k.startswith("ref/")})

    small = {}
    for k, v in out.items():
        if k.endswith(("/flip", "/outside")):
            small[k] = v.astype(np.uint8)
        elif k.endswith("/setdiff") or k.startswith("ref/"):
            small[k] = v.astype(np.int32)
        else:
            small[k] = v.astype(np.float32)
    rdir = f"{VOL}/flip/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **small)
    json.dump(dict(stats=stats, rule_sha=rule_sha, t_star=T_STAR[model], rows=mine),
              open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    flips = {k[:-5]: int(v.sum()) for k, v in small.items() if k.endswith("/flip") and k.startswith("t0/")}
    return dict(mode=mode, start=start, n=n, f0=sum(f0_diff.values()), f0_max=max(f0_max.values()), flips_t0=flips)


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def bf16_chunk(model: str, mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    """flip_addendum_bf16.json: the BF16 product as a copy, with the corrected control F1."""
    import numpy as np
    import torch

    import flip_lib
    from a3_modal import MODELS, RULESETS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert start % A3_CHUNK == 0
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    Wb = _load_head(cfg).cuda()
    W = Wb.float()
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in RULESETS.items() if md == mode}
    out = {}
    for c0 in range(0, len(rows), 128):                              # 128 states at a time, as in exp01's C1
        rec = flip_lib.run_bf16_chunk(W, Wb, H[c0:c0 + 128].cuda().float(), rows[c0:c0 + 128], rules)
        for k, x in rec.items():
            out.setdefault(k, []).append(x)
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    n = len(rows)
    mine = [(r["pid"], r["group"], r["step"]) for r in rows]

    # reference tokens must be the F1 run's (flip/<model>, 8192-state jobs), on every state and rule
    f1 = np.load(f"{VOL}/flip/{model}/{mode}_{start - start % 8192:06d}.npz")
    off = start % 8192
    ref_diff = {k: int((out[k] != f1[k][off:off + n]).sum()) for k in out if k.startswith("ref/")}
    stats = dict(n=n, ref_diff=ref_diff)
    if mode == "greedy":
        emitted = np.array([r["tok"] for r in rows])
        stats["c1_match"] = int((out["bf16_prod/argmax"] == emitted).sum())
        if not smoke and stop - start <= A3_CHUNK:                   # exp01 stored one count per 4096 states
            stats["c1_match_exp01"] = json.load(open(f"{VOL}/results/{model}/{mode}_{start:06d}.json"))["stats"]["c1_match"]
    small = {}
    for k, v in out.items():
        if k.endswith(("/flip", "/outside", "/tie")):
            small[k] = v.astype(np.uint8)
        elif k.endswith(("/setdiff", "/argmax")) or k.startswith("ref/"):
            small[k] = v.astype(np.int32)
        else:
            small[k] = v.astype(np.float32)
    rdir = f"{VOL}/flip_bf16/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **small)
    json.dump(dict(stats=stats, rule_sha=rule_sha, rows=mine), open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(mode=mode, start=start, n=n, ref_diff=sum(ref_diff.values()), c1=stats.get("c1_match"),
                c1_exp01=stats.get("c1_match_exp01"),
                flips={k[10:-5]: int(v.sum()) for k, v in small.items() if k.endswith("/flip")})


@app.local_entrypoint()
def bf16(model: str = "qwen3_8b", smoke: bool = False):
    """modal run flip_modal.py::bf16 --model qwen3_8b [--smoke]"""
    add = os.path.join(HERE, "flip_addendum_bf16.json")
    sha = hashlib.sha256(open(add, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"flip_addendum_bf16.json sha256 {sha[:12]} is not logged")
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{model}/rows_{mode}.jsonl")).count(b"\n")
        n = min(n, 512) if smoke else n
        jobs += [(model, mode, s, min(s + A3_CHUNK, n), smoke, sha) for s in range(0, n, A3_CHUNK)]
    print(f"{len(jobs)} jobs, addendum {sha[:12]}")
    tot = dict(n=0, ref_diff=0)
    c1 = c1e = 0
    for r in bf16_chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        c1 += r["c1"] or 0
        c1e += r["c1_exp01"] or 0
        print(r, flush=True)
    print("TOTAL", tot, "C1 here", c1, "C1 exp01", c1e)


@app.local_entrypoint()
def main(model: str = "qwen3_8b", smoke: bool = False, chunk_size: int = 8192):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"flip_rule.json sha256 {sha[:12]} is not logged")
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{model}/rows_{mode}.jsonl")).count(b"\n")
        n = min(n, 512) if smoke else n
        jobs += [(model, mode, s, min(s + chunk_size, n), smoke, sha) for s in range(0, n, chunk_size)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, f0=0)
    f0_max = 0.0
    for r in chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        f0_max = max(f0_max, r["f0_max"])
        print(r, flush=True)
    print("TOTAL", tot, "F0 largest difference", f0_max)
