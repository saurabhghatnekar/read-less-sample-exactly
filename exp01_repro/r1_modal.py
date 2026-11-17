"""R1 on Modal: Sarvam-30B on a second prompt set and with other dither seeds (r1_rule.json).

  modal run r1_modal.py --smoke        # 256 states per mode, second set, seeds 1234 and 1
  modal run r1_modal.py
Writes logs/r1_summary[_smoke].json (numbers only). Per-state arrays stay on the volume (r1/).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "r1_rule.json")
app = modal.App("exp01-r1")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "hdir_lib.py", "mix_lib.py", "rules2_lib.py",
           "pipe_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

MODEL, T, GROUP8, CHUNK, BATCH, V = "sarvam_30b", 0.5, 26, 4096, 128, 262144
SETS = {"p1": "states/sarvam_30b", "p2": "states/sarvam_30b_p2"}
SEEDS = (1234, 1, 2, 3)
RULES = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
}
ARMS = ("none", "k2", "staged", "staged_d8")


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=4 * 3600)
def chunk(pset: str, seed: int, mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import time

    import numpy as np
    import torch

    import a3_lib
    import hdir_lib
    import pipe_lib
    from a3_modal import MODELS, _load_head

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[MODEL]
    sdir = f"{VOL}/{SETS[pset]}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    d = W.shape[1]
    ng = d // 128
    mu64 = a3_lib.mean_row(W)
    w_norm = W.norm(dim=1)
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in RULES.items() if md == mode}
    vs, Wt = pipe_lib.make_variants(W, mu64, T, seed=seed, groups8=(GROUP8,), dith8=True)
    if pset == "p1":                                     # as M2: the direction fitted without the state's language
        Pl = {k: x.cuda()[:, :1] for k, x in torch.load(f"{VOL}/hdir/{MODEL}/P_lolo.pt").items()}
        gn = sorted(Pl)
        Pcat = torch.cat([Pl[k] for k in gn], 1)
        r = {k: hdir_lib.err_proj(vs[k].Wq, Wt, Pcat) for k in ("i4", "mix", "d8")}
        sets = {k: {g: (Pl[g], r[k][:, i:i + 1]) for i, g in enumerate(gn)} for k in r}
    else:                                                # nothing refitted: K2's pca1 from the first set's even prompts
        p1 = torch.load(f"{VOL}/hdir/{MODEL}/P.pt").cuda()[:, 1:2]
        sets = {k: (p1, hdir_lib.err_proj(vs[k].Wq, Wt, p1)) for k in ("i4", "mix", "d8")}
    del Wt
    torch.cuda.empty_cache()
    src = {"raw": dict(v=vs["i4"], kind="subg", sets=None),
           "first": dict(v=vs["i4"], kind="subg", sets=sets["i4"]),
           "G": dict(v=vs["mix"], kind="subg", sets=sets["mix"], row_bytes=204),
           "i8": dict(v=vs["i8"], kind="det", row_bytes=d + 4),
           "d8": dict(v=vs["d8"], kind="subg", sets=sets["d8"], row_bytes=d + 2 * ng + 8)}
    arms = {"none": dict(first="raw", stages=["i8"]), "k2": dict(first="first", stages=["i8"], first_extra=4),
            "staged": dict(first="first", stages=["G", "i8"], first_extra=4),
            "staged_d8": dict(first="first", stages=["G", "d8"], first_extra=4)}
    out, stats = {}, dict(viol={}, mismatch={})
    for c0 in range(0, len(rows), BATCH):
        rec, st = pipe_lib.run_chunk(W, mu64, w_norm, H[c0:c0 + BATCH].cuda().float(), rows[c0:c0 + BATCH], rules, src, arms)
        for k, x in rec.items():
            out.setdefault(k, []).append(x)
        for f in ("viol", "mismatch"):
            for k, x in st[f].items():
                stats[f][k] = stats[f].get(k, 0) + x
    out = {k: torch.cat(x).numpy() for k, x in out.items()}
    n = len(rows)
    mine = [(r_["pid"], r_["group"], r_["step"]) for r_ in rows]
    diff = -1
    if pset == "p1" and seed == 1234:                    # control R0_reproduce against M2
        f = f"{VOL}/stage/{MODEL}/{mode}_{start:06d}"
        old = np.load(f + ".npz")
        assert [tuple(x) for x in json.load(open(f + ".json"))["rows"]][:n] == mine
        diff = 0
        for rn in rules:
            diff += int((old[f"k2/{rn}/S"][:n] != out[f"k2/{rn}/S_i8"]).sum()) + int((old[f"k2/{rn}/R"][:n] != out[f"k2/{rn}/R"]).sum())
            for a, b in (("Sg", "S_G"), ("S", "S_i8"), ("R", "R")):
                diff += int((old[f"staged/{rn}/{a}"][:n] != out[f"staged/{rn}/{b}"]).sum())
    stats["r0_diff"] = diff
    rdir = f"{VOL}/r1{'_smoke' if smoke else ''}/{pset}_seed{seed}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: x.astype(np.float32) for k, x in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, rows=mine), open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(pset=pset, seed=seed, mode=mode, start=start, n=n, r0=max(diff, 0), viol=sum(stats["viol"].values()),
                mism=sum(stats["mismatch"].values()), seconds=round(time.time() - t0),
                mean_bytes={rn: {a: round(float(out[f"{a}/{rn}/bytes"].mean()), 4) for a in arms} for rn in rules})


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=16 * 1024, timeout=3600)
def collect(smoke: bool):
    import glob

    import numpy as np

    vol.reload()
    res = {}
    for d_ in sorted(glob.glob(f"{VOL}/r1{'_smoke' if smoke else ''}/*")):
        key = os.path.basename(d_)
        m = dict(controls={}, table={})
        for mode in ("greedy", "t1"):
            arr, rows, st = {}, [], []
            for f in sorted(glob.glob(f"{d_}/{mode}_*.npz")):
                z = np.load(f)
                j = json.load(open(f[:-4] + ".json"))
                rows += j["rows"]
                st.append(j["stats"])
                for k in z.files:
                    arr.setdefault(k, []).append(z[k])
            if not rows:
                continue
            arr = {k: np.concatenate(x) for k, x in arr.items()}
            grp = np.array([r[1] for r in rows])
            m["controls"][mode] = dict(states=len(rows), r0_diff=sum(max(s["r0_diff"], 0) for s in st),
                                       r0_checked=sum(s["r0_diff"] >= 0 for s in st),
                                       violations=sum(v for s in st for v in s["viol"].values()),
                                       mismatches=sum(v for s in st for v in s["mismatch"].values()))
            m["bound_k2_vs_none"] = float(np.median(arr["bmed/first"] / arr["bmed/raw"]))
            for rn, (md, _) in RULES.items():
                if md != mode:
                    continue
                tab = {}
                for a in ARMS:
                    B, R = arr[f"{a}/{rn}/bytes"], arr[f"{a}/{rn}/R"]
                    ok = np.median(R) <= 0.001 * V and np.percentile(R, 99) <= 0.01 * V and np.percentile(B, 99) < 0.5
                    tab[a] = dict(bytes=dict(median=float(np.median(B)), p99=float(np.percentile(B, 99)), mean=float(B.mean())),
                                  R=dict(median=float(np.median(R)), p99=float(np.percentile(R, 99))),
                                  status="DEAD" if np.median(B) >= 0.5 else "GATES_PASS" if ok else "UNDER_HALF",
                                  mean_by_language={g: float(B[grp == g].mean()) for g in sorted(set(grp))},
                                  p99_by_language={g: float(np.percentile(B[grp == g], 99)) for g in sorted(set(grp))})
                m["table"][rn] = tab
        res[key] = m
    return res


@app.local_entrypoint()
def main(smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"r1_rule.json sha256 {sha[:12]} is not logged")
    jobs = []
    for pset, sdir in SETS.items():
        for mode in ("greedy", "t1"):
            n = b"".join(vol.read_file(f"{sdir}/rows_{mode}.jsonl")).count(b"\n")
            if smoke:
                if pset == "p2":
                    jobs += [(pset, s, mode, 0, min(256, n), True, sha) for s in (1234, 1)]
                continue
            for seed in SEEDS:
                starts = range(0, n, CHUNK)
                if pset == "p1" and seed == 1234:
                    starts = [0]
                jobs += [(pset, seed, mode, s, min(s + CHUNK, n), False, sha) for s in starts]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, r0=0, viol=0, mism=0, seconds=0)
    for r in chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)
    res = collect.remote(smoke)
    res["rule_sha"] = sha
    json.dump(res, open(os.path.join(HERE, "logs", f"r1_summary{'_smoke' if smoke else ''}.json"), "w"), indent=1)
    print("wrote the summary")
