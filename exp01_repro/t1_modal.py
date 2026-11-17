"""T1 on Modal: a dithered INT8 stage, a leaner top-k rule and top-p (t1_rule.json).

  modal run t1_modal.py --smoke            # first 256 states per mode and model
  modal run t1_modal.py                    # all states of the three models, then the tables
Writes logs/t1_summary[_smoke].json (numbers only). Per-state arrays stay on the volume (t1/<model>/).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "t1_rule.json")
app = modal.App("exp01-t1")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "hdir_lib.py", "mix_lib.py", "rules2_lib.py",
           "pipe_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

T_STAR = {"qwen3_8b": 0.5, "gemma4_e4b": 1.5, "sarvam_30b": 0.5}
GROUP8, CHUNK, BATCH = 26, 4096, 128       # 128 as in B1 and M2: a batch of 64 changed 2 stored values of a million
RULES = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
    "top50_lean": ("t1", dict(rule="topk_lean", k=50, T=1.0)),
    "topp_0.9": ("t1", dict(rule="topp", p=0.9, T=1.0)),
}
OLD = ("greedy", "gumbel_1.0", "minp_0.05", "top50")          # rules with stored values to reproduce


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=4 * 3600)
def chunk(model: str, mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
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
    cfg = MODELS[model]
    t = T_STAR[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    V, d = W.shape
    ng = d // 128
    mu64 = a3_lib.mean_row(W)
    w_norm = W.norm(dim=1)
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in RULES.items() if md == mode}
    sarvam = model == "sarvam_30b"
    vs, Wt = pipe_lib.make_variants(W, mu64, t, groups8=(GROUP8,) if sarvam else (), dith8=True)
    if sarvam:
        Pl = {k: x.cuda()[:, :1] for k, x in torch.load(f"{VOL}/hdir/{model}/P_lolo.pt").items()}
        gn = sorted(Pl)
        Pcat = torch.cat([Pl[k] for k in gn], 1)
        r = {k: hdir_lib.err_proj(vs[k].Wq, Wt, Pcat) for k in ("i4", "mix", "d8")}
        sets = {k: {g: (Pl[g], r[k][:, i:i + 1]) for i, g in enumerate(gn)} for k in r}
        src = {"first": dict(v=vs["i4"], kind="subg", sets=sets["i4"]),
               "G": dict(v=vs["mix"], kind="subg", sets=sets["mix"], row_bytes=204),
               "i8": dict(v=vs["i8"], kind="det", row_bytes=d + 4),
               "d8": dict(v=vs["d8"], kind="subg", sets=sets["d8"], row_bytes=d + 2 * ng + 8)}
        arms = {"base": dict(first="first", stages=["i8"], first_extra=4), "d8": dict(first="first", stages=["d8"], first_extra=4),
                "staged": dict(first="first", stages=["G", "i8"], first_extra=4),
                "staged_d8": dict(first="first", stages=["G", "d8"], first_extra=4)}
    else:
        src = {"first": dict(v=vs["i4"], kind="subg", sets=None), "i8": dict(v=vs["i8"], kind="det", row_bytes=d + 4),
               "d8": dict(v=vs["d8"], kind="subg", sets=None, row_bytes=d + 2 * ng + 4)}
        arms = {"base": dict(first="first", stages=["i8"]), "d8": dict(first="first", stages=["d8"])}
    del Wt
    torch.cuda.empty_cache()
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

    # control T0_reproduce
    diff = 0
    if sarvam:
        f = f"{VOL}/stage/{model}/{mode}_{start:06d}"
        assert [tuple(x) for x in json.load(open(f + ".json"))["rows"]][:n] == mine, "state order differs from M2"
        old = np.load(f + ".npz")
        for rn in rules:
            if rn in OLD:
                diff += int((old[f"k2/{rn}/S"][:n] != out[f"base/{rn}/S_i8"]).sum())
                diff += int((old[f"k2/{rn}/R"][:n] != out[f"base/{rn}/R"]).sum())
                for a, b in (("Sg", "S_G"), ("S", "S_i8"), ("R", "R")):
                    diff += int((old[f"staged/{rn}/{a}"][:n] != out[f"staged/{rn}/{b}"]).sum())
    else:
        f = f"{VOL}/subg/{model}/{mode}_{start:06d}"
        assert [tuple(x) for x in json.load(open(f + ".json"))["rows"]][:n] == mine, "state order differs from B1"
        old = np.load(f + ".npz")
        for rn in rules:
            if rn in OLD:
                diff += int((old[f"t{t:g}/int4d_g128/subg/casc/{rn}/S"][:n] != out[f"base/{rn}/S_i8"]).sum())
    stats["t0_diff"] = diff
    rdir = f"{VOL}/t1/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: x.astype(np.float32) for k, x in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, rows=mine), open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(model=model, mode=mode, start=start, n=n, t0=diff, viol=sum(stats["viol"].values()),
                mism=sum(stats["mismatch"].values()), seconds=round(time.time() - t0),
                mean_bytes={rn: {a: round(float(out[f"{a}/{rn}/bytes"].mean()), 4) for a in arms} for rn in rules})


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=16 * 1024, timeout=3600)
def collect(smoke: bool):
    import glob

    import numpy as np

    vol.reload()
    res = {}
    for model in T_STAR:
        rdir = f"{VOL}/t1/{model}{'_smoke' if smoke else ''}"
        if not glob.glob(f"{rdir}/*.npz"):
            continue
        Vn = None
        m = dict(controls={}, table={}, bound_ratio={})
        for mode in ("greedy", "t1"):
            arr, rows, st = {}, [], []
            for f in sorted(glob.glob(f"{rdir}/{mode}_*.npz")):
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
            Vn = {"qwen3_8b": 151936, "gemma4_e4b": 262144, "sarvam_30b": 262144}[model]
            m["controls"][mode] = dict(states=len(rows), t0_diff=sum(s["t0_diff"] for s in st),
                                       violations={k: sum(s["viol"].get(k, 0) for s in st) for k in st[0]["viol"]},
                                       mismatches={k: sum(s["mismatch"].get(k, 0) for s in st) for k in st[0]["mismatch"]})
            m["bound_ratio"][mode] = float(np.median(arr["bmed/d8"] / arr["bmed/i8"]))
            arms = sorted({k.split("/")[0] for k in arr if not k.startswith("bmed/")})
            for rn in sorted({k.split("/")[1] for k in arr if not k.startswith("bmed/")}):
                tab = {}
                for a in arms:
                    B, R = arr[f"{a}/{rn}/bytes"], arr[f"{a}/{rn}/R"]
                    ok = np.median(R) <= 0.001 * Vn and np.percentile(R, 99) <= 0.01 * Vn and np.percentile(B, 99) < 0.5
                    row = dict(bytes=dict(median=float(np.median(B)), p99=float(np.percentile(B, 99)), mean=float(B.mean())),
                               R=dict(median=float(np.median(R)), p99=float(np.percentile(R, 99))),
                               status="DEAD" if np.median(B) >= 0.5 else "GATES_PASS" if ok else "UNDER_HALF",
                               stages={k.split("/")[2]: dict(median=float(np.median(arr[k])), p99=float(np.percentile(arr[k], 99)))
                                       for k in arr if k.startswith(f"{a}/{rn}/S_")},
                               mean_by_language={g: float(B[grp == g].mean()) for g in sorted(set(grp))})
                    tab[a] = row
                m["table"][rn] = tab
        res[model] = m
    return res


@app.local_entrypoint()
def main(smoke: bool = False, models: str = "qwen3_8b,gemma4_e4b,sarvam_30b"):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"t1_rule.json sha256 {sha[:12]} is not logged")
    jobs = []
    for model in models.split(","):
        for mode in ("greedy", "t1"):
            n = b"".join(vol.read_file(f"states/{model}/rows_{mode}.jsonl")).count(b"\n")
            n = min(n, 256) if smoke else n
            jobs += [(model, mode, s, min(s + CHUNK, n), smoke, sha) for s in range(0, n, CHUNK)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, t0=0, viol=0, mism=0, seconds=0)
    for r in chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)
    res = collect.remote(smoke)
    res["rule_sha"] = sha
    json.dump(res, open(os.path.join(HERE, "logs", f"t1_summary{'_smoke' if smoke else ''}.json"), "w"), indent=1)
    print("wrote the summary")
