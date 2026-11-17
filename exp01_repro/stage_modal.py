"""M2 on Modal: the heavy group read in INT8 only for the rows the rule still needs (stage_rule.json). Sarvam-30B.

  modal run stage_modal.py --smoke        # first 512 states per mode
  modal run stage_modal.py                # all states, then the per-language table
Writes logs/stage_summary[_smoke].json (numbers only). Per-state arrays stay on the volume (stage/sarvam_30b/).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "stage_rule.json")
app = modal.App("exp01-stage")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "hdir_lib.py", "mix_lib.py", "stage_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

MODEL, T, GROUP8, CHUNK, V = "sarvam_30b", 0.5, 26, 4096, 262144
ALL_RULES = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_0.6": ("t1", dict(rule="gumbel", T=0.6)),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
}


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=4 * 3600)
def chunk(mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import time

    import numpy as np
    import torch

    import a3_lib
    import cert_core as cc
    import hdir_lib
    import mix_lib
    import stage_lib
    from a3_modal import MODELS, _load_head

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[MODEL]
    sdir = f"{VOL}/states/{MODEL}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    mu64 = a3_lib.mean_row(W)
    w_norm = W.norm(dim=1)
    Pl = {k: x.cuda()[:, :1] for k, x in torch.load(f"{VOL}/hdir/{MODEL}/P_lolo.pt").items()}
    gnames = sorted(Pl)
    Pcat = torch.cat([Pl[k] for k in gnames], 1)
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in ALL_RULES.items() if md == mode}
    vs = a3_lib.build(W, mu64, T)
    del vs["int4_g128"]
    Wt = a3_lib.shifted(W, mu64, T)
    v = vs["int4d_g128"]
    m = cc.prepare(mix_lib.make_mixed_dither(Wt, (GROUP8,)))
    m.t, m.wt_norm = T, v.wt_norm
    rp, rm = hdir_lib.err_proj(v.Wq, Wt, Pcat), hdir_lib.err_proj(m.Wq, Wt, Pcat)
    del Wt
    torch.cuda.empty_cache()
    sp = {k: (Pl[k], rp[:, i:i + 1]) for i, k in enumerate(gnames)}
    sm = {k: (Pl[k], rm[:, i:i + 1]) for i, k in enumerate(gnames)}
    out, stats = {}, dict(viol={}, mismatch={})
    for c0 in range(0, len(rows), 128):
        rec, st = stage_lib.run_chunk_staged(W, mu64, w_norm, v, m, vs["int8_row"], sp, sm,
                                             H[c0:c0 + 128].cuda().float(), rows[c0:c0 + 128], rules)
        for k, x in rec.items():
            out.setdefault(k, []).append(x)
        for f in ("viol", "mismatch"):
            for k, x in st[f].items():
                stats[f][k] = stats[f].get(k, 0) + x
    out = {k: torch.cat(x).numpy() for k, x in out.items()}
    n = len(rows)
    mine = [(r_["pid"], r_["group"], r_["step"]) for r_ in rows]

    # control S0_reproduce: k2 is M1's stored plain/k2; M1's arm both is copied alongside for the tables
    f = f"{VOL}/mix/{MODEL}/{mode}_{start:06d}"
    assert [tuple(x) for x in json.load(open(f + ".json"))["rows"]][:n] == mine, "state order differs from M1"
    old = np.load(f + ".npz")
    s0 = 0
    for rn in rules:
        for qn in ("S", "bytes"):
            s0 += int((old[f"plain/k2/{rn}/{qn}"][:n] != out[f"k2/{rn}/{qn}"]).sum())
        for qn in ("S", "R", "bytes"):
            out[f"both/{rn}/{qn}"] = old[f"mixed/k2/{rn}/{qn}"][:n]
    stats["s0_diff"] = s0
    rdir = f"{VOL}/stage/{MODEL}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: x.astype(np.float32) for k, x in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, rows=mine), open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(mode=mode, start=start, n=n, s0=s0, viol=sum(stats["viol"].values()), mism=sum(stats["mismatch"].values()),
                seconds=round(time.time() - t0),
                mean_bytes={rn: {a: round(float(out[f"{a}/{rn}/bytes"].mean()), 4) for a in ("k2", "both", "staged")}
                            for rn in rules})


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=16 * 1024, timeout=3600)
def collect(smoke: bool):
    import glob

    import numpy as np

    vol.reload()
    rdir = f"{VOL}/stage/{MODEL}{'_smoke' if smoke else ''}"
    rng = np.random.default_rng(0)

    def q(x):
        return dict(median=float(np.median(x)), p99=float(np.percentile(x, 99)), mean=float(x.mean()))

    def boot(d, pid):
        ids = np.unique(pid)
        sums = np.array([d[pid == i].sum() for i in ids])
        cnt = np.array([(pid == i).sum() for i in ids])
        bs = []
        for _ in range(1000):
            j = rng.integers(0, len(ids), len(ids))
            bs.append(sums[j].sum() / cnt[j].sum())
        return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def status(R, B):
        if np.median(B) >= 0.5:
            return "DEAD"
        ok = np.median(R) <= 0.001 * V and np.percentile(R, 99) <= 0.01 * V and np.percentile(B, 99) < 0.5
        return "GATES_PASS" if ok else "UNDER_HALF"

    res = dict(controls={}, table={})
    for mode in ("greedy", "t1"):
        arr, rows, st = {}, [], []
        for f in sorted(glob.glob(f"{rdir}/{mode}_*.npz")):
            z = np.load(f)
            j = json.load(open(f[:-4] + ".json"))
            rows += j["rows"]
            st.append(j["stats"])
            for k in z.files:
                arr.setdefault(k, []).append(z[k])
        arr = {k: np.concatenate(x) for k, x in arr.items()}
        pid = np.array([r[0] for r in rows])
        grp = np.array([r[1] for r in rows])
        res["controls"][mode] = dict(states=len(rows), s0_diff=sum(s["s0_diff"] for s in st),
                                     violations={k: sum(s["viol"][k] for s in st) for k in ("plain", "group")},
                                     mismatches={a: sum(v for s in st for k, v in s["mismatch"].items() if k.startswith(a + "/"))
                                                 for a in ("k2", "staged")})
        for rn, (md, _) in ALL_RULES.items():
            if md != mode:
                continue
            tab = {}
            for g in list(sorted(set(grp))) + ["all"]:
                mk = np.ones(len(grp), bool) if g == "all" else grp == g
                row = dict(states=int(mk.sum()))
                for a in ("k2", "both", "staged"):
                    B, S, R = arr[f"{a}/{rn}/bytes"][mk], arr[f"{a}/{rn}/S"][mk], arr[f"{a}/{rn}/R"][mk]
                    row[a] = dict(bytes=q(B), S=dict(median=float(np.median(S)), p99=float(np.percentile(S, 99))),
                                  R=dict(median=float(np.median(R)), p99=float(np.percentile(R, 99))), status=status(R, B))
                Sg = arr[f"staged/{rn}/Sg"][mk]
                row["staged"]["Sg"] = dict(median=float(np.median(Sg)), p99=float(np.percentile(Sg, 99)), mean=float(Sg.mean()))
                row["staged"]["bytes_kept_sum"] = q(arr[f"staged/{rn}/bytes_kept_sum"][mk])
                dlt = (arr[f"staged/{rn}/bytes"] - arr[f"k2/{rn}/bytes"])[mk]
                row["staged_minus_k2"] = dict(mean_ci=boot(dlt, pid[mk]), share_lower=float((dlt < 0).mean()),
                                              share_higher=float((dlt > 0).mean()), largest_increase=float(dlt.max()))
                tab[g] = row
            res["table"][rn] = tab
    return res


@app.local_entrypoint()
def main(smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"stage_rule.json sha256 {sha[:12]} is not logged")
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{MODEL}/rows_{mode}.jsonl")).count(b"\n")
        n = min(n, 512) if smoke else n
        jobs += [(mode, s, min(s + CHUNK, n), smoke, sha) for s in range(0, n, CHUNK)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, s0=0, viol=0, mism=0, seconds=0)
    for r in chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)
    res = collect.remote(smoke)
    res["rule_sha"] = sha
    json.dump(res, open(os.path.join(HERE, "logs", f"stage_summary{'_smoke' if smoke else ''}.json"), "w"), indent=1)
    print("wrote the summary")
