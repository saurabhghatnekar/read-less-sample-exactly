"""M1 on Modal: one group in INT8 for every row, with and without the K2 correction (mix_rule.json). Sarvam-30B.

  modal run mix_modal.py --smoke        # first 512 states per mode; prints seconds per job for a cost estimate
  modal run mix_modal.py                # all states, then the per-language table
Writes logs/mix_summary[_smoke].json (numbers only). Per-state arrays stay on the volume (mix/sarvam_30b/).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "mix_rule.json")
app = modal.App("exp01-mix")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "hdir_lib.py", "mix_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

MODEL, T, GROUP8, CHUNK = "sarvam_30b", 0.5, 26, 4096
ALL_RULES = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_0.6": ("t1", dict(rule="gumbel", T=0.6)),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
}
ARMS = {"none": "plain/none", "k2": "plain/k2", "int8g": "mixed/none", "both": "mixed/k2", "k2_g26dir": "plain/k2_g26dir"}


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3600)
def prep(rule_sha: str):
    """Control M0_group, and for each language a direction inside group 26 fitted on the other languages."""
    import torch

    import a3_lib
    import mix_lib
    from a3_modal import MODELS, _load_head

    sdir = f"{VOL}/states/{MODEL}"
    H = torch.load(f"{sdir}/h_greedy.pt").cuda().float()
    rows = [json.loads(l) for l in open(f"{sdir}/rows_greedy.jsonl")]
    grp = [r["group"] for r in rows]
    pid = torch.tensor([r["pid"] for r in rows], device="cuda")
    W = _load_head(MODELS[MODEL]).cuda().float()
    mu64 = a3_lib.mean_row(W)
    v = a3_lib.build(W, mu64, T)["int4d_g128"]
    S = mix_lib.variance_weights(v)
    del v, W
    torch.cuda.empty_cache()
    info = dict(group=GROUP8, by_language={}, direction={})
    for gname in sorted(set(grp)):
        mk = torch.tensor([x == gname for x in grp], device="cuda") & (pid % 2 == 0)
        arg, top = mix_lib.largest_group(H[mk], S)
        mode = int(arg.mode().values)
        info["by_language"][gname] = dict(states=int(mk.sum()), most_common_largest_group=mode,
                                          how_often=float((arg == mode).float().mean()),
                                          group26_is_largest=float((arg == GROUP8).float().mean()),
                                          largest_share_median=float(top.median()))
    mk = torch.tensor([x in ("tamil", "telugu") for x in grp], device="cuda") & (pid % 2 == 0)
    arg, _ = mix_lib.largest_group(H[mk], S)
    info["tamil_telugu_even_most_common_largest_group"] = int(arg.mode().values)
    Pl = {k: x.cuda() for k, x in torch.load(f"{VOL}/hdir/{MODEL}/P_lolo.pt").items()}
    Q = {}
    for gname in sorted(Pl):
        keep = torch.tensor([x != gname for x in grp], device="cuda")
        q, share, inside = mix_lib.group_direction(H[keep], Pl[gname][:, 0], GROUP8)
        Q[gname] = q.cpu()
        info["direction"][gname] = dict(share_of_group_energy=share, energy_inside_group=inside)
    names = sorted(Q)
    info["direction_cosines"] = {f"{a}/{b}": float((Q[a] @ Q[b]).abs()) for i, a in enumerate(names) for b in names[i + 1:]}
    os.makedirs(f"{VOL}/mix/{MODEL}", exist_ok=True)
    torch.save(Q, f"{VOL}/mix/{MODEL}/Q_lolo.pt")
    json.dump(dict(info, rule_sha=rule_sha), open(f"{VOL}/mix/{MODEL}/prep.json", "w"))
    vol.commit()
    return info


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=4 * 3600)
def chunk(mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import time

    import numpy as np
    import torch

    import a3_lib
    import cert_core as cc
    import hdir_lib
    import mix_lib
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
    Ql = {k: x.cuda()[:, None] for k, x in torch.load(f"{VOL}/mix/{MODEL}/Q_lolo.pt").items()}
    gnames = sorted(Pl)
    Pcat = torch.cat([torch.cat([Pl[k], Ql[k]], 1) for k in gnames], 1)      # per language: K2's direction, then q
    col = {k: 2 * i for i, k in enumerate(gnames)}
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in ALL_RULES.items() if md == mode}
    vs = a3_lib.build(W, mu64, T)
    del vs["int4_g128"]
    Wt = a3_lib.shifted(W, mu64, T)
    out, stats = {}, dict(viol={}, mismatch={}, viol_excess_max={})

    def run(head, v, sets):
        for c0 in range(0, len(rows), 128):
            rec, st = hdir_lib.run_chunk_sets(W, mu64, w_norm, {"int4d_g128": v, "int8_row": vs["int8_row"]}, sets,
                                              H[c0:c0 + 128].cuda().float(), rows[c0:c0 + 128], rules)
            for k, x in rec.items():
                out.setdefault(f"{head}/" + k.split("/", 1)[1], []).append(x)
            for f in ("viol", "mismatch"):
                for k, x in st[f].items():
                    kk = f"{head}/" + k.split("/", 1)[1]
                    stats[f][kk] = stats[f].get(kk, 0) + x
            for k, x in st["viol_excess_max"].items():
                kk = f"{head}/" + k.split("/", 1)[1]
                stats["viol_excess_max"][kk] = max(stats["viol_excess_max"].get(kk, -1e30), x)

    v = vs["int4d_g128"]
    r = hdir_lib.err_proj(v.Wq, Wt, Pcat)
    run("plain", v, {"k2": {k: (Pl[k], r[:, col[k]:col[k] + 1]) for k in gnames},
                     "k2_g26dir": {k: (Pcat[:, col[k]:col[k] + 2], r[:, col[k]:col[k] + 2]) for k in gnames}})
    wt_norm = v.wt_norm
    del v, r, vs["int4d_g128"]
    torch.cuda.empty_cache()
    m = cc.prepare(mix_lib.make_mixed_dither(Wt, (GROUP8,)))
    m.t, m.wt_norm = T, wt_norm
    r = hdir_lib.err_proj(m.Wq, Wt, Pcat)
    run("mixed", m, {"k2": {k: (Pl[k], r[:, col[k]:col[k] + 1]) for k in gnames}})
    out = {k: torch.cat(x).numpy() for k, x in out.items()}
    n = len(rows)
    mine = [(r_["pid"], r_["group"], r_["step"]) for r_ in rows]

    # control M0_reproduce: none and k2 are the K2 addendum's stored t0.5/none and t0.5/lolo1
    f = f"{VOL}/hdir_robust/{MODEL}/{mode}_{start:06d}"
    assert [tuple(x) for x in json.load(open(f + ".json"))["rows"]][:n] == mine, "state order differs from the K2 addendum"
    old = np.load(f + ".npz")
    m0 = 0
    for rn in rules:
        for qn in ("S", "bytes"):
            m0 += int((old[f"t{T:g}/none/{rn}/{qn}"][:n] != out[f"plain/none/{rn}/{qn}"]).sum())
            m0 += int((old[f"t{T:g}/lolo1/{rn}/{qn}"][:n] != out[f"plain/k2/{rn}/{qn}"]).sum())
    stats["m0_diff"] = m0
    rdir = f"{VOL}/mix/{MODEL}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: x.astype(np.float32) for k, x in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, rows=mine), open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(mode=mode, start=start, n=n, m0=m0, viol=sum(stats["viol"].values()), mism=sum(stats["mismatch"].values()),
                seconds=round(time.time() - t0),
                mean_bytes={rn: {a: round(float(out[f"{k}/{rn}/bytes"].mean()), 4) for a, k in ARMS.items()} for rn in rules})


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=16 * 1024, timeout=3600)
def collect(smoke: bool):
    """Per-language tables from the per-state arrays on the volume."""
    import glob

    import numpy as np

    vol.reload()
    rdir = f"{VOL}/mix/{MODEL}{'_smoke' if smoke else ''}"
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

    res = dict(prep=json.load(open(f"{VOL}/mix/{MODEL}/prep.json")), controls={}, table={})
    for mode in ("greedy", "t1"):
        files = sorted(glob.glob(f"{rdir}/{mode}_*.npz"))
        arr, rows, st = {}, [], []
        for f in files:
            z = np.load(f)
            j = json.load(open(f[:-4] + ".json"))
            rows += j["rows"]
            st.append(j["stats"])
            for k in z.files:
                arr.setdefault(k, []).append(z[k])
        arr = {k: np.concatenate(x) for k, x in arr.items()}
        pid = np.array([r[0] for r in rows])
        grp = np.array([r[1] for r in rows])
        res["controls"][mode] = dict(states=len(rows), m0_diff=sum(s["m0_diff"] for s in st),
                                     violations={a: sum(v for s in st for kk, v in s["viol"].items() if kk == k) for a, k in ARMS.items()},
                                     largest_excess=max(v for s in st for v in s["viol_excess_max"].values()),
                                     mismatches=sum(v for s in st for v in s["mismatch"].values()))
        for rn, (md, _) in ALL_RULES.items():
            if md != mode:
                continue
            tab = {}
            for g in list(sorted(set(grp))) + ["all"]:
                mk = np.ones(len(grp), bool) if g == "all" else grp == g
                row = dict(states=int(mk.sum()))
                for a, k in ARMS.items():
                    B, S = arr[f"{k}/{rn}/bytes"][mk], arr[f"{k}/{rn}/S"][mk]
                    row[a] = dict(bytes=q(B), S=dict(median=float(np.median(S)), p99=float(np.percentile(S, 99))),
                                  bound_vs_none=float(np.median(arr[f"{k}/b_med"][mk] / arr["plain/none/b_med"][mk])))
                for name, (a, b) in (("both_minus_k2", ("both", "k2")), ("int8g_minus_none", ("int8g", "none")),
                                     ("k2_g26dir_minus_k2", ("k2_g26dir", "k2")), ("both_minus_int8g", ("both", "int8g"))):
                    dlt = (arr[f"{ARMS[a]}/{rn}/bytes"] - arr[f"{ARMS[b]}/{rn}/bytes"])[mk]
                    row[name] = dict(mean_ci=boot(dlt, pid[mk]), share_of_states_lower=float((dlt < 0).mean()))
                tab[g] = row
            res["table"][rn] = tab
    return res


@app.local_entrypoint()
def main(smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"mix_rule.json sha256 {sha[:12]} is not logged")
    print("prep", json.dumps(prep.remote(sha)), flush=True)
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{MODEL}/rows_{mode}.jsonl")).count(b"\n")
        n = min(n, 512) if smoke else n
        jobs += [(mode, s, min(s + CHUNK, n), smoke, sha) for s in range(0, n, CHUNK)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, m0=0, viol=0, mism=0, seconds=0)
    for r in chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)
    res = collect.remote(smoke)
    res["rule_sha"] = sha
    json.dump(res, open(os.path.join(HERE, "logs", f"mix_summary{'_smoke' if smoke else ''}.json"), "w"), indent=1)
    print("wrote the summary")
