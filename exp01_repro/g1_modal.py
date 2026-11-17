"""G1 on Modal: the method's parts on the other off-floor models (g1_rule.json).

  modal run g1_modal.py --smoke        # 1,024 states per mode
  modal run g1_modal.py                # Sarvam-1, SmolLM3-3B, Phi-4-mini
Writes logs/g1_<model>[_smoke].json (numbers only). Per-state arrays stay on the volume (g1/<model>/).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "g1_rule.json")
app = modal.App("exp01-g1")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "subg_lib.py", "hdir_lib.py", "mix_lib.py", "rules2_lib.py", "pipe_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

MODELS = ("sarvam_1", "smollm3_3b", "phi4_mini")
SHIFTS, KS, BATCH = (0.0, 0.5, 1.0, 1.5), (1, 4, 16), 128
RULES = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
}


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=4 * 3600)
def run(model: str, smoke: bool, rule_sha: str):
    import time

    import numpy as np
    import torch

    import a3_lib
    import cert_core as cc
    import hdir_lib
    import mix_lib
    import pipe_lib

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sdir = f"{VOL}/states/{model}"
    meta = json.load(open(f"{sdir}/meta.json"))
    cap = meta["cap"]
    W = torch.load(f"{sdir}/W.pt").cuda().float()
    V, d = W.shape
    ng = d // 128
    mu64 = a3_lib.mean_row(W)
    w_norm = W.norm(dim=1)
    data = {}
    for mode in ("greedy", "t1"):
        H = torch.load(f"{sdir}/h_{mode}.pt")
        rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")]
        if smoke:
            H, rows = H[:1024], rows[:1024]
        data[mode] = (H, rows)
    Hg, rg = data["greedy"]
    even = [i for i, r in enumerate(rg) if r["pid"] % 2 == 0]
    He, re_ = Hg[even], [rg[i] for i in even]
    greedy = {"greedy": dict(rule="greedy", cap=cap)}

    def sweep(H, rows, rules, src, arms):
        out, stats = {}, dict(viol={}, mismatch={})
        for c0 in range(0, len(rows), BATCH):
            rec, st = pipe_lib.run_chunk(W, mu64, w_norm, H[c0:c0 + BATCH].cuda().float(), rows[c0:c0 + BATCH], rules, src, arms)
            for k, x in rec.items():
                out.setdefault(k, []).append(x)
            for f in ("viol", "mismatch"):
                for k, x in st[f].items():
                    stats[f][k] = stats[f].get(k, 0) + x
        return {k: torch.cat(x).numpy() for k, x in out.items()}, stats

    sel = dict(even_states=len(re_), shift={}, k={}, groups={})
    # 1. the shift, with no correction
    for t in SHIFTS:
        vs, Wt = pipe_lib.make_variants(W, mu64, t)
        o, _ = sweep(He, re_, greedy, {"f": dict(v=vs["i4"], kind="subg", sets=None), "i8": dict(v=vs["i8"], kind="det", row_bytes=d + 4)},
                     {"a": dict(first="f", stages=["i8"])})
        sel["shift"][f"{t:g}"] = float(o["a/greedy/bytes"].mean())
        del vs, Wt
        torch.cuda.empty_cache()
    t_star = min(SHIFTS, key=lambda t: sel["shift"][f"{t:g}"])
    # 2. the number of directions
    Pall, info = hdir_lib.fit_directions(mu64, He.cuda().float(), kmax=16)
    P16 = Pall[:, 1:17]
    vs, Wt = pipe_lib.make_variants(W, mu64, t_star, dith8=True)
    r16 = hdir_lib.err_proj(vs["i4"].Wq, Wt, P16)
    i8src = dict(v=vs["i8"], kind="det", row_bytes=d + 4)
    sel["k"]["0"] = sel["shift"][f"{t_star:g}"]
    for K in KS:
        o, _ = sweep(He, re_, greedy, {"f": dict(v=vs["i4"], kind="subg", sets=(P16[:, :K], r16[:, :K])), "i8": i8src},
                     {"a": dict(first="f", stages=["i8"], first_extra=4 * K)})
        sel["k"][str(K)] = float(o["a/greedy/bytes"].mean())
    k_star = int(min(sel["k"], key=lambda k: sel["k"][k]))
    Pk = P16[:, :k_star] if k_star else None
    first_sets = (Pk, r16[:, :k_star]) if k_star else None
    # 3. staged groups, chosen by the mean share of the variance term after the directions are removed
    S = mix_lib.variance_weights(vs["i4"])
    Hd = He.cuda().double()
    if k_star:
        Hd = Hd - (Hd @ Pk) @ Pk.T
    e = (Hd ** 2).view(Hd.shape[0], ng, 128).sum(-1) * S[None, :]
    share = (e / e.sum(1, keepdim=True)).mean(0)
    order = share.argsort(descending=True)[:2].tolist()
    sel["group_shares"] = dict(largest=[int(order[0]), float(share[order[0]])], second=[int(order[1]), float(share[order[1]])],
                               uniform=1.0 / ng)
    del Hd, e
    sel["groups"]["none"] = sel["k"][str(k_star)]
    mixed = {}
    for G in (tuple(order[:1]), tuple(order[:2])):
        m = cc.prepare(mix_lib.make_mixed_dither(Wt, G))
        m.t, m.wt_norm = t_star, vs["i4"].wt_norm
        rm = hdir_lib.err_proj(m.Wq, Wt, Pk) if k_star else None
        gb = len(G) * (128 + 2 + 64 + 2) + 4 + 4 * k_star
        mixed[G] = (m, rm, gb)
        o, _ = sweep(He, re_, greedy, {"f": dict(v=vs["i4"], kind="subg", sets=first_sets), "i8": i8src,
                                       "G": dict(v=m, kind="subg", sets=(Pk, rm) if k_star else None, row_bytes=gb)},
                     {"a": dict(first="f", stages=["G", "i8"], first_extra=4 * k_star)})
        sel["groups"][",".join(map(str, G))] = float(o["a/greedy/bytes"].mean())
    g_key = min(sel["groups"], key=lambda k: sel["groups"][k])
    G_star = tuple(int(x) for x in g_key.split(",")) if g_key != "none" else ()
    sel.update(t_star=t_star, k_star=k_star, groups_star=list(G_star), top1_share=info["share"][0], top4_share=info["share"][1],
               top16_share=info["share"][2])
    # final arms, every state, every rule
    v0, W0 = pipe_lib.make_variants(W, mu64, 0.0)
    rd8 = hdir_lib.err_proj(vs["d8"].Wq, Wt, Pk) if k_star else None
    src = {"plain": dict(v=v0["i4"], kind="subg", sets=None), "i8_0": dict(v=v0["i8"], kind="det", row_bytes=d + 4),
           "shift": dict(v=vs["i4"], kind="subg", sets=None), "k": dict(v=vs["i4"], kind="subg", sets=first_sets), "i8": i8src,
           "d8": dict(v=vs["d8"], kind="subg", sets=(Pk, rd8) if k_star else None, row_bytes=d + 2 * ng + 4 + 4 * k_star)}
    pre = []
    if G_star:
        m, rm, gb = mixed[G_star]
        src["G"] = dict(v=m, kind="subg", sets=(Pk, rm) if k_star else None, row_bytes=gb)
        pre = ["G"]
    for G, (m, rm, gb) in list(mixed.items()):
        if G != G_star:
            del mixed[G]
    del Wt, W0
    torch.cuda.empty_cache()
    fx = 4 * k_star
    arms = {"plain": dict(first="plain", stages=["i8_0"]), "shift": dict(first="shift", stages=["i8"]),
            "k": dict(first="k", stages=["i8"], first_extra=fx), "staged": dict(first="k", stages=pre + ["i8"], first_extra=fx),
            "full": dict(first="k", stages=pre + ["d8"], first_extra=fx)}
    res = dict(model=model, meta=meta, V=V, d=d, selection=sel, controls={}, table={}, table_all_states={})
    rdir = f"{VOL}/g1/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    for mode, (H, rows) in data.items():
        rules = {k: dict(c, cap=cap) for k, (md, c) in RULES.items() if md == mode}
        o, st = sweep(H, rows, rules, src, arms)
        np.savez_compressed(f"{rdir}/{mode}.npz", **{k: x.astype(np.float32) for k, x in o.items()})
        odd = np.array([r["pid"] % 2 == 1 for r in rows])
        res["controls"][mode] = dict(states=len(rows), odd_states=int(odd.sum()), violations=sum(st["viol"].values()),
                                     mismatches=sum(st["mismatch"].values()))
        for rn in rules:
            for name, mk in (("table", odd), ("table_all_states", np.ones(len(rows), bool))):
                tab = {}
                for a in arms:
                    B, R = o[f"{a}/{rn}/bytes"][mk], o[f"{a}/{rn}/R"][mk]
                    ok = np.median(R) <= 0.001 * V and np.percentile(R, 99) <= 0.01 * V and np.percentile(B, 99) < 0.5
                    tab[a] = dict(bytes=dict(median=float(np.median(B)), p99=float(np.percentile(B, 99)), mean=float(B.mean())),
                                  R=dict(median=float(np.median(R)), p99=float(np.percentile(R, 99))),
                                  status="DEAD" if np.median(B) >= 0.5 else "GATES_PASS" if ok else "UNDER_HALF")
                res[name][rn] = tab
        if mode == "greedy":
            res["bound_vs_plain"] = {s: float(np.median(o[f"bmed/{s}"] / o["bmed/plain"])) for s in ("shift", "k")}
            res["d8_vs_i8_bound"] = float(np.median(o["bmed/d8"] / o["bmed/i8"]))
    res.update(rule_sha=rule_sha, seconds=round(time.time() - t0), gpu=torch.cuda.get_device_name(0))
    json.dump(res, open(f"{rdir}/summary.json", "w"))
    vol.commit()
    return res


@app.local_entrypoint()
def main(smoke: bool = False, models: str = ",".join(MODELS)):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"g1_rule.json sha256 {sha[:12]} is not logged")
    for r in run.starmap([(m, smoke, sha) for m in models.split(",")], order_outputs=False, return_exceptions=True):
        if not isinstance(r, dict):
            print(f"FAILED {r!r}"[:800], flush=True)
            continue
        json.dump(r, open(os.path.join(HERE, "logs", f"g1_{r['model']}{'_smoke' if smoke else ''}.json"), "w"), indent=1)
        s, g = r["selection"], r["table"]["greedy"]
        print(r["model"], "selected t", s["t_star"], "K", s["k_star"], "groups", s["groups_star"], "| viol/mism",
              {m: (c["violations"], c["mismatches"]) for m, c in r["controls"].items()}, "| greedy odd (median, p99, mean):",
              {a: tuple(round(g[a]["bytes"][k], 3) for k in ("median", "p99", "mean")) for a in g}, f"| {r['seconds']}s", flush=True)
