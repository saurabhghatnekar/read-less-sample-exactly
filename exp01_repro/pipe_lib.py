"""A certificate assembled from named interval sources, for the runs that compare several pipelines on the same states
(T1 rules and stages, R1 prompts and seeds, G1 other models).

  source  an interval set for every row of every state: (variant, kind, sets, bytes charged per row read)
            kind "subg"  dithered copy with the sub-Gaussian bound, with K2's correction if sets is given
            kind "det"   the worst-case bound of the INT8-row copy (cert_core's wc and res, the smaller)
            sets         None, (P, r), or {language group: (P, r)}
  arm     first = a source read for every row; stages = sources read only for the rows the rule still needs
"""
from __future__ import annotations

import torch

import a3_lib
import cert_core as cc
import hdir_lib
import rules2_lib
import subg_lib


def make_variants(W, mu64, t, seed=1234, groups8=(), dith8=False):
    """Copies of W_t: dithered INT4 g128 ('i4'), INT8 rows with the worst-case bound ('i8'), and on request the mixed
    head ('mix', groups8 in INT8) and a dithered INT8 g128 copy ('d8'). All use the same dither seed."""
    import mix_lib

    Wt = a3_lib.shifted(W, mu64, t)
    wt_norm = Wt.norm(dim=1)
    vs = {"i4": cc.make_int4_dither(Wt, seed=seed), "i8": cc.make_int8_row(Wt)}
    if groups8:
        vs["mix"] = mix_lib.make_mixed_dither(Wt, tuple(groups8), seed=seed)
    if dith8:
        vs["d8"] = mix_lib.make_mixed_dither(Wt, tuple(range(W.shape[1] // 128)), seed=seed)
    for v in vs.values():
        cc.prepare(v)
        v.t, v.wt_norm = t, wt_norm
    return vs, Wt


def intervals(v, kind, sets, h, w_norm, mu64, groups):
    zt, bd, slack = a3_lib.intervals(v, h, w_norm, mu64)
    if kind == "det":
        return zt, torch.minimum(bd["wc"], bd["res"])
    if sets is None:
        return zt, subg_lib.subg_bound(v, h, slack)
    if not isinstance(sets, dict):
        ztc, b, _ = hdir_lib.corrected(v, sets[0], sets[1], h, zt, slack)
        return ztc, b
    ztc, b = torch.empty_like(zt), torch.empty_like(zt)
    for gname in sorted(set(groups)):
        idx = torch.tensor([i for i, x in enumerate(groups) if x == gname], device=h.device)
        P, r = sets[gname]
        ztc[idx], b[idx], _ = hdir_lib.corrected(v, P, r, h[idx], zt[idx], slack[idx])
    return ztc, b


def run_chunk(W, mu64, w_norm, h, rows, rules, sources, arms, noise_mode=None):
    """sources: {name: dict(v=variant, kind=, sets=, row_bytes=)}. arms: {name: dict(first=source name,
    stages=[source names], first_extra=bytes per row added to the first pass)}.
    Returns per-state records {arm/rule/quantity} and stats. noise_mode overrides the mode in the noise key."""
    V, d = W.shape
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], noise_mode or r["mode"]) for r in rows], dtype=torch.int64,
                        device=W.device)
    g = cc.gumbel(keys, V, W.device)
    groups = [r["group"] for r in rows]
    iv, stats = {}, dict(viol={}, mismatch={})
    for name, s in sources.items():
        iv[name] = intervals(s["v"], s["kind"], s.get("sets"), h, w_norm, mu64, groups)
        stats["viol"][name] = int(((z_ref - iv[name][0]).abs() > iv[name][1]).sum())
    rec = {}
    for name in sources:
        rec[f"bmed/{name}"] = iv[name][1].median(1).values
    ref = {rn: rules2_lib.decide(z_ref, cfg, g) for rn, cfg in rules.items()}
    for an, arm in arms.items():
        f = sources[arm["first"]]
        first_bytes = (f["v"].bytes_per_weight * d + arm.get("first_extra", 0.0)) * V
        for rn, cfg in rules.items():
            reads, R, tok = rules2_lib.certify(z_ref, iv[arm["first"]][0], iv[arm["first"]][1], cfg, g,
                                               [iv[s] for s in arm["stages"]])
            Rc = R.sum(1).float()
            total = first_bytes + Rc * 2.0 * d
            for s, S in zip(arm["stages"], reads):
                Sc = S.sum(1).float()
                rec[f"{an}/{rn}/S_{s}"] = Sc
                total = total + Sc * sources[s]["row_bytes"]
            rec[f"{an}/{rn}/R"] = Rc
            rec[f"{an}/{rn}/bytes"] = total / (2.0 * V * d)
            stats["mismatch"][f"{an}/{rn}"] = int((tok != ref[rn]).sum())
    return {k: x.float().cpu() for k, x in rec.items()}, stats
