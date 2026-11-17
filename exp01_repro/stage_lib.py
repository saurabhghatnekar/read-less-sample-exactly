"""M2: read the heavy group in INT8 only for the rows the rule still needs (stage_rule.json).

Stages: (1) the plain dithered INT4 head with K2's correction, every row; (G) for the rows the rule needs, the
row's group in INT8, which turns the row's interval into the mixed head's (mix_lib.make_mixed_dither: the two
heads differ only in that group and share the dither); (2) INT8 rows; (3) BF16 rows.
"""
from __future__ import annotations

import torch

import a3_lib
import cert_core as cc
import hdir_lib

# Bytes charged per row read in stage G, all counted as read from the head:
#   128 INT8 codes of the group + 2 its FP16 step + 4 the row's K2 number for the mixed copy + 4 the mixed copy's
#   row norm (FP32 slack) + 64 + 2 re-reading the row's INT4 codes and step for that group, to take them out.
STAGE_G_BYTES = 128 + 2 + 4 + 4 + 64 + 2
STAGE_G_BYTES_KEPT_SUM = 128 + 2 + 4 + 4      # if the first pass keeps each row's partial sum for the group


def certify_staged(z_ref, zt, b, cfg, g, stage_g, stage2):
    """Like cert_core.certify with one more stage before the INT8 rows. Returns (Sg, S, R, token)."""
    L, U = zt - b, zt + b
    zg, bg = stage_g
    L, U, Sg = cc.refine_until_stable(L, U, cfg, g, zg - bg, zg + bg)
    z8, b8 = stage2
    L, U, S = cc.refine_until_stable(L, U, cfg, g, z8 - b8, z8 + b8)
    L, U, R = cc.refine_until_stable(L, U, cfg, g, z_ref, z_ref)
    tok = cc.decide(torch.where(R, z_ref, torch.full_like(z_ref, -float("inf"))), cfg, g) \
        if cfg["rule"] in ("greedy", "gumbel") else cc._decide_with_intervals(z_ref, R, L, U, cfg, g)
    return Sg, S, R, tok


def _corrected_by_group(v, sets, h, zt, slack, groups):
    ztc, b = torch.empty_like(zt), torch.empty_like(zt)
    for gname in sorted(set(groups)):
        idx = torch.tensor([i for i, x in enumerate(groups) if x == gname], device=h.device)
        P, r = sets[gname]
        ztc[idx], b[idx], _ = hdir_lib.corrected(v, P, r, h[idx], zt[idx], slack[idx])
    return ztc, b


def run_chunk_staged(W, mu64, w_norm, v, m, v8, sets_plain, sets_mixed, h, rows, rules):
    """v: plain dithered INT4 variant, m: mixed variant, v8: INT8-row variant (all of W_t, prepared as a3_lib.build
    does). sets_*: {language group: (P [d, K] float64, r [V, K] float32)} for each head."""
    V, d = W.shape
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64, device=W.device)
    g = cc.gumbel(keys, V, W.device)
    groups = [r["group"] for r in rows]
    K = next(iter(sets_plain.values()))[0].shape[1]
    z8, b8all, _ = a3_lib.intervals(v8, h, w_norm, mu64)
    b8 = torch.minimum(b8all["wc"], b8all["res"])
    zt, _, slack = a3_lib.intervals(v, h, w_norm, mu64)
    ztc, b = _corrected_by_group(v, sets_plain, h, zt, slack, groups)
    del zt, slack
    zm, _, slack_m = a3_lib.intervals(m, h, w_norm, mu64)
    zg, bg = _corrected_by_group(m, sets_mixed, h, zm, slack_m, groups)
    del zm, slack_m
    rec, stats = {}, dict(viol={}, mismatch={})
    stats["viol"]["plain"] = int(((z_ref - ztc).abs() > b).sum())
    stats["viol"]["group"] = int(((z_ref - zg).abs() > bg).sum())
    base = lambda S, R: cc.bytes_fraction(V, d, v.bytes_per_weight, 0.0, d // v.G, S, R, d + 4) + 2.0 * K / d
    for rn, cfg in rules.items():
        ref = cc.decide(z_ref, cfg, g)
        S, R, tok = cc.certify(z_ref, ztc, b, cfg, g, (z8, b8))
        Sc, Rc = S.sum(1).float(), R.sum(1).float()
        rec[f"k2/{rn}/S"], rec[f"k2/{rn}/R"], rec[f"k2/{rn}/bytes"] = Sc, Rc, base(Sc, Rc)
        stats["mismatch"][f"k2/{rn}"] = int((tok != ref).sum())
        Sg, S, R, tok = certify_staged(z_ref, ztc, b, cfg, g, (zg, bg), (z8, b8))
        Gc, Sc, Rc = Sg.sum(1).float(), S.sum(1).float(), R.sum(1).float()
        rec[f"staged/{rn}/Sg"], rec[f"staged/{rn}/S"], rec[f"staged/{rn}/R"] = Gc, Sc, Rc
        rec[f"staged/{rn}/bytes"] = base(Sc, Rc) + Gc * STAGE_G_BYTES / (2.0 * V * d)
        rec[f"staged/{rn}/bytes_kept_sum"] = base(Sc, Rc) + Gc * STAGE_G_BYTES_KEPT_SUM / (2.0 * V * d)
        stats["mismatch"][f"staged/{rn}"] = int((tok != ref).sum())
    return {k: x.float().cpu() for k, x in rec.items()}, stats
