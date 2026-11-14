"""B1: sub-Gaussian bound for the dithered INT4 head (subg_rule.json).

b_subg = sqrt(ln(2/delta)/6) * sqrt(sum_g ||h_g||^2 s_vg^2) + slack, i.e. 1/sqrt(3) of the Hoeffding term.
Uses a3_lib's shifted copies and intervals, so the slack and the add-back are identical to A3.
"""
from __future__ import annotations

import math

import torch

import a3_lib
import cert_core as cc

T_GRID = (0.0, 1.0, 0.5, 1.5)
DELTA = 1e-15


def subg_c(delta=DELTA):
    return math.sqrt(math.log(2.0 / delta) / 6.0)


def subg_bound(v, h, slack, delta=DELTA):
    hm = v.map_h(h)
    B, d = hm.shape
    h2 = hm.view(B, d // v.G, v.G).norm(dim=-1)
    return subg_c(delta) * torch.sqrt((h2 ** 2) @ (v.scale ** 2).T) + slack


def run_chunk(W, mu64, h, rows, rules, grid=T_GRID):
    V, d = W.shape
    device = W.device
    w_norm = W.norm(dim=1)
    Bc = h.shape[0]
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64, device=device)
    g = cc.gumbel(keys, V, device)
    ref_toks = {rn: cc.decide(z_ref, rcfg, g) for rn, rcfg in rules.items()}
    rec, stats = {}, dict(viol={}, viol_excess_max={}, mismatch={}, dither_se={}, dither_n={})
    for t in grid:
        tag = f"t{t:g}"
        vs = a3_lib.build(W, mu64, t)
        z8, b8all, _ = a3_lib.intervals(vs["int8_row"], h, w_norm, mu64)
        b8 = torch.minimum(b8all["wc"], b8all["res"])
        v = vs["int4d_g128"]
        zt, bounds, slack = a3_lib.intervals(v, h, w_norm, mu64)
        bs = subg_bound(v, h, slack)
        err = (z_ref - zt).abs()
        over = err - bs
        stats["viol"][tag] = int((over > 0).sum())
        stats["viol_excess_max"][tag] = float(over.max())
        hg = h.view(Bc, d // v.G, v.G)
        var = (hg.norm(dim=-1) ** 2) @ (v.scale ** 2).T / 12
        stats["dither_se"][tag] = float(((z_ref - zt) ** 2 / var).sum())
        stats["dither_n"][tag] = var.numel()
        rec[f"{tag}/int4d_g128/subg/meanbound"] = bs.mean(1)
        rec[f"{tag}/int4d_g128/hoeff/meanbound"] = bounds["hoeff"].mean(1)
        rec[f"{tag}/int4d_g128/maxerr"] = err.amax(1)
        for bn, b, stages in (("subg", bs, (None, (z8, b8))), ("hoeff", bounds["hoeff"], ((z8, b8),))):
            for st in stages:
                for rn, rcfg in rules.items():
                    S, R, tok = cc.certify(z_ref, zt, b, rcfg, g, st)
                    key = f"{tag}/int4d_g128/{bn}/{'casc' if st else 'single'}/{rn}"
                    Sc = None if S is None else S.sum(1).float()
                    Rc = R.sum(1).float()
                    rec[f"{key}/S"] = Sc if Sc is not None else torch.zeros_like(Rc)
                    rec[f"{key}/R"] = Rc
                    rec[f"{key}/bytes"] = cc.bytes_fraction(V, d, v.bytes_per_weight, 0.0, d // v.G, Sc, Rc, d + 4)
                    stats["mismatch"][key] = stats["mismatch"].get(key, 0) + int((tok != ref_toks[rn]).sum())
        del vs, z8, b8, b8all, zt, bounds, bs, err, over
    return {k: x.float().cpu() for k, x in rec.items()}, stats
