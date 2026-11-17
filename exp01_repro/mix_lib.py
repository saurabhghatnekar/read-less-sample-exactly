"""M1: a dithered INT4 g128 head with some groups stored as dithered INT8 (mix_rule.json).

make_mixed_dither is cert_core.make_int4_dither with a per-group number of levels: a group listed in groups8 uses
the step (largest |w|) / 126.5 and codes in [-127, 127]. The dither pattern, the rounding and the bound are the
same, so every routine that takes the INT4 variant (a3_lib.intervals, subg_lib.subg_bound, hdir_lib.corrected)
takes this one unchanged; only its step in those groups and its bytes per weight differ.
"""
from __future__ import annotations

import torch

import cert_core as cc


def make_mixed_dither(W, groups8=(), G=128, seed=1234):
    V, d = W.shape
    ng = d // G
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = (torch.rand(d, generator=gen, dtype=torch.float64) - 0.5).float().to(W.device)
    is8 = torch.zeros(ng, dtype=torch.bool, device=W.device)
    is8[list(groups8)] = True
    div = torch.where(is8, 126.5, 6.5).float()[None, :, None]
    lo = torch.where(is8, -127.0, -8.0).float()[None, :, None]
    hi = torch.where(is8, 127.0, 7.0).float()[None, :, None]
    Wg = W.view(V, ng, G)
    s = (Wg.abs().amax(-1, keepdim=True) / div).half().float().clamp_min(1e-12)
    ug = u.view(1, ng, G)
    q = torch.round(Wg / s + ug)
    assert bool((q <= hi + (~is8)[None, :, None]).all()) and bool((q >= lo).all()), "dither clipped"
    q = torch.minimum(torch.maximum(q, lo), hi)
    Wq = ((q - ug) * s).view(V, d)
    emax, enorm = cc._group_stats(Wq - W, G)
    name = f"int4d_g{G}" + (f"_int8x{len(groups8)}" if len(groups8) else "")
    v = cc.Variant(name, Wq, G, emax, enorm, s.squeeze(-1), 0.5 + 2.0 / G + 0.5 * G * len(groups8) / d)
    v.dither = u
    v.groups8 = tuple(groups8)
    return v


def variance_weights(v):
    """[ng] mean over rows of the squared step: the weight of each group's energy in the first-pass variance term."""
    return (v.scale.double() ** 2).mean(0)


def largest_group(H, S, G=128):
    """For each state, the group with the largest share of sum_g ||h_g||^2 S_g, and that share."""
    e = (H.double() ** 2).view(H.shape[0], -1, G).sum(-1) * S[None, :]
    q = e / e.sum(1, keepdim=True)
    top, arg = q.max(1)
    return arg, top


def group_direction(H, p, g, G=128):
    """Top direction, inside group g, of the states after removing the unit direction p; made orthogonal to p.
    Returns (q [d] float64, share of the group's remaining energy along it, share of q's energy inside the group)."""
    Hd = H.double()
    R = Hd - (Hd @ p)[:, None] * p[None, :]
    X = R[:, g * G:(g + 1) * G]
    w, v = torch.linalg.eigh(X.T @ X)
    q = torch.zeros(Hd.shape[1], dtype=torch.float64, device=H.device)
    q[g * G:(g + 1) * G] = v[:, -1]
    q = q - (q @ p) * p
    q = q / q.norm()
    return q, float(w[-1] / w.sum()), float((q[g * G:(g + 1) * G] ** 2).sum())
