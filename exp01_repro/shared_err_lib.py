"""Shared-error ("gap-to-leader") probe: how much of the quantization error is common to many rows?

For a head copy with error matrix E = W~ - W (rows e_v), split e_v = a_v^T Phi + e''_v, where Phi [r, d] holds r
directions shared by all rows and a_v [r] is stored per row in FP16. Then exactly
    z_v = z~_v + a_v . (Phi h) + h . e''_v,
so the logit can be corrected by r exact dot products and bounded by a residual bound on e'' alone.
r = 0 is the exp01 residual bound. "mean" is Phi = mean row of E with a_v = 1 (no per-row storage).
Deciding rows against each other only ever needs differences z_v - z_w, so removing the shared part is
what a gap-to-leader bound would gain; doing it per row keeps every rule's code unchanged.

Exploratory: this reports bound widths only, no recompute sets, bytes or verdicts.
"""
from __future__ import annotations

import torch

import cert_core as cc


def decompose(E, r, chunk=16384):
    """Return (Phi [r, d] float32, A [V, r] FP16-rounded float32, E'' [V, d]) with E'' = E - A Phi exactly in FP32
    up to the rounding of that subtraction (covered by the caller's slack)."""
    V, d = E.shape
    if r == "mean":
        Phi = E.double().mean(0, keepdim=True).float()
        A = torch.ones(V, 1, device=E.device)
    else:
        C = torch.zeros(d, d, dtype=torch.float64, device=E.device)
        for i in range(0, V, chunk):
            x = E[i:i + chunk].double()
            C += x.T @ x
        _, vecs = torch.linalg.eigh(C)
        Phi = vecs[:, -r:].T.float().contiguous()               # top-r directions, orthonormal
        A = (E @ Phi.T).half().float()                          # stored per row in FP16
    return Phi, A, E - A @ Phi


def energy_share(E, Phi, A):
    return float(((A @ Phi).double().norm() ** 2) / (E.double().norm() ** 2))


def res_bound(Er, h, G=128):
    """sum_g ||e_vg|| ||h_g||, [B, V]."""
    V, d = Er.shape
    en = Er.view(V, d // G, G).norm(dim=-1)
    hn = h.view(h.shape[0], d // G, G).norm(dim=-1)
    return hn @ en.T


def probe(h, variants, ranks=("mean", 1, 4, 16), G=128):
    """For each head copy: energy share of the shared part and the median over states of
    (mean over rows, max over rows) of the shared-error residual bound relative to the plain residual bound.
    Also the Hoeffding bound for the dithered copy, for comparison with the bound it actually certifies with."""
    out = {}
    for name, (Wb, v) in variants.items():   # Wb: the matrix this copy quantized (W, or W_t for a shifted copy)
        E = v.Wq - Wb
        hm = v.map_h(h)
        b0 = res_bound(E, hm, G)
        rec = {"res_mean_abs": float(b0.mean(1).median()), "res_max_abs": float(b0.amax(1).median())}
        if hasattr(v, "dither"):
            bh = cc.bounds_for(v, h, Wb.norm(dim=1))[0]["hoeff"]
            rec["hoeff_over_res_mean"] = float((bh.mean(1) / b0.mean(1)).median())
        for r in ranks:
            Phi, A, Er = decompose(E, r)
            b = res_bound(Er, hm, G)
            err_ok = ((hm @ E.T) - (hm @ Phi.T) @ A.T - hm @ Er.T).abs().max().item()   # identity check
            rec[f"r{r}"] = dict(energy_share=energy_share(E, Phi, A),
                                mean_ratio=float((b.mean(1) / b0.mean(1)).median()),
                                max_ratio=float((b.amax(1) / b0.amax(1)).median()),
                                identity_err=err_ok)
            del Er
        out[name] = rec
        del E
    return out
