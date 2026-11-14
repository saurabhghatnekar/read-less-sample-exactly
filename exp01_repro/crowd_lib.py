"""K1: how crowded the top of the reference logits is, in logits and in units of the certificate's bound
(crowd_rule.json). Greedy states only."""
from __future__ import annotations

import torch

import a3_lib
import cert_core as cc
import subg_lib

WIDTHS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
KS = (2, 10, 100, 1000, 10000)


def run_chunk(W, mu64, w_norm, copies, h, cap):
    """copies: {"t0": dithered copy at t = 0, "ts": dithered copy at the B1 shift}. Returns per-state arrays."""
    cfg = dict(rule="greedy", cap=cap)
    z = h @ W.T
    c = cc.softcap(z, cap)
    top = c.amax(1, keepdim=True)
    ti = z.argmax(1, keepdim=True)
    srt = c.topk(min(max(KS), c.shape[1]), dim=1).values
    rec = {"top_raw": z.amax(1), "top": top[:, 0], "logit_std": c.std(1), "logit_median": c.median(1).values,
           "mu_dot_h": (h.double() @ mu64).float()}
    for k in KS:
        if k <= srt.shape[1]:
            rec[f"gap_k{k}"] = srt[:, 0] - srt[:, k - 1]
    for w in WIDTHS:
        rec[f"n_within_{w:g}"] = (c >= top - w).sum(1)
    lp = torch.log_softmax(c.double(), 1)
    rec["p_top"] = lp.amax(1).exp()
    rec["entropy"] = -(lp.exp() * lp).sum(1)
    del lp
    for tag, v in copies.items():
        zt, bounds, slack = a3_lib.intervals(v, h, w_norm, mu64)
        for bn, b in (("hoeff", bounds["hoeff"]), ("subg", subg_lib.subg_bound(v, h, slack))):
            if tag != "t0" and bn == "hoeff":
                continue
            key = f"{tag}/{bn}"
            rec[f"{key}/b_top"] = b.gather(1, ti).squeeze(1)
            rec[f"{key}/b_med"] = b.median(1).values
            rec[f"{key}/need_actual"] = cc.need_greedy(zt - b, zt + b, cfg).sum(1)
            rec[f"{key}/need_exact"] = cc.need_greedy(z - b, z + b, cfg).sum(1)
    return {k: x.float().cpu() for k, x in rec.items()}
