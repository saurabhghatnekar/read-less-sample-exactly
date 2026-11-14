"""F1: flip rate of an uncertified low-bit head against the BF16 reference (flip_rule.json).

For each state, copy and shift, the rule is applied to the copy's approximate logits with the same noise as the
reference, and the two tokens are compared. No bound is involved. Copies and the add-back are a3_lib's.
"""
from __future__ import annotations

import math

import torch

import a3_lib
import cert_core as cc

COPIES = ("int4_g128", "int4d_g128", "int8_row")


def approx(v, h, mu64):
    """The copy's logits in the original frame: the same arithmetic as a3_lib.intervals, without the bounds."""
    zt = cc.approx_logits(v, h)
    if v.t != 0:
        zt = zt + (v.t * (h.double() @ mu64)).float()[:, None]
    return zt


def score(z, cfg, g):
    c = cc.softcap(z, cfg["cap"])
    return c if cfg["rule"] == "greedy" else c / cfg["T"] + g


def kept(z, cfg):
    """Kept set of a truncation rule on logits z, as cert_core.decide computes it; None for the other rules."""
    c = cc.softcap(z, cfg["cap"])
    if cfg["rule"] == "topk":
        return c >= c.topk(cfg["k"], dim=1).values[:, -1:]
    if cfg["rule"] == "minp":
        return c >= c.amax(1, keepdim=True) + cfg["T"] * math.log(cfg["p"])
    return None


def run_chunk(W, mu64, vs, h, rows, rules):
    """vs: copies from a3_lib.build(W, mu64, t) (their .t gives the shift). Returns per-state arrays."""
    V = W.shape[0]
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64,
                        device=W.device)
    g = cc.gumbel(keys, V, W.device)
    t = next(iter(vs.values())).t
    tag = f"t{t:g}"
    rec = {}
    ref = {}
    for rn, cfg in rules.items():
        tok = cc.decide(z_ref, cfg, g)
        ref[rn] = (tok, score(z_ref, cfg, g), kept(z_ref, cfg))
        rec[f"ref/{rn}"] = tok
    for vn in COPIES:
        zt = approx(vs[vn], h, mu64)
        rec[f"{tag}/{vn}/maxerr"] = (z_ref - zt).abs().amax(1)
        for rn, cfg in rules.items():
            rtok, rscore, rkept = ref[rn]
            tok = cc.decide(zt, cfg, g)
            flip = tok != rtok
            key = f"{tag}/{vn}/{rn}"
            rec[f"{key}/flip"] = flip
            if rkept is None:
                gap = rscore.gather(1, rtok[:, None]) - rscore.gather(1, tok[:, None])
                rec[f"{key}/gap"] = gap.squeeze(1)
            else:
                rec[f"{key}/setdiff"] = (kept(zt, cfg) ^ rkept).sum(1)
                rec[f"{key}/outside"] = ~rkept.gather(1, tok[:, None]).squeeze(1)
        del zt
    return {k: x.cpu() for k, x in rec.items()}


def build_copies(W, mu64, t):
    return a3_lib.build(W, mu64, t)


def run_bf16_chunk(W, Wb, h, rows, rules):
    """flip_addendum_bf16.json: the BF16 product (the stock head, as exp01's C1 computes it) against the
    FP32 reference. W is Wb.float(); h holds BF16 values in FP32."""
    V = W.shape[0]
    z_ref = h @ W.T
    zb = (h.bfloat16() @ Wb.T).float()
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64,
                        device=W.device)
    g = cc.gumbel(keys, V, W.device)
    top2 = zb.topk(2, dim=1).values
    rec = {"bf16_prod/argmax": zb.argmax(1), "bf16_prod/tie": top2[:, 0] == top2[:, 1],
           "bf16_prod/maxerr": (z_ref - zb).abs().amax(1)}
    for rn, cfg in rules.items():
        rtok, rscore, rkept = cc.decide(z_ref, cfg, g), score(z_ref, cfg, g), kept(z_ref, cfg)
        tok = cc.decide(zb, cfg, g)
        key = f"bf16_prod/{rn}"
        rec[f"ref/{rn}"] = rtok
        rec[f"{key}/flip"] = tok != rtok
        if rkept is None:
            rec[f"{key}/gap"] = (rscore.gather(1, rtok[:, None]) - rscore.gather(1, tok[:, None])).squeeze(1)
        else:
            rec[f"{key}/setdiff"] = (kept(zb, cfg) ^ rkept).sum(1)
            rec[f"{key}/outside"] = ~rkept.gather(1, tok[:, None]).squeeze(1)
    return {k: x.cpu() for k, x in rec.items()}
