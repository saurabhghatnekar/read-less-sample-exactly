"""Ablation A3: mean-row shift before quantizing the head copies (Kadav et al., arXiv:2609.31291).

The certificate's reference stays the original head W. Low-bit copies are built from
W_t = W - t * 1 mu^T, and their logits are mapped back with the exact rank-one term t * (h . mu)
before any decision rule (and before Gemma 4's soft-cap). See a3_decision_rule.json.
"""
from __future__ import annotations

import torch

import cert_core as cc

U32 = cc.U32
GRID = (0.0, 1.0, 0.5, -0.5, -1.0, 1.5, 2.0, 3.0, 4.0)


def mean_row(W):
    return W.double().mean(0)


def shifted(W, mu64, t):
    if t == 0:
        return W
    return (W.double() - t * mu64).float()


def build(W, mu64, t):
    """Low-bit copies of W_t with the exp01 quantizers. t = 0 is the exp01 path unchanged."""
    Wt = shifted(W, mu64, t)
    vs = {"int8_row": cc.make_int8_row(Wt), "int4_g128": cc.make_int4_g(Wt), "int4d_g128": cc.make_int4_dither(Wt)}
    wt_norm = Wt.norm(dim=1)
    for v in vs.values():
        cc.prepare(v)
        v.t, v.wt_norm = t, wt_norm
    del Wt
    return vs


def intervals(v, h, w_norm, mu64, add_back=True):
    """Approximate logits in the ORIGINAL frame, their bounds (dict) and the FP32 slack.
    add_back=False is the planted fault for control C6_shift."""
    zt = cc.approx_logits(v, h)
    b, slack = cc.bounds_for(v, h, w_norm)
    if v.t != 0:
        c = v.t * (h.double() @ mu64)                       # [B], exact up to FP64 roundoff
        c32 = c.float()[:, None]
        if add_back:
            zt = zt + c32
        extra = 1.01 * U32 * h.norm(dim=1, keepdim=True) * v.wt_norm[None, :] \
            + 2 * U32 * (zt.abs() + c32.abs())
        b = {k: x + extra for k, x in b.items()}
        slack = slack + extra
    return zt, b, slack


def run_chunk(W, mu64, h, rows, rules, grid=GRID, int8_row_bytes=None, controls=False):
    """Certify one chunk of states at every t in grid. Returns (per-state arrays, stats)."""
    V, d = W.shape
    device = W.device
    int8_row_bytes = int8_row_bytes or d + 4
    w_norm = W.norm(dim=1)
    Bc = h.shape[0]
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64, device=device)
    g = cc.gumbel(keys, V, device)
    ref_toks = {rn: cc.decide(z_ref, rcfg, g) for rn, rcfg in rules.items()}
    rec, stats = {}, dict(c2_viol={}, c3_mismatch={}, dither_se={}, dither_n={}, c6_shift_states=None)
    for t in grid:
        vs = build(W, mu64, t)
        tag = f"t{t:g}"
        z8, b8all, _ = intervals(vs["int8_row"], h, w_norm, mu64)
        b8 = torch.minimum(b8all["wc"], b8all["res"])
        for vn, v in vs.items():
            zt, bounds, slack = intervals(v, h, w_norm, mu64)
            err = (z_ref - zt).abs()
            rec[f"{tag}/{vn}/maxerr"] = err.amax(1)
            bounds["oracle"] = err + slack
            for bn, b in bounds.items():
                rec[f"{tag}/{vn}/{bn}/meanbound"] = b.mean(1)
                rec[f"{tag}/{vn}/{bn}/maxbound"] = b.amax(1)
                if bn in ("wc", "res"):
                    k = f"{tag}/{vn}/{bn}"
                    stats["c2_viol"][k] = stats["c2_viol"].get(k, 0) + int((err > b).sum())
            if vn == "int4d_g128":
                hg = v.map_h(h).view(Bc, d // v.G, v.G)
                var = (hg.norm(dim=-1) ** 2) @ (v.scale ** 2).T / 12
                stats["dither_se"][tag] = float(((z_ref - zt) ** 2 / var).sum())
                stats["dither_n"][tag] = var.numel()
            plan = {"int8_row": [("min", None)], "int4_g128": [("res", (z8, b8))],
                    "int4d_g128": [("hoeff", None), ("hoeff", (z8, b8))]}[vn]
            for bn, st in plan:
                b = b8 if bn == "min" else bounds[bn]
                meta_pg = 0.0 if bn == "hoeff" else 2.0
                for rn, rcfg in rules.items():
                    S, R, tok = cc.certify(z_ref, zt, b, rcfg, g, st)
                    key = f"{tag}/{vn}/{bn}/{'casc' if st else 'single'}/{rn}"
                    Sc = None if S is None else S.sum(1).float()
                    Rc = R.sum(1).float()
                    rec[f"{key}/S"] = Sc if Sc is not None else torch.zeros_like(Rc)
                    rec[f"{key}/R"] = Rc
                    rec[f"{key}/bytes"] = cc.bytes_fraction(V, d, v.bytes_per_weight, meta_pg, d // v.G, Sc, Rc,
                                                           int8_row_bytes)
                    stats["c3_mismatch"][key] = stats["c3_mismatch"].get(key, 0) + int((tok != ref_toks[rn]).sum())
        if controls and t == 1.0:
            # C6_shift: drop the add-back. Detection is certain where |t h.mu| > 2 x the largest bound,
            # so every such state must show a violation; the share caught overall is reported.
            zt_bad, bb, _ = intervals(vs["int4_g128"], h, w_norm, mu64, add_back=False)
            caught = ((z_ref - zt_bad).abs() > bb["res"]).any(1)
            elig = (t * (h.double() @ mu64)).abs().float() > 2 * bb["res"].amax(1)
            stats["c6_shift_states"] = int(caught.sum())
            stats["c6_eligible"] = int(elig.sum())
            stats["c6_eligible_missed"] = int((elig & ~caught).sum())
        del vs, z8, b8, b8all
    top2 = cc.softcap(z_ref, next(iter(rules.values()))["cap"]).topk(2, dim=1).values
    rec["margin"] = top2[:, 0] - top2[:, 1]
    rec["mu_dot_h"] = (h.double() @ mu64).float()
    return {k: x.float().cpu() for k, x in rec.items()}, stats
