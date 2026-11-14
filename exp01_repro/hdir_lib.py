"""K2: cancel, exactly, the quantization error against the part of the state that lies along a few fixed
directions, and bound only the remainder (hdir_rule.json).

  h . e_v = alpha . (P^T e_v) + (h - P alpha) . e_v,   alpha = P^T h
The first term is K stored numbers per row times K numbers per state. The second is bounded by the B1
sub-Gaussian bound with h replaced by h - P alpha.
"""
from __future__ import annotations

import torch

import a3_lib
import cert_core as cc
import subg_lib

U32 = cc.U32
KSETS = (("mu", None), ("pca1", 1), ("pca4", 4), ("pca16", 16), ("pca64", 64))


def fit_directions(mu64, Hcal, kmax=64):
    """mu64 [d] float64; Hcal [n, d] calibration states. Returns (P_all [d, 1 + kmax] float64, info)."""
    Hd = Hcal.double()
    C = Hd.T @ Hd
    evals, evecs = torch.linalg.eigh(C)
    kmax = min(kmax, C.shape[0])
    top = evecs[:, -kmax:].flip(1)                           # largest first
    ev = evals.flip(0)
    mu_hat = mu64 / mu64.norm()
    info = dict(n_cal=int(Hd.shape[0]), share=[float(ev[:k].sum() / ev.sum()) for k in (1, 4, 16, 64) if k <= kmax],
                cos_mu_pc1=float(mu_hat @ top[:, 0]), mu_norm=float(mu64.norm()))
    return torch.cat([mu_hat[:, None], top], 1), info


def split(Pall):
    """Named direction sets from fit_directions' matrix; pca sets are nested."""
    out = {"mu": Pall[:, :1]}
    for name, k in KSETS[1:]:
        if 1 + k <= Pall.shape[1]:
            out[name] = Pall[:, 1:1 + k]
    return out


def err_proj(Wq, Wt, Pall, rows=4096):
    """r = (Wq - Wt) P in float64, stored as float32 [V, K]."""
    out = []
    for i in range(0, Wq.shape[0], rows):
        out.append(((Wq[i:i + rows].double() - Wt[i:i + rows].double()) @ Pall).float())
    return torch.cat(out)


def corrected(v, P, r, h, zt, slack):
    """Corrected first-pass logits, their bound, and the share of ||h||^2 removed."""
    B, d = h.shape
    hd = h.double()
    alpha = hd @ P                                           # [B, K]
    corr = (alpha @ r.double().T).float()                    # [B, V]
    hp = hd - alpha @ P.T
    rho = 1 - (hp ** 2).sum(1) / (hd ** 2).sum(1)
    hg = hp.float().view(B, d // v.G, v.G).norm(dim=-1)
    b = subg_lib.subg_c() * (1 + 1e-6) * torch.sqrt((hg ** 2) @ (v.scale ** 2).T) + slack \
        + 3 * U32 * (alpha.abs().float() @ r.abs().T) + 2 * U32 * corr.abs()
    return zt - corr, b, rho.float()


def run_chunk(W, mu64, w_norm, vs, P, R, h, rows, rules):
    """vs: a3_lib.build(W, mu64, t). P, R: {name: directions [d, K] float64}, {name: r [V, K] float32}."""
    V, d = W.shape
    v = vs["int4d_g128"]
    tag = f"t{v.t:g}"
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64,
                        device=W.device)
    g = cc.gumbel(keys, V, W.device)
    ref = {rn: cc.decide(z_ref, cfg, g) for rn, cfg in rules.items()}
    z8, b8all, _ = a3_lib.intervals(vs["int8_row"], h, w_norm, mu64)
    b8 = torch.minimum(b8all["wc"], b8all["res"])
    zt, _, slack = a3_lib.intervals(v, h, w_norm, mu64)
    ti = z_ref.argmax(1, keepdim=True)
    rec, stats = {}, dict(viol={}, viol_excess_max={}, mismatch={})
    for name in ("none",) + tuple(P):
        if name == "none":
            ztc, b, rho, K = zt, subg_lib.subg_bound(v, h, slack), torch.zeros(h.shape[0], device=h.device), 0
        else:
            ztc, b, rho = corrected(v, P[name], R[name], h, zt, slack)
            K = P[name].shape[1]
        key = f"{tag}/{name}"
        err = (z_ref - ztc).abs()
        over = err - b
        stats["viol"][key] = int((over > 0).sum())
        stats["viol_excess_max"][key] = float(over.max())
        rec[f"{key}/rho"] = rho
        rec[f"{key}/b_med"] = b.median(1).values
        rec[f"{key}/b_top"] = b.gather(1, ti).squeeze(1)
        rec[f"{key}/maxerr"] = err.amax(1)
        for rn, cfg in rules.items():
            S, Rr, tok = cc.certify(z_ref, ztc, b, cfg, g, (z8, b8))
            Sc, Rc = S.sum(1).float(), Rr.sum(1).float()
            rec[f"{key}/{rn}/S"] = Sc
            rec[f"{key}/{rn}/R"] = Rc
            rec[f"{key}/{rn}/bytes"] = cc.bytes_fraction(V, d, v.bytes_per_weight, 0.0, d // v.G, Sc, Rc, d + 4) \
                + 2.0 * K / d
            stats["mismatch"][f"{key}/{rn}"] = int((tok != ref[rn]).sum())
        del ztc, b, err, over
    return {k: x.float().cpu() for k, x in rec.items()}, stats


# ---- hdir_addendum_robust.json: all five rules, and directions that depend on the state's language group

def run_chunk_sets(W, mu64, w_norm, vs, sets, h, rows, rules):
    """sets: {name: (P [d, K], r [V, K])} for one set of directions, or {name: {group: (P, r)}} where each
    state is corrected with its own group's entry (leave-one-language-out). 'none' is always run first."""
    V, d = W.shape
    v = vs["int4d_g128"]
    tag = f"t{v.t:g}"
    Bn = h.shape[0]
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64,
                        device=W.device)
    g = cc.gumbel(keys, V, W.device)
    ref = {rn: cc.decide(z_ref, cfg, g) for rn, cfg in rules.items()}
    z8, b8all, _ = a3_lib.intervals(vs["int8_row"], h, w_norm, mu64)
    b8 = torch.minimum(b8all["wc"], b8all["res"])
    zt, _, slack = a3_lib.intervals(v, h, w_norm, mu64)
    groups = [r["group"] for r in rows]
    rec, stats = {}, dict(viol={}, viol_excess_max={}, mismatch={})
    for name in ("none",) + tuple(sets):
        if name == "none":
            ztc, b, rho, K = zt, subg_lib.subg_bound(v, h, slack), torch.zeros(Bn, device=h.device), 0
        elif isinstance(sets[name], dict):
            ztc, b = torch.empty_like(zt), torch.empty_like(zt)
            rho = torch.empty(Bn, device=h.device)
            for gname in sorted(set(groups)):
                idx = torch.tensor([i for i, x in enumerate(groups) if x == gname], device=h.device)
                Pg, rg = sets[name][gname]
                ztc[idx], b[idx], rho[idx] = corrected(v, Pg, rg, h[idx], zt[idx], slack[idx])
                K = Pg.shape[1]
        else:
            Pn, rn_ = sets[name]
            ztc, b, rho = corrected(v, Pn, rn_, h, zt, slack)
            K = Pn.shape[1]
        key = f"{tag}/{name}"
        over = (z_ref - ztc).abs() - b
        stats["viol"][key] = int((over > 0).sum())
        stats["viol_excess_max"][key] = float(over.max())
        rec[f"{key}/rho"] = rho
        rec[f"{key}/b_med"] = b.median(1).values
        for rn, cfg in rules.items():
            S, Rr, tok = cc.certify(z_ref, ztc, b, cfg, g, (z8, b8))
            Sc, Rc = S.sum(1).float(), Rr.sum(1).float()
            rec[f"{key}/{rn}/S"] = Sc
            rec[f"{key}/{rn}/R"] = Rc
            rec[f"{key}/{rn}/bytes"] = cc.bytes_fraction(V, d, v.bytes_per_weight, 0.0, d // v.G, Sc, Rc, d + 4) \
                + 2.0 * K / d
            stats["mismatch"][f"{key}/{rn}"] = int((tok != ref[rn]).sum())
        del ztc, b, over
    return {k: x.float().cpu() for k, x in rec.items()}, stats
