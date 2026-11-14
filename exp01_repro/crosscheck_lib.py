"""Cross-check our Exp01 numbers against Vizuara's cert_core.py on the same decode states.

Two 2x2 swaps, everything else held fixed (same states, same reference, same Gumbel noise):

  A. Dithered INT4 + INT8 cascade (top-50 and Gumbel T=1):
       intervals {ours, theirs}  x  read plan {ours (iterated), theirs (one-shot)}
     "intervals" = quantised head + bound + FP32 slack, for both the INT4 pass and the INT8 stage.
  B. NVFP4 with the residual bound, single stage (Gumbel T=1):
       quantiser {ours, theirs}  x  bound+plan {ours, theirs}

All work happens on soft-capped intervals (the cap is monotone, so capping the ends is equivalent).
`vz` is Vizuara's cert_core module, `cc` ours. Neither is modified.
"""
import math

import torch


def _common_bytes(vz, head, bound, d, V, R, S=None):
    return vz.total_bytes(head, bound, d, V, R.float(), None if S is None else S.float())


def _our_plan(cc, L4, U4, L8, U8, refc, cfg, g):
    """Our certify(), run on already-capped intervals: iterate INT8 until stable, then BF16 until stable."""
    S = None
    if L8 is not None:
        L4, U4, S = cc.refine_until_stable(L4, U4, cfg, g, L8, U8)
    L, U, R = cc.refine_until_stable(L4, U4, cfg, g, refc, refc)
    if cfg["rule"] in ("greedy", "gumbel"):
        tok = cc.decide(torch.where(R, refc, torch.full_like(refc, -math.inf)), cfg, g)
    else:
        tok = cc._decide_with_intervals(refc, R, L, U, cfg, g)
    return S, R, tok


def _their_plan(vz, L4, U4, L8, U8, refc, rule, g):
    """Their pre-registered plan: one rule pass on the 4-bit intervals, intersect with INT8 on S, one more pass."""
    S, aux = vz.rule_set(L4, U4, rule, g)
    if L8 is None:
        return None, S, vz.cert_token(refc, S, aux, rule, g)
    L2, U2 = vz.refine(L4, U4, S, L8, U8)
    R, aux = vz.rule_set(L2, U2, rule, g)
    return S, R, vz.cert_token(refc, R, aux, rule, g)


class Heads:
    """Both codebases' heads for one W, built once per job."""

    def __init__(self, cc, vz, W, vz_seed=20261105):
        V, d = W.shape
        self.V, self.d = V, d
        self.w_norm = W.norm(dim=1)
        # ours
        self.o_d = cc.prepare(cc.make_int4_dither(W))
        self.o_8 = cc.prepare(cc.make_int8_row(W))
        self.o_n = cc.prepare(cc.make_nvfp4(W))
        # theirs (same seeds as their exp01_modal.build_heads)
        dev = W.device
        self.t_d = vz.int_head(W, vz.A_INT4_DITHER, vz.Q_INT4, vz.GROUP, vz.dither_pattern(d, vz_seed + 2, dev))
        self.t_8 = vz.int_head(W, vz.A_INT8, vz.Q_INT8, d)
        self.t_n = vz.nvfp4_head(W)
        # quantiser swap: our NVFP4 values with their bound statistics, and theirs with ours
        G = vz.GROUP
        e, r = vz._meta(W.view(V, d // G, G), self.o_n.Wq.view(V, d // G, G))
        self.x_n_ours_in_vz = dict(W=self.o_n.Wq, s=None, e=e, r=r, group=G)
        emax, enorm = cc._group_stats(self.t_n["W"] - W, 128)
        self.x_n_theirs_in_cc = cc.prepare(cc.Variant("nvfp4_vz", self.t_n["W"], 128, emax, enorm, None, 0.5 + 1.0 / 16))


def chunk_stats(cc, vz, hd: Heads, W, h, keys, cap, do_A=True, do_B=True):
    """h: [B, d] float32 states (T=1 mode). Returns dict name -> 1-D tensor over states."""
    V, d = hd.V, hd.d
    z_ref = h @ W.T
    refc = cc.softcap(z_ref, cap)
    g = cc.gumbel(keys, V, h.device)
    phi = lambda x: cc.softcap(x, cap)
    out = {}
    rules = {
        "top50": (dict(rule="topk", k=50, T=1.0, cap=None), vz.RULE["topk50_t1.0"]),
        "gumbel_1.0": (dict(rule="gumbel", T=1.0, cap=None), vz.RULE["gumbel_t1.0"]),
    }
    ref_tok = {rn: cc.decide(z_ref, dict(c, cap=cap), g) for rn, (c, _) in rules.items()}
    for rn in rules:  # the two reference definitions must agree
        out[f"refagree/{rn}"] = (vz.ref_token(refc, rules[rn][1], g) == ref_tok[rn]).float()
    H1, H2 = vz.group_norms(h, vz.GROUP)
    tau = vz.fp_slack(h, hd.w_norm)

    if do_A:
        # ours: hoeff bound (includes slack), INT8 stage with min(wc, res)
        zt = cc.approx_logits(hd.o_d, h)
        b = cc.bounds_for(hd.o_d, h, hd.w_norm)[0]["hoeff"]
        z8 = cc.approx_logits(hd.o_8, h)
        b8a = cc.bounds_for(hd.o_8, h, hd.w_norm)[0]
        b8 = torch.minimum(b8a["wc"], b8a["res"])
        ivs = {"ours": (phi(zt - b), phi(zt + b), phi(z8 - b8), phi(z8 + b8))}
        del zt, b, z8, b8, b8a
        # theirs: dither bound + tau, INT8 resid + tau, their interval() (adds 1e-4 at each end)
        a4 = h @ hd.t_d["W"].T
        a8 = h @ hd.t_8["W"].T
        L4, U4 = vz.interval(a4, vz.bound_dither(hd.t_d, H2) + tau, cap)
        L8, U8 = vz.interval(a8, vz.bound_resid(hd.t_8, vz.group_norms(h, d)[1]) + tau, cap)
        ivs["theirs"] = (L4, U4, L8, U8)
        del a4, a8
        for iv, (L4, U4, L8, U8) in ivs.items():
            # width of the 4-bit interval, before capping is not recoverable here; use the capped width
            out[f"A/{iv}/w4_mean"] = (U4 - L4).mean(1)
            for rn, (ccfg, vrule) in rules.items():
                for plan in ("ours", "theirs"):
                    if plan == "ours":
                        S, R, tok = _our_plan(cc, L4, U4, L8, U8, refc, ccfg, g)
                    else:
                        S, R, tok = _their_plan(vz, L4, U4, L8, U8, refc, vrule, g)
                    k = f"A/{iv}-iv/{plan}-plan/{rn}"
                    out[f"{k}/S"] = S.sum(1).float()
                    out[f"{k}/R"] = R.sum(1).float()
                    out[f"{k}/bytes"] = _common_bytes(vz, "int4d", "dither", d, V, R.sum(1), S.sum(1))
                    out[f"{k}/mismatch"] = (tok != ref_tok[rn]).float()
        del ivs

    if do_B:
        ccfg, vrule = rules["gumbel_1.0"]
        variants = {"ours": (hd.o_n, hd.x_n_ours_in_vz), "theirs": (hd.x_n_theirs_in_cc, hd.t_n)}
        for qn, (v_cc, h_vz) in variants.items():
            for pipe in ("ours", "theirs"):
                if pipe == "ours":
                    zt = cc.approx_logits(v_cc, h)
                    b = cc.bounds_for(v_cc, h, hd.w_norm)[0]["res"]
                    L, U = phi(zt - b), phi(zt + b)
                    _, R, tok = _our_plan(cc, L, U, None, None, refc, ccfg, g)
                else:
                    a = h @ h_vz["W"].T
                    L, U = vz.interval(a, vz.bound_resid(h_vz, H2) + tau, cap)
                    _, R, tok = _their_plan(vz, L, U, None, None, refc, vrule, g)
                k = f"B/{qn}-quant/{pipe}-pipe/gumbel_1.0"
                out[f"{k}/R"] = R.sum(1).float()
                out[f"{k}/bytes"] = _common_bytes(vz, "nvfp4", "resid", d, V, R.sum(1))
                out[f"{k}/mismatch"] = (tok != ref_tok["gumbel_1.0"]).float()
                del L, U, R
        # how big the NVFP4 errors are under each quantiser (mean residual l2 per 128-group)
        out["B/ours-quant/resid_g_mean"] = torch.full((h.shape[0],), hd.o_n.enorm.mean().item(), device=h.device)
        out["B/theirs-quant/resid_g_mean"] = torch.full((h.shape[0],), hd.t_n["r"].mean().item(), device=h.device)
    return out
