"""C6 planted faults: each check breaks one thing on purpose and the harness must notice.

  a  2% of the residual bound must show violations on every head
  b  a head with shifted rows must show violations of the worst-case and residual bounds
  c  a recompute set with the winner removed must give a wrong token on every state
  d  an exact head with a zero bound must leave exactly one row on at least 95% of states
  e  an empty head (all zeros) must leave at least 99% of rows and show no violations
  f  FP32 against FP64 differences (reference and every head) must sit inside the slack

`cc` is cert_core. All counters are plain Python numbers so chunks can be summed.
"""
import torch


def _masked_token(cc, z_ref, keep, cfg, g):
    """Best-scoring row among `keep`, scored as the rule scores members."""
    s = cc.softcap(z_ref, cfg["cap"])
    if cfg["rule"] != "greedy":
        s = s / cfg["T"] + g
    tok = torch.where(keep, s, torch.full_like(s, -float("inf"))).argmax(1)
    return torch.where(keep.any(1), tok, torch.full_like(tok, -1))   # an empty set gives no token, not token 0


def planted_chunk(cc, variants, W, w_norm, h, keys, rules, n_fp64=0):
    """variants: name -> prepared Variant (must hold int8_row and int4d_g128). rules: name -> cfg with cap."""
    V, d = W.shape
    B = h.shape[0]
    z_ref = h @ W.T
    g = cc.gumbel(keys, V, h.device)
    ref = {rn: cc.decide(z_ref, cfg, g) for rn, cfg in rules.items()}
    perm = torch.roll(torch.arange(V, device=h.device), 1)
    out = {"n_states": B, "n_rows": B * V}

    if n_fp64:
        hs = h[:n_fp64]
        ref_gap = (z_ref[:n_fp64].double() - hs.double() @ W.double().T).abs()
        out["f/n_states"] = hs.shape[0]

    for vn, v in variants.items():
        zt = cc.approx_logits(v, h)
        bounds, slack = cc.bounds_for(v, h, w_norm)
        err = (z_ref - zt).abs()
        out[f"a/{vn}"] = (err > 0.02 * (bounds["res"] - slack) + slack).sum().item()
        err_sh = (z_ref - zt[:, perm]).abs()
        for bn in ("wc", "res"):
            out[f"b/{vn}/{bn}"] = (err_sh > bounds[bn]).sum().item()
        if n_fp64:
            hm = hs.double() if v.rot is None else hs.double() @ v.rot.double()
            gap = ref_gap + (zt[:n_fp64].double() - hm @ v.Wq.double().T).abs()
            out[f"f/{vn}/viol"] = (gap > slack[:n_fp64]).sum().item()
            out[f"f/{vn}/max_ratio"] = (gap / slack[:n_fp64]).max().item()
        del zt, bounds, err, err_sh

    # c: dithered INT4 + INT8, winner taken out of R
    vd, v8 = variants["int4d_g128"], variants["int8_row"]
    zt = cc.approx_logits(vd, h)
    b = cc.bounds_for(vd, h, w_norm)[0]["hoeff"]
    z8 = cc.approx_logits(v8, h)
    b8a = cc.bounds_for(v8, h, w_norm)[0]
    b8 = torch.minimum(b8a["wc"], b8a["res"])
    rows = torch.arange(B, device=h.device)
    for rn, cfg in rules.items():
        _, R, tok = cc.certify(z_ref, zt, b, cfg, g, (z8, b8))
        out[f"c/{rn}/clean_mismatch"] = (tok != ref[rn]).sum().item()
        R2 = R.clone()
        R2[rows, ref[rn]] = False
        out[f"c/{rn}/detected"] = (_masked_token(cc, z_ref, R2, cfg, g) != ref[rn]).sum().item()
    del zt, b, z8, b8, b8a

    # d: exact head, zero bound
    zero = torch.zeros_like(z_ref)
    for rn, cfg in rules.items():
        _, R, tok = cc.certify(z_ref, z_ref, zero, cfg, g)
        out[f"d/{rn}/one_row"] = (R.sum(1) == 1).sum().item()
        out[f"d/{rn}/mismatch"] = (tok != ref[rn]).sum().item()

    # e: empty head. The error is W itself, so the bound statistics come from -W.
    emax, enorm = cc._group_stats(-W, 128)
    ve = cc.Variant("empty", None, 128, emax, enorm, None, 0.0)
    ve.wq_norm = torch.zeros(V, device=h.device)
    eb = cc.bounds_for(ve, h, w_norm)[0]
    for bn in ("wc", "res"):
        out[f"e/viol/{bn}"] = (z_ref.abs() > eb[bn]).sum().item()
    for rn, cfg in rules.items():
        _, R, tok = cc.certify(z_ref, zero, eb["res"], cfg, g)
        out[f"e/{rn}/rows_left"] = R.sum().item()
        out[f"e/{rn}/mismatch"] = (tok != ref[rn]).sum().item()
    return out


def merge(total, part):
    for k, v in part.items():
        total[k] = max(total.get(k, 0.0), v) if k.endswith("max_ratio") else total.get(k, 0) + v
    return total


def verdict(tot, rule_names, variant_names):
    """Return (checks: name -> bool, passed: bool) for one mode's summed counters."""
    n, nr = tot["n_states"], tot["n_rows"]
    c = {
        "a_2pct_resid_violates": all(tot[f"a/{v}"] > 0 for v in variant_names),
        "b_shifted_rows_violate": all(tot[f"b/{v}/{b}"] > 0 for v in variant_names for b in ("wc", "res")),
        "c_winner_removed_detected": all(tot[f"c/{r}/detected"] == n and tot[f"c/{r}/clean_mismatch"] == 0
                                         for r in rule_names),
        "d_exact_head_one_row": all(tot[f"d/{r}/one_row"] >= 0.95 * n and tot[f"d/{r}/mismatch"] == 0
                                    for r in rule_names),
        "e_empty_head": all(tot[f"e/viol/{b}"] == 0 for b in ("wc", "res"))
                        and all(tot[f"e/{r}/rows_left"] >= 0.99 * nr and tot[f"e/{r}/mismatch"] == 0
                                for r in rule_names),
        "f_fp64_inside_slack": all(tot[f"f/{v}/viol"] == 0 for v in variant_names),
    }
    return c, all(c.values())
