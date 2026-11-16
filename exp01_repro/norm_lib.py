"""N1: where the dominant state direction appears (norm_rule.json). Torch only, so it runs in both model images.

find_sites     the decoder blocks, the final norm and the head of a Hugging Face causal LM
greedy_capture a minimal greedy decode that keeps the last-position state at every site
site_stats     the measures of the rule for one site
analyse        every site, the norm, the 128-coordinate groups and the verdict
"""
import torch

LAYER_NAMES = ("layers", "h", "blocks", "block")
NORM_NAMES = ("norm", "ln_f", "final_layernorm", "final_layer_norm", "final_norm", "norm_f")
G = 128


def find_sites(m):
    """(name of the block list, blocks, name of the final norm, norm, head). The shallowest match wins; the run's
    control (norm output == head input on every step) is what confirms the choice."""
    head = m.get_output_embeddings()
    cands = [(n, x) for n, x in m.named_modules()
             if isinstance(x, torch.nn.ModuleList) and n.split(".")[-1] in LAYER_NAMES and len(x) >= 2]
    lname, layers = min(cands, key=lambda c: c[0].count("."))
    norms = [(n, x) for n, x in m.named_modules()
             if n and not n.startswith(lname + ".") and n.split(".")[-1] in NORM_NAMES]
    nname, norm = min(norms, key=lambda c: c[0].count("."))
    return lname, list(layers), nname, norm, head


def greedy_capture(m, encode, prompts, B, ntok, eos):
    """Greedy decode with the cache. Returns (sites, rows, control).
    sites: dict name -> [n, d] tensor on the CPU in the model's dtype; names block_1..block_L, pre_norm, post_norm.
    rows: one dict per state (pid, group, step, tok). encode(batch) -> (ids, mask) on the model's device."""
    lname, layers, nname, norm, head = find_sites(m)
    cur = {}

    def last(x):
        return (x[0] if isinstance(x, (tuple, list)) else x)[:, -1, :].detach()

    hooks = [blk.register_forward_hook(lambda mod, a, out, i=i: cur.__setitem__(f"block_{i + 1}", last(out)))
             for i, blk in enumerate(layers)]
    hooks.append(norm.register_forward_pre_hook(lambda mod, a: cur.__setitem__("pre_norm", last(a[0]))))
    hooks.append(norm.register_forward_hook(lambda mod, a, out: cur.__setitem__("post_norm", last(out))))
    hooks.append(head.register_forward_pre_hook(lambda mod, a: cur.__setitem__("head_in", last(a[0]))))
    names = [f"block_{i + 1}" for i in range(len(layers))] + ["pre_norm", "post_norm"]
    keep = {k: [] for k in names}
    rows = []
    ctl = dict(steps=0, norm_out_differs_from_head_in=0, last_block_differs_from_norm_in=0,
               layers=lname, n_layers=len(layers), norm=nname, norm_class=type(norm).__name__)
    eos_t = None
    with torch.no_grad():
        for bi in range(0, len(prompts), B):
            batch = prompts[bi:bi + B]
            ids, mask = encode(batch)
            if eos_t is None:
                eos_t = torch.tensor(sorted(eos), device=ids.device)
            alive = torch.ones(len(batch), dtype=torch.bool, device=ids.device)
            pkv, inp = None, ids
            for t in range(ntok):
                cur.clear()
                out = m(input_ids=inp, attention_mask=mask, past_key_values=pkv, use_cache=True)
                tok = out.logits[:, -1, :].argmax(-1)
                ctl["steps"] += 1
                ctl["norm_out_differs_from_head_in"] += int(not torch.equal(cur["post_norm"], cur["head_in"]))
                ctl["last_block_differs_from_norm_in"] += int(not torch.equal(cur[names[len(layers) - 1]], cur["pre_norm"]))
                idx = alive.nonzero().squeeze(1)
                for k in names:
                    keep[k].append(cur[k][idx].cpu())
                tl = tok.tolist()
                rows += [dict(pid=batch[j]["id"], group=batch[j]["group"], step=t, tok=tl[j]) for j in idx.tolist()]
                alive = alive & ~torch.isin(tok, eos_t)
                if not alive.any():
                    break
                pkv, inp = out.past_key_values, tok[:, None]
                mask = torch.cat([mask, mask.new_ones(len(batch), 1)], 1)
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in keep.items()}, rows, ctl


def _unit(v):
    return v / v.norm().clamp_min(1e-300)


def site_stats(X, even, odd):
    """X: [n, d] float64. Returns (stats, p, mean direction). Fits on the even prompts."""
    Xe = X[even]
    n2 = (X ** 2).sum(1)
    energy = n2[even].mean()
    mean = Xe.mean(0)
    w, v = torch.linalg.eigh(Xe.T @ Xe)
    p = v[:, -1]
    mh = _unit(mean)
    p2 = p ** 2
    out = dict(top1=float(w[-1] / w.sum()), mean_share=float(mean @ mean / energy),
               unit_mean_share=float((Xe / n2[even].sqrt()[:, None]).mean(0).pow(2).sum()),
               along_mean=float(((Xe @ mh) ** 2).mean() / energy),
               rho_median_odd=float(((X[odd] @ p) ** 2 / n2[odd]).median()),
               cos_p_mean=float((p @ mh).abs()), norm_median=float(n2.sqrt().median()),
               p_top5_coordinates=float(p2.sort().values[-5:].sum()))
    if X.shape[1] % G == 0:
        out["p_largest_group"] = float(p2.view(-1, G).sum(1).max())
    return out, p, mh


def _norm_info(norm):
    g = getattr(norm, "weight", None)
    b = getattr(norm, "bias", None)
    out = dict(cls=type(norm).__name__, eps=float(getattr(norm, "eps", getattr(norm, "variance_epsilon", float("nan")))),
               has_gain=g is not None, has_bias=b is not None)
    if g is not None:
        gd = g.detach().double().flatten()
        out.update(gain_mean=float(gd.mean()), gain_std=float(gd.std()), gain_min=float(gd.min()), gain_max=float(gd.max()),
                   gain_negative=int((gd < 0).sum()),
                   gain_top5_share=float((gd ** 2).sort().values[-5:].sum() / (gd ** 2).sum()))
    if b is not None:
        out["bias_norm"] = float(b.detach().double().norm())
    return out


def _group_part(Y, p, S, odd, groups):
    """Share of the first-pass variance term by 128-coordinate group, per language, before and after removing p.
    S: [d / 128] mean over rows of the squared INT4 group scale, or None for raw energy."""
    Yo = Y[odd]
    res = {}
    for tag, Z in (("h", Yo), ("h_minus_p", Yo - (Yo @ p)[:, None] * p[None, :])):
        e = (Z ** 2).view(Z.shape[0], -1, G).sum(-1)
        for wname, wt in (("variance_term", S), ("energy", None)):
            if wname == "variance_term" and S is None:
                continue
            q = e * wt[None, :] if wt is not None else e
            q = q / q.sum(1, keepdim=True)
            top, arg = q.max(1)
            per = {}
            for g in sorted(set(groups)):
                mk = torch.tensor([x == g for x in groups], device=Y.device)
                a = arg[mk]
                mode = int(a.mode().values)
                per[g] = dict(states=int(mk.sum()), largest_group_share_median=float(top[mk].median()),
                              most_common_largest_group=mode, how_often=float((a == mode).float().mean()),
                              top4_groups_share_median=float(q[mk].sort(1).values[:, -4:].sum(1).median()))
            res[f"{tag}/{wname}"] = per
    return res


def verdict(gain_free_top1, post_top1):
    r = gain_free_top1 / post_top1
    return "norm_makes_it" if r <= 0.5 else "in_the_model" if r >= 0.8 else "both"


def analyse(sites, rows, norm, W=None, ref_p=None, dev="cpu"):
    """sites: dict from greedy_capture. norm: the final norm module. W: head [V, d] for the group scales (optional).
    ref_p: a saved direction to compare the post-norm p with (optional)."""
    pid = torch.tensor([r["pid"] for r in rows])
    even, odd = (pid % 2 == 0).to(dev), (pid % 2 == 1).to(dev)
    info = _norm_info(norm)
    X = sites["pre_norm"].to(dev).double()
    Y = sites["post_norm"].to(dev).double()
    d = Y.shape[1]
    g = norm.weight.detach().to(dev).double().flatten() if info["has_gain"] else torch.ones(d, dtype=torch.float64, device=dev)
    b = norm.bias.detach().to(dev).double().flatten() if info["has_bias"] else torch.zeros(d, dtype=torch.float64, device=dev)
    U = (Y - b) / g                                              # the post-norm state with gain and bias undone
    Xc = X - X.mean(1, keepdim=True) if "LayerNorm" in info["cls"] else X
    Xn = Xc / Xc.pow(2).mean(1, keepdim=True).sqrt()             # the pre-norm state over its root mean square
    cosUX = torch.nn.functional.cosine_similarity(U, Xn, dim=1)
    info["gain_source"] = "module weight"
    info["cos_with_module_weight_median"] = float(cosUX.median())
    if float(cosUX.median()) < 0.999:
        # the module does not apply its weight as y = g x^ + b (for example a 1 + weight convention): fit the
        # gain and bias it does apply, per coordinate, by least squares of y on the normalised pre-norm state
        xm, ym = Xn.mean(0), Y.mean(0)
        g = ((Xn - xm) * (Y - ym)).sum(0) / ((Xn - xm) ** 2).sum(0)
        b = ym - g * xm
        if not info["has_bias"]:
            g, b = (Xn * Y).sum(0) / (Xn ** 2).sum(0), torch.zeros_like(g)
        U = (Y - b) / g
        cosUX = torch.nn.functional.cosine_similarity(U, Xn, dim=1)
        gd = g
        info.update(gain_source="fitted per coordinate", gain_mean=float(gd.mean()), gain_std=float(gd.std()),
                    gain_min=float(gd.min()), gain_max=float(gd.max()), gain_negative=int((gd < 0).sum()),
                    gain_top5_share=float((gd ** 2).sort().values[-5:].sum() / (gd ** 2).sum()),
                    fit_residual_median=float(((Y - (g * Xn + b)).norm(dim=1) / Y.norm(dim=1)).median()))
    info.update(gain_free_rms_median=float(U.pow(2).mean(1).sqrt().median()),
                cos_gain_free_vs_normalised_pre_median=float(cosUX.median()),
                cos_gain_free_vs_normalised_pre_min=float(cosUX.min()))
    table, P, M = {}, {}, {}
    order = [k for k in sites if k.startswith("block_")] + ["pre_norm", "pre_norm_over_rms", "gain_free", "post_norm"]
    for k in order:
        Z = Xn if k == "pre_norm_over_rms" else U if k == "gain_free" else sites[k].to(dev).double()
        table[k], P[k], M[k] = site_stats(Z, even, odd)
        del Z
    prev = None
    for k in order:
        table[k]["cos_p_with_post"] = float((P[k] @ P["post_norm"]).abs())
        table[k]["cos_mean_with_post"] = float((M[k] @ M["post_norm"]).abs())
        if prev is not None:
            table[k]["cos_p_with_previous"] = float((P[k] @ P[prev]).abs())
        prev = k
    if info["has_gain"]:
        mu_abs = U[even].mean(0).abs()
        info["corr_gain_with_abs_mean_of_gain_free"] = float(torch.corrcoef(torch.stack((g.abs(), mu_abs)))[0, 1])
        info["cos_post_mean_with_gain"] = float((M["post_norm"] @ _unit(g)).abs())
    pre = table["pre_norm"]["top1"]
    blocks = [k for k in order if k.startswith("block_")]
    first = next((k for k in blocks if table[k]["top1"] >= 0.5 * pre), None)
    two_thirds = blocks[max(0, (2 * len(blocks) + 2) // 3 - 1)]
    out = dict(n_states=len(rows), n_even=int(even.sum()), n_odd=int(odd.sum()), prompts=len(set(pid.tolist())), d=d,
               norm=info, sites=table,
               verdict=dict(gain_free_top1=table["gain_free"]["top1"], post_top1=table["post_norm"]["top1"],
                            ratio=table["gain_free"]["top1"] / table["post_norm"]["top1"],
                            verdict=verdict(table["gain_free"]["top1"], table["post_norm"]["top1"]),
                            mean_share_gain_free=table["gain_free"]["mean_share"],
                            mean_share_post=table["post_norm"]["mean_share"],
                            first_block_at_half_of_pre_norm=first, block_at_two_thirds=two_thirds,
                            top1_at_two_thirds=table[two_thirds]["top1"],
                            last_block_change=(table[blocks[-1]]["top1"] - table[blocks[-2]]["top1"])))
    if ref_p is not None:
        out["cos_post_p_with_saved"] = float((P["post_norm"] @ _unit(ref_p.to(dev).double())).abs())
    if d % G == 0:
        S = None
        if W is not None:
            S = torch.zeros(d // G, dtype=torch.float64, device=dev)
            for r0 in range(0, W.shape[0], 16384):
                w = W[r0:r0 + 16384].to(dev).float()
                s = (w.view(w.shape[0], -1, G).abs().amax(-1) / 6.5).half().float().clamp_min(1e-12)
                S += (s.double() ** 2).sum(0)
            S /= W.shape[0]
        groups = [r["group"] for r, o in zip(rows, odd.tolist()) if o]
        out["groups_of_128"] = _group_part(Y, P["post_norm"], S, odd, groups)
        if S is not None:
            out["group_scale_sq_max_over_mean"] = float(S.max() / S.mean())
    return out
