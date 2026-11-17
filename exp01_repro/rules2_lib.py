"""T1: a leaner top-k rule, top-p (nucleus) sampling, and a certificate with any number of middle stages.

Rules are registered in cert_core.RULES so that cert_core.refine_until_stable can drive them:
  topk_lean  top-k with fewer reads: only the race candidates, and, if a candidate's membership is uncertain,
             the rows whose interval overlaps the candidates' intervals
  topp       nucleus sampling: a token is a member if the probability mass of the tokens strictly above it is
             below p; the winner is the member with the best Gumbel score

Soundness of the stable state (the rule needs no row that the last, exact stage has not read):
  topk_lean  every race candidate is exact. If one had uncertain membership, every row whose interval overlaps
             the hull of the candidates' values would be exact too, so the count of rows above it would be exact
             and its membership certain. So no candidate is uncertain, and the winner is the best certain member.
  topp       every race candidate is exact. If one had uncertain membership, the rule would still need the rows
             that carry at least half of the remaining mass uncertainty, and those are unread rows. So no
             candidate is uncertain unless every row has been read.
"""
from __future__ import annotations

import math

import torch

import cert_core as cc

MARGIN = 1e-12


def need_topk_lean(L, U, cfg, g):
    cL, cU = cc.softcap(L, cfg["cap"]), cc.softcap(U, cfg["cap"])
    cin, cout = cc.topk_membership(cL, cU, cfg["k"])
    T = cfg["T"]
    sL, sU = cL / T + g, cU / T + g
    neg = torch.finfo(sL.dtype).min
    m = torch.where(cin, sL, torch.full_like(sL, neg)).amax(1, keepdim=True)
    race = (~cout) & (sU >= m)
    cand = race & ~cin
    lo = torch.where(cand, cL, torch.full_like(cL, float("inf"))).amin(1, keepdim=True)
    hi = torch.where(cand, cU, torch.full_like(cU, -float("inf"))).amax(1, keepdim=True)
    hull = (cU >= lo) & (cL <= hi)
    return race | (hull & cand.any(1, keepdim=True))


def _pad_gather(cs, idx):
    """cs: cumulative sums [B, V]; idx in [0, V]: the sum of the first idx entries."""
    z = torch.zeros(cs.shape[0], 1, dtype=cs.dtype, device=cs.device)
    return torch.cat([z, cs], 1).gather(1, idx)


def topp_membership(aL, aU, p):
    """aL, aU: float64 intervals on a = softcap(z) / T. Returns (certainly in, certainly out, wL, wU).
    Mass above token i is A / (A + R): A the mass of tokens strictly above it, R the rest (it included).
    upper = A_up / (A_up + R_low) and lower = A_low / (A_low + R_up) bracket it."""
    amax = aU.amax(1, keepdim=True)
    wL, wU = torch.exp(aL - amax), torch.exp(aU - amax)
    open_ = (aU > aL).to(wL.dtype)
    Us, iu = aU.sort(1)
    cs_wU_byU = wU.gather(1, iu).cumsum(1)
    cs_wL_byU = wL.gather(1, iu).cumsum(1)
    n_u_le = torch.searchsorted(Us, aL.contiguous(), right=True)          # rows j with aU_j <= aL_i
    A_up = cs_wU_byU[:, -1:] - _pad_gather(cs_wU_byU, n_u_le) - wU * open_
    R_low = _pad_gather(cs_wL_byU, n_u_le) + wL * open_
    Ls, il = aL.sort(1)
    cs_wL_byL = wL.gather(1, il).cumsum(1)
    cs_wU_byL = wU.gather(1, il).cumsum(1)
    n_l_le = torch.searchsorted(Ls, aU.contiguous(), right=True)          # rows j with aL_j <= aU_i (i included)
    A_low = cs_wL_byL[:, -1:] - _pad_gather(cs_wL_byL, n_l_le)
    R_up = _pad_gather(cs_wU_byL, n_l_le)
    A_up, A_low = A_up.clamp_min(0), A_low.clamp_min(0)
    upper = A_up / (A_up + R_low)
    lower = A_low / (A_low + R_up)
    return upper * (1 + MARGIN) + MARGIN < p, lower * (1 - MARGIN) - MARGIN >= p, wL, wU


def need_topp(L, U, cfg, g):
    cL, cU = cc.softcap(L, cfg["cap"]), cc.softcap(U, cfg["cap"])
    T = cfg["T"]
    cin, cout, wL, wU = topp_membership(cL.double() / T, cU.double() / T, cfg["p"])
    sL, sU = cL / T + g, cU / T + g
    neg = torch.finfo(sL.dtype).min
    m = torch.where(cin, sL, torch.full_like(sL, neg)).amax(1, keepdim=True)
    race = (~cout) & (sU >= m)
    cand = race & ~cin
    width = wU - wL
    heavy = (width > 0) & (width >= 0.5 * width.mean(1, keepdim=True))
    return race | (heavy & cand.any(1, keepdim=True))


cc.RULES["topk_lean"] = need_topk_lean
cc.RULES["topp"] = need_topp


def decide(z, cfg, g):
    """Exact decision on full reference logits, for every rule here."""
    r = cfg["rule"]
    if r == "topk_lean":
        return cc.decide(z, dict(cfg, rule="topk"), g)
    if r != "topp":
        return cc.decide(z, cfg, g)
    c = cc.softcap(z, cfg["cap"])
    T = cfg["T"]
    a = c.double() / T
    w = torch.exp(a - a.amax(1, keepdim=True))
    As, ia = a.sort(1)
    cs = w.gather(1, ia).cumsum(1)
    n_le = torch.searchsorted(As, a.contiguous(), right=True)
    above = cs[:, -1:] - _pad_gather(cs, n_le)                               # mass strictly above each token
    member = above / cs[:, -1:] < cfg["p"]
    s = c / T + g
    return torch.where(member, s, torch.full_like(s, -float("inf"))).argmax(1)


def _final(z_ref, R, L, U, cfg, g):
    r = cfg["rule"]
    if r in ("greedy", "gumbel"):
        return cc.decide(torch.where(R, z_ref, torch.full_like(z_ref, -float("inf"))), cfg, g)
    if r in ("topk", "minp"):
        return cc._decide_with_intervals(z_ref, R, L, U, cfg, g)
    if r == "topk_lean":
        return cc._decide_with_intervals(z_ref, R, L, U, dict(cfg, rule="topk"), g)
    cL, cU = cc.softcap(L, cfg["cap"]), cc.softcap(U, cfg["cap"])
    T = cfg["T"]
    cin, _, _, _ = topp_membership(cL.double() / T, cU.double() / T, cfg["p"])
    s = cc.softcap(z_ref, cfg["cap"]) / T + g
    return torch.where(R & cin, s, torch.full_like(s, -float("inf"))).argmax(1)


def certify(z_ref, zt, b, cfg, g, stages):
    """stages: list of (z, b) intervals for the middle stages, in order. Returns (list of read masks, R, token)."""
    L, U = zt - b, zt + b
    reads = []
    for zs, bs in stages:
        L, U, S = cc.refine_until_stable(L, U, cfg, g, zs - bs, zs + bs)
        reads.append(S)
    L, U, R = cc.refine_until_stable(L, U, cfg, g, z_ref, z_ref)
    return reads, R, _final(z_ref, R, L, U, cfg, g)
