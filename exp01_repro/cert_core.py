"""Exp01 independent reproduction: head variants, logit bounds, rule-specific recompute sets.

Everything here is torch and runs on CPU (self-tests) or CUDA (the real run).
TF32 must be off: the reference is the FP32-accumulated product of BF16 rows and BF16 h.

Notation: W [V, d] the BF16 head (held as float32), h [B, d] final hidden states (BF16 values as float32).
An approximate head W~ gives z~ = h W~^T and a per-row bound b with |z_ref - z~| <= b.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

U32 = 2.0 ** -24  # unit roundoff of FP32


# ----------------------------------------------------------------------------------------------
# Head variants
# ----------------------------------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    Wq: torch.Tensor                 # dequantized weights, float32 [V, d] (possibly in a rotated basis)
    G: int                           # group size the bound statistics use
    emax: torch.Tensor               # [V, d/G] max |error| per group (worst-case bound)
    enorm: torch.Tensor              # [V, d/G] l2 norm of error per group (residual-norm bound)
    scale: torch.Tensor | None       # [V, d/G] quantization step per group (dither bound), else None
    bytes_per_weight: float          # first-pass bytes per weight incl. scales (not bound metadata)
    rot: torch.Tensor | None = None  # [d, d] orthogonal R; the variant stores W R, and h is mapped to h R
    meta_bytes_per_group: dict = field(default_factory=lambda: {"wc": 2.0, "res": 2.0, "hoeff": 0.0})

    def map_h(self, h):
        return h if self.rot is None else h @ self.rot


def _group_stats(E, G):
    V, d = E.shape
    Eg = E.view(V, d // G, G)
    return Eg.abs().amax(-1), Eg.norm(dim=-1)


def make_int8_row(W):
    """Symmetric INT8, one FP16 scale per row."""
    s = (W.abs().amax(1, keepdim=True) / 127).half().float().clamp_min(1e-12)
    q = torch.clamp(torch.round(W / s), -127, 127)
    Wq = q * s
    emax, enorm = _group_stats(Wq - W, W.shape[1])
    return Variant("int8_row", Wq, W.shape[1], emax, enorm, None, 1.0 + 2.0 / W.shape[1])


def make_int4_g(W, G=128):
    """Symmetric INT4 round-to-nearest, groups of G, FP16 scales."""
    V, d = W.shape
    Wg = W.view(V, d // G, G)
    s = (Wg.abs().amax(-1, keepdim=True) / 7).half().float().clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / s), -8, 7)
    Wq = (q * s).view(V, d)
    emax, enorm = _group_stats(Wq - W, G)
    return Variant(f"int4_g{G}", Wq, G, emax, enorm, s.squeeze(-1), 0.5 + 2.0 / G)


def make_int4_dither(W, G=128, seed=1234):
    """Subtractive-dither INT4: one dither pattern u[d] ~ U(-1/2, 1/2) shared by all rows,
    scaled by each row's group step. q = round(w/s + u), w~ = (q - u) s.
    The error s*(round(x+u) - (x+u)) is uniform on [-s/2, s/2] and independent of w.
    Step s = amax/6.5 keeps |x + u| <= 7 so q fits in [-8, 7] without clipping."""
    V, d = W.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = (torch.rand(d, generator=gen, dtype=torch.float64) - 0.5).float().to(W.device)
    Wg = W.view(V, d // G, G)
    s = (Wg.abs().amax(-1, keepdim=True) / 6.5).half().float().clamp_min(1e-12)
    ug = u.view(1, d // G, G)
    x = Wg / s + ug
    q = torch.round(x)
    assert q.abs().max() <= 8, "dither clipped"
    q = torch.clamp(q, -8, 7)
    Wq = ((q - ug) * s).view(V, d)
    emax, enorm = _group_stats(Wq - W, G)
    v = Variant(f"int4d_g{G}", Wq, G, emax, enorm, s.squeeze(-1), 0.5 + 2.0 / G)
    v.dither = u
    return v


def hadamard(n, device):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(n)).to(device)


def make_int4_rot(W, G=128, seed=4321):
    """Block randomized Hadamard (block = G) with random signs, then INT4 g128 RTN."""
    V, d = W.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (d,), generator=gen) * 2 - 1).double().to(W.device)
    Hb = hadamard(G, W.device)
    R = torch.block_diag(*([Hb] * (d // G)))
    R = (signs[:, None] * R).float()       # R = D H, orthogonal
    Wr = W @ R
    v = make_int4_g(Wr, G)
    # bound stats must be in the rotated basis (errors live there); keep them
    v.name, v.rot = f"int4rot_g{G}", R
    return v


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def make_nvfp4(W):
    """NVFP4 emulation: E2M1 values in 16-element blocks, E4M3 block scales, FP32 tensor scale."""
    V, d = W.shape
    B = 16
    Wb = W.view(V, d // B, B)
    ts = (W.abs().max() / (6.0 * 448.0)).float()
    bs = (Wb.abs().amax(-1, keepdim=True) / 6.0 / ts).to(torch.float8_e4m3fn).float()
    sc = (bs * ts).clamp_min(1e-30)
    grid = E2M1.to(W.device)
    Wq = torch.empty_like(W)
    for r0 in range(0, V, 8192):  # row chunks: the grid search is 8x the chunk's size
        x = Wb[r0:r0 + 8192] / sc[r0:r0 + 8192]
        idx = (x.abs().unsqueeze(-1) - grid).abs().argmin(-1)
        Wq[r0:r0 + 8192] = (grid[idx] * torch.sign(x) * sc[r0:r0 + 8192]).view(-1, d)
    # bound statistics over 128-element groups (one FP16 per 128 weights of metadata, like INT4)
    emax, enorm = _group_stats(Wq - W, 128)
    return Variant("nvfp4", Wq, 128, emax, enorm, None, 0.5 + 1.0 / B)


# ----------------------------------------------------------------------------------------------
# Bounds
# ----------------------------------------------------------------------------------------------

def hoeffding_c(delta):
    return math.sqrt(math.log(2.0 / delta) / 2.0)


def bounds_for(v: Variant, h, w_norm, delta=1e-15):
    """Return dict bound_name -> [B, V] for states h [B, d] (already in the original basis).
    Each bound includes FP32 slack for both our approximate product and the FP32 reference."""
    hm = v.map_h(h)
    Bsz, d = hm.shape
    hg = hm.view(Bsz, d // v.G, v.G)
    h1 = hg.abs().sum(-1)                  # [B, ng]
    h2 = hg.norm(dim=-1)                   # [B, ng]
    out = {
        "wc": h1 @ v.emax.T,               # sum_g emax_g * ||h_g||_1
        "res": h2 @ v.enorm.T,             # sum_g ||e_g||_2 ||h_g||_2
    }
    if v.scale is not None and hasattr(v, "dither"):
        out["hoeff"] = hoeffding_c(delta) * torch.sqrt((h2 ** 2) @ (v.scale ** 2).T)
    gamma = d * U32 / (1 - d * U32)
    hn = h.norm(dim=1, keepdim=True)
    slack = gamma * hn * (w_norm[None, :] + v.wq_norm[None, :])
    return {k: b + slack for k, b in out.items()}, slack


def prepare(v: Variant):
    v.wq_norm = v.Wq.norm(dim=1)
    return v


def approx_logits(v: Variant, h):
    hm = v.map_h(h)
    return hm @ v.Wq.T


# ----------------------------------------------------------------------------------------------
# Decision rules. Each takes intervals [L, U] on raw logits (B x V) and returns the mask of rows
# that must be refined (read more precisely) for the decision to be determined.
# Soft-cap c(z) = cap * tanh(z / cap) is monotone, so intervals map through it.
# ----------------------------------------------------------------------------------------------

def softcap(z, cap):
    return z if cap is None else cap * torch.tanh(z / cap)


def _race_need(sL, sU, eligible):
    """Rows that can still win argmax of a score among eligible rows (scores in [sL, sU])."""
    neg = torch.finfo(sL.dtype).min
    m = torch.where(eligible, sL, torch.full_like(sL, neg)).amax(1, keepdim=True)
    return eligible & (sU >= m)


def need_greedy(L, U, cfg, g=None):
    cL, cU = softcap(L, cfg["cap"]), softcap(U, cfg["cap"])
    return _race_need(cL, cU, torch.ones_like(L, dtype=torch.bool))


def need_gumbel(L, U, cfg, g):
    T = cfg["T"]
    sL = softcap(L, cfg["cap"]) / T + g
    sU = softcap(U, cfg["cap"]) / T + g
    return _race_need(sL, sU, torch.ones_like(L, dtype=torch.bool))


def _count_greater(sorted_vals, q):
    """For each q[b, i], how many entries of row b of sorted (ascending) values are strictly > q."""
    n = sorted_vals.shape[1]
    return n - torch.searchsorted(sorted_vals, q.contiguous(), right=True)


def topk_membership(L, U, k):
    """certain_in: fewer than k other rows can exceed it. certain_out: at least k rows surely exceed it."""
    Us, _ = U.sort(1)
    Ls, _ = L.sort(1)
    n_above_possible = _count_greater(Us, L) - (U > L).long()   # other rows with U_j > L_i
    n_above_sure = _count_greater(Ls, U)                         # rows with L_j > U_i (never i itself)
    cin = n_above_possible < k
    cout = n_above_sure >= k
    return cin, cout


def need_topk(L, U, cfg, g):
    cL, cU = softcap(L, cfg["cap"]), softcap(U, cfg["cap"])
    cin, cout = topk_membership(cL, cU, cfg["k"])
    T = cfg["T"]
    sL, sU = cL / T + g, cU / T + g
    # winner must be a member: the race threshold is the best sure score among certain members
    neg = torch.finfo(sL.dtype).min
    m = torch.where(cin, sL, torch.full_like(sL, neg)).amax(1, keepdim=True)
    race = (~cout) & (sU >= m)
    unc = ~cin & ~cout
    any_unc_in_race = (race & unc).any(1, keepdim=True)
    # pre-registered plan: if any race candidate has uncertain membership, read every uncertain row,
    # plus every row whose interval overlaps the k-th value's interval (needed to pin the boundary)
    k = cfg["k"]
    kthL = cL.topk(k, dim=1).values[:, -1:]
    kthU = cU.topk(k, dim=1).values[:, -1:]
    band = (cU >= kthL) & (cL <= kthU)
    return race | ((unc | band) & any_unc_in_race)


def need_minp(L, U, cfg, g):
    cL, cU = softcap(L, cfg["cap"]), softcap(U, cfg["cap"])
    T, lp = cfg["T"], math.log(cfg["p"])
    maxL = cL.amax(1, keepdim=True)
    maxU = cU.amax(1, keepdim=True)
    cin = cL >= maxU + T * lp
    cout = cU < maxL + T * lp
    sL, sU = cL / T + g, cU / T + g
    neg = torch.finfo(sL.dtype).min
    m = torch.where(cin, sL, torch.full_like(sL, neg)).amax(1, keepdim=True)
    race = (~cout) & (sU >= m)
    unc = ~cin & ~cout
    any_unc_in_race = (race & unc).any(1, keepdim=True)
    max_cand = cU >= maxL                          # rows that could be the max (pin the threshold)
    return race | ((unc | max_cand) & any_unc_in_race)


RULES = {"greedy": need_greedy, "gumbel": need_gumbel, "topk": need_topk, "minp": need_minp}


def decide(z, cfg, g):
    """Exact decision on full reference logits z [B, V]."""
    c = softcap(z, cfg["cap"])
    r = cfg["rule"]
    if r == "greedy":
        return c.argmax(1)
    T = cfg["T"]
    s = c / T + g
    if r == "gumbel":
        return s.argmax(1)
    if r == "topk":
        kth = c.topk(cfg["k"], dim=1).values[:, -1:]
        return torch.where(c >= kth, s, torch.full_like(s, -float("inf"))).argmax(1)
    if r == "minp":
        keep = c >= c.amax(1, keepdim=True) + T * math.log(cfg["p"])
        return torch.where(keep, s, torch.full_like(s, -float("inf"))).argmax(1)
    raise ValueError(r)


def refine_until_stable(L, U, cfg, g, newL, newU, read=None, max_iter=64):
    """Repeatedly ask the rule which rows it needs; replace their intervals with [newL, newU]
    (intersected with the old ones) until nothing new is needed. Returns (L, U, read_mask).
    States still unstable after max_iter read every row (always determines the decision)."""
    need_fn = RULES[cfg["rule"]]
    read = torch.zeros_like(L, dtype=torch.bool) if read is None else read.clone()
    for _ in range(max_iter):
        need = need_fn(L, U, cfg, g)
        new = need & ~read
        if not new.any():
            return L, U, read
        read |= new
        L = torch.where(new, torch.maximum(L, newL), L)
        U = torch.where(new, torch.minimum(U, newU), U)
    unstable = (need_fn(L, U, cfg, g) & ~read).any(1, keepdim=True)
    read = read | unstable
    L = torch.where(unstable, torch.maximum(L, newL), L)
    U = torch.where(unstable, torch.minimum(U, newU), U)
    return L, U, read


def certify(z_ref, zt, b, cfg, g, stage2=None):
    """One certificate on a chunk. zt, b: first-pass logits and bounds. stage2: optional (z8, b8).
    Returns S (rows read in stage 2, or None), R (rows read exactly), certified token."""
    L, U = zt - b, zt + b
    S = None
    if stage2 is not None:
        z8, b8 = stage2
        L, U, S = refine_until_stable(L, U, cfg, g, z8 - b8, z8 + b8)
    L, U, R = refine_until_stable(L, U, cfg, g, z_ref, z_ref)
    # every row's decision-relevant info is now determined; the winner is a read row
    tok = decide(torch.where(R, z_ref, torch.full_like(z_ref, -float("inf"))), cfg, g) \
        if cfg["rule"] in ("greedy", "gumbel") else _decide_with_intervals(z_ref, R, L, U, cfg, g)
    return S, R, tok


def _decide_with_intervals(z_ref, R, L, U, cfg, g):
    # For top-k / min-p the stable state guarantees: every race candidate is exact and its membership
    # certain. The winner is the best-scoring exact row that is certainly a member.
    cL, cU = softcap(L, cfg["cap"]), softcap(U, cfg["cap"])
    if cfg["rule"] == "topk":
        cin, _ = topk_membership(cL, cU, cfg["k"])
    else:
        T, lp = cfg["T"], math.log(cfg["p"])
        cin = cL >= cU.amax(1, keepdim=True) + T * lp
    s = softcap(z_ref, cfg["cap"]) / cfg["T"] + g
    return torch.where(R & cin, s, torch.full_like(s, -float("inf"))).argmax(1)


# ----------------------------------------------------------------------------------------------
# Counter-based Gumbel noise keyed by (prompt id, step, mode, token id)
# ----------------------------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def _fmix32(x):
    x = x & M32
    x = x ^ (x >> 16); x = (x * 0x85EBCA6B) & M32
    x = x ^ (x >> 13); x = (x * 0xC2B2AE35) & M32
    x = x ^ (x >> 16)
    return x


def gumbel(keys, V, device):
    """keys: int64 [B] (one per state). Returns Gumbel noise [B, V] in float32, identical on CPU/GPU."""
    tok = torch.arange(V, device=device, dtype=torch.int64)[None, :]
    a = _fmix32(keys[:, None] * 0x9E3779B1 + tok * 0x7FEB352D + 0x165667B1)
    b = _fmix32(a ^ (tok * 0x27D4EB2F + 0x61C88647))
    u = (b.double() + 0.5) / 2.0 ** 32       # in (0, 1)
    return (-torch.log(-torch.log(u))).float()


def state_key(prompt_id, step, mode):
    return (int(prompt_id) * 1_000_003 + int(step) * 7919 + (0 if mode == "greedy" else 1)) & 0x7FFFFFFF


# ----------------------------------------------------------------------------------------------
# Byte accounting (fractions of the BF16 head)
# ----------------------------------------------------------------------------------------------

def bytes_fraction(V, d, first_bpw, meta_per_group, n_groups, S_count, R_count, int8_bpr=None):
    bf16 = 2.0 * V * d
    first = first_bpw * V * d + meta_per_group * V * n_groups
    s = 0.0 if S_count is None else S_count * int8_bpr
    r = R_count * 2.0 * d
    return (first + s + r) / bf16
