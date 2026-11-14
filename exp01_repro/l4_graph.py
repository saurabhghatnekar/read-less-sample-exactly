"""Single-sync certified greedy step, capturable in a CUDA graph.

  1. fused first pass: dithered INT4 logits, sub-Gaussian bound, L and U, and LMAX[b] = max_v L[b, v] (atomic max)
  2. select: rows with U[b, v] >= LMAX[b] for any b, compacted on the device into IDX8 (capacity CAP8)
  3. INT8 stage on IDX8: tighten L and U, raise LMAX
  4. select from IDX8 the rows still needed into IDXR (capacity CAPR)
  5. BF16 stage on IDXR: exact logits into ZR; token = largest ZR, ties to the lowest token id
One round per stage suffices: LMAX only rises, so a row needed after a stage was already needed before it.
If a buffer overflows the result is flagged and the caller falls back to CertHead.step.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

U32 = 2.0 ** -24


@triton.jit
def _first_pass(H, UH, HSQ, Q, S, L, U, LMAX, C, HN, NSUM, WTN, V, csub, gam, ext_on,
                B: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    rmask = rows < V
    b = tl.arange(0, 16)
    k2 = tl.arange(0, G // 2)
    NG: tl.constexpr = D // G
    acc = tl.zeros((16, BLOCK_V), dtype=tl.float32)
    var = tl.zeros((16, BLOCK_V), dtype=tl.float32)
    for g in range(NG):
        qb = tl.load(Q + rows[:, None] * (D // 2) + g * (G // 2) + k2[None, :], mask=rmask[:, None], other=0x88)
        lo = ((qb & 0xF).to(tl.int8) - 8).to(tl.bfloat16)
        hi = ((qb >> 4).to(tl.int8) - 8).to(tl.bfloat16)
        he = tl.load(H + b[:, None] * D + g * G + 2 * k2[None, :])
        ho = tl.load(H + b[:, None] * D + g * G + 2 * k2[None, :] + 1)
        part = tl.dot(he, tl.trans(lo)) + tl.dot(ho, tl.trans(hi))
        s = tl.load(S + rows * NG + g, mask=rmask, other=0.0).to(tl.float32)
        uh = tl.load(UH + b * NG + g)
        hsq = tl.load(HSQ + b * NG + g)
        acc += (part - uh[:, None]) * s[None, :]
        var += hsq[:, None] * (s * s)[None, :]
    c = tl.load(C + b)
    hn = tl.load(HN + b)
    z = acc + c[:, None]
    nsum = tl.load(NSUM + rows, mask=rmask, other=0.0)
    wtn = tl.load(WTN + rows, mask=rmask, other=0.0)
    bnd = csub * tl.sqrt(var) + gam * hn[:, None] * nsum[None, :] \
        + 2 * 5.960464477539063e-08 * (tl.abs(z) + tl.abs(c)[:, None]) \
        + ext_on * 1.01 * 5.960464477539063e-08 * hn[:, None] * wtn[None, :]
    lo_v = z - bnd
    out = b[:, None] * V + rows[None, :]
    tl.store(L + out, lo_v, mask=rmask[None, :])
    tl.store(U + out, z + bnd, mask=rmask[None, :])
    m = tl.max(tl.where(rmask[None, :], lo_v, float("-inf")), axis=1)
    tl.atomic_max(LMAX + b, m, mask=b < B)


@triton.jit
def _select_all(U, LMAX, IDX, CNT, V, CAP, B: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    need = tl.zeros((BLOCK,), dtype=tl.int32)
    for bb in tl.static_range(B):
        lm = tl.load(LMAX + bb)
        u = tl.load(U + bb * V + offs, mask=m, other=float("-inf"))
        need = need | (u >= lm).to(tl.int32)
    need = need & m.to(tl.int32)
    base = tl.atomic_add(CNT, tl.sum(need, axis=0))
    pos = base + tl.cumsum(need, axis=0) - 1
    tl.store(IDX + pos, offs, mask=(need > 0) & (pos < CAP))


@triton.jit
def _int8_stage(H, Q8, S8, E8MAX, E8NORM, NSUM8, WTN, IDX, CNT, L, U, LMAX, C, HN, H1, V, CAP, gam, ext_on,
                B: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.minimum(tl.load(CNT), CAP)
    if pid * BLOCK_N >= n:
        return
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs < n
    rows = tl.load(IDX + offs, mask=m, other=0).to(tl.int64)
    b = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(Q8 + rows[:, None] * D + k[None, :], mask=m[:, None], other=0).to(tl.bfloat16)
        h = tl.load(H + b[:, None] * D + k[None, :])
        acc += tl.dot(h, tl.trans(w))
    c = tl.load(C + b)
    hn = tl.load(HN + b)
    h1 = tl.load(H1 + b)
    z8 = acc * tl.load(S8 + rows, mask=m, other=0.0)[None, :] + c[:, None]
    b8 = tl.minimum(tl.load(E8MAX + rows, mask=m, other=0.0)[None, :] * h1[:, None],
                    tl.load(E8NORM + rows, mask=m, other=0.0)[None, :] * hn[:, None]) \
        + gam * hn[:, None] * tl.load(NSUM8 + rows, mask=m, other=0.0)[None, :] \
        + 2 * 5.960464477539063e-08 * (tl.abs(z8) + tl.abs(c)[:, None]) \
        + ext_on * 1.01 * 5.960464477539063e-08 * hn[:, None] * tl.load(WTN + rows, mask=m, other=0.0)[None, :]
    ptr = b[:, None].to(tl.int64) * V + rows[None, :]
    mm = m[None, :] & (b[:, None] < B)
    lo_new = tl.maximum(tl.load(L + ptr, mask=mm, other=float("-inf")), z8 - b8)
    up_new = tl.minimum(tl.load(U + ptr, mask=mm, other=float("inf")), z8 + b8)
    tl.store(L + ptr, lo_new, mask=mm)
    tl.store(U + ptr, up_new, mask=mm)
    tl.atomic_max(LMAX + b, tl.max(tl.where(mm, lo_new, float("-inf")), axis=1), mask=b < B)


@triton.jit
def _select_from(U, LMAX, IDX8, CNT8, CAP8, IDXR, CNTR, CAPR, V, B: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.minimum(tl.load(CNT8), CAP8)
    if pid * BLOCK >= n:
        return
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    rows = tl.load(IDX8 + offs, mask=m, other=0)
    need = tl.zeros((BLOCK,), dtype=tl.int32)
    for bb in tl.static_range(B):
        u = tl.load(U + bb * V + rows.to(tl.int64), mask=m, other=float("-inf"))
        need = need | (u >= tl.load(LMAX + bb)).to(tl.int32)
    need = need & m.to(tl.int32)
    base = tl.atomic_add(CNTR, tl.sum(need, axis=0))
    pos = base + tl.cumsum(need, axis=0) - 1
    tl.store(IDXR + pos, rows, mask=(need > 0) & (pos < CAPR))


@triton.jit
def _bf16_stage(H, W, IDXR, CNTR, CAPR, ZR, B: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.minimum(tl.load(CNTR), CAPR)
    if pid * BLOCK_N >= n:
        return
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs < n
    rows = tl.load(IDXR + offs, mask=m, other=0).to(tl.int64)
    b = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(W + rows[:, None] * D + k[None, :], mask=m[:, None], other=0)
        h = tl.load(H + b[:, None] * D + k[None, :])
        acc += tl.dot(h, tl.trans(w))
    tl.store(ZR + b[:, None] * CAPR + offs[None, :], acc, mask=m[None, :] & (b[:, None] < B))


class GraphStep:
    """Static-shape certified greedy step for batch size B, built on a CertHead's tensors."""

    def __init__(self, head, B, cap8=None, capr=1024, block_v=64, block_n=32, block_k=256, block_sel=1024):
        cap8 = head.V if cap8 is None else cap8            # whole vocabulary: the INT8 stage cannot overflow
        self.h, self.B, self.cap8, self.capr = head, B, cap8, capr
        self.cfg = dict(block_v=block_v, block_n=block_n, block_k=block_k, block_sel=block_sel)
        dev, V, D, NG = head.Wb.device, head.V, head.D, head.NG
        f32 = dict(device=dev, dtype=torch.float32)
        self.hb = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)          # input buffer
        self.H16 = torch.zeros(16, D, device=dev, dtype=torch.bfloat16)
        self.UH, self.HSQ = torch.zeros(16, NG, **f32), torch.zeros(16, NG, **f32)
        self.C, self.HN, self.H1 = torch.zeros(16, **f32), torch.zeros(16, **f32), torch.zeros(16, **f32)
        self.L, self.U = torch.empty(16, V, **f32), torch.empty(16, V, **f32)
        self.LMAX = torch.empty(16, **f32)
        self.IDX8 = torch.empty(cap8, device=dev, dtype=torch.int32)
        self.IDXR = torch.empty(capr, device=dev, dtype=torch.int32)
        self.CNT = torch.zeros(2, device=dev, dtype=torch.int32)
        self.ZR = torch.empty(16, capr, **f32)
        self.out = torch.zeros(B + 2, device=dev, dtype=torch.int64)            # tokens, |IDX8|, |IDXR|
        self.ids = torch.arange(capr, device=dev)
        self.graph = None

    def _body(self):
        h, B, V, D = self.h, self.B, self.h.V, self.h.D
        cf = self.cfg
        hf = self.hb.float()
        hg = hf.view(B, h.NG, h.G)
        self.H16[:B].copy_(self.hb)
        self.UH[:B].copy_((hg * h.u.view(h.NG, h.G)).sum(-1))
        self.HSQ[:B].copy_((hg * hg).sum(-1))
        if h.t != 0:
            self.C[:B].copy_((h.t * (hf.double() @ h.mu64)).float())
        self.HN[:B].copy_(hf.norm(dim=1))
        self.H1[:B].copy_(hf.abs().sum(1))
        self.LMAX.fill_(float("-inf"))
        self.CNT.zero_()
        self.ZR.fill_(float("-inf"))
        ext = 1.0 if h.t != 0 else 0.0
        _first_pass[(triton.cdiv(V, cf["block_v"]),)](
            self.H16, self.UH, self.HSQ, h.Q, h.S, self.L, self.U, self.LMAX, self.C, self.HN, h.nsum4, h.wt_norm,
            V, h.csub, h.gam, ext, B=B, D=D, G=128, BLOCK_V=cf["block_v"], num_warps=4, num_stages=3)
        _select_all[(triton.cdiv(V, cf["block_sel"]),)](self.U, self.LMAX, self.IDX8, self.CNT[0:1], V, self.cap8,
                                                        B=B, BLOCK=cf["block_sel"], num_warps=4)
        _int8_stage[(triton.cdiv(self.cap8, cf["block_n"]),)](
            self.H16, h.Q8, h.S8, h.emax8, h.enorm8, h.nsum8, h.wt_norm, self.IDX8, self.CNT[0:1], self.L, self.U,
            self.LMAX, self.C, self.HN, self.H1, V, self.cap8, h.gam, ext,
            B=B, D=D, BLOCK_N=cf["block_n"], BLOCK_K=cf["block_k"], num_warps=4)
        _select_from[(triton.cdiv(self.cap8, cf["block_sel"]),)](
            self.U, self.LMAX, self.IDX8, self.CNT[0:1], self.cap8, self.IDXR, self.CNT[1:2], self.capr, V,
            B=B, BLOCK=cf["block_sel"], num_warps=4)
        _bf16_stage[(triton.cdiv(self.capr, cf["block_n"]),)](
            self.H16, h.Wb, self.IDXR, self.CNT[1:2], self.capr, self.ZR, B=B, D=D,
            BLOCK_N=cf["block_n"], BLOCK_K=cf["block_k"], num_warps=4)
        z = self.ZR[:B]
        valid = self.ids[None, :] < self.CNT[1].clamp(max=self.capr)
        z = torch.where(valid, z, torch.full_like(z, float("-inf")))
        best = z.amax(1, keepdim=True)
        tokid = torch.where(valid & (z == best), self.IDXR.long()[None, :], torch.full_like(z, V, dtype=torch.long))
        self.out[:B].copy_(tokid.amin(1))
        self.out[B:].copy_(self.CNT.long())

    def capture(self, warm=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warm):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    def step(self, hb, use_graph=True):
        """Returns (tokens [B] on the CPU, rows read in INT8, rows read in BF16, fell_back). One host sync."""
        self.hb.copy_(hb)
        if use_graph and self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        o = self.out.cpu()
        n8, nr = int(o[self.B]), int(o[self.B + 1])
        if n8 > self.cap8 or nr > self.capr:
            tok, a, r = self.h.step(hb)
            return tok.cpu(), a, r, True
        return o[:self.B], n8, nr, False
