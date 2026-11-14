"""Certified greedy step with the K2 state-direction correction, K = 1 (l4_corr_protocol.json).

  h . e_v = alpha r_v + (h - alpha p) . e_v,   alpha = h . p,   r_v = e_v . p (one FP32 number per row)

CorrHead adds r and an eager corrected step to l4_cert.CertHead. GraphStepCorr is l4_graph.GraphStep with a
first-pass kernel that subtracts alpha r_v and bounds only the remainder; the selection, INT8 and BF16 kernels
are l4_graph's, unchanged. The identity holds for whatever alpha is used, so alpha is rounded to FP32 once and
the same value goes into the correction and into h - alpha p.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

import a3_lib
import cert_core as cc
import l4_cert
import l4_graph
import l4_kernels as K

U32 = cc.U32
INFL = 1.0 + 1e-6                     # covers the FP32 rounding of h - alpha p in the variance term


class CorrHead(l4_cert.CertHead):
    def __init__(self, Wb, t, dirs):
        """dirs: {name: unit direction [d], float64}. use(name) picks one; use(None) is the uncorrected head."""
        super().__init__(Wb, t)
        dev = Wb.device
        W = Wb.float().cpu()
        Wt = a3_lib.shifted(W, self.mu64.cpu(), t)
        d4 = cc.make_int4_dither(Wt)                          # the copy CertHead packed (it asserts equality)
        names = list(dirs)
        Pm = torch.stack([dirs[k].double().cpu() for k in names], 1)
        r = torch.cat([((d4.Wq[i:i + 8192].double() - Wt[i:i + 8192].double()) @ Pm).float()
                       for i in range(0, self.V, 8192)])
        del W, Wt, d4
        self.dirs = {k: (dirs[k].double().to(dev), r[:, i].contiguous().to(dev)) for i, k in enumerate(names)}
        self.use(None)

    def use(self, name):
        self.active = name
        self.P, self.r = self.dirs[name] if name is not None else (None, None)

    def first_pass_bytes(self):
        return super().first_pass_bytes() + (4 * self.V if self.active is not None else 0)

    def first_pass(self, hb):
        """Corrected first-pass logits and bound for hb [B, D] (eager; the loop step and the checks use it)."""
        B = hb.shape[0]
        hf = hb.float()
        hd = hf.double()
        hg = hf.view(B, self.NG, self.G)
        alpha = (hd @ self.P).float()                         # [B], FP32: the value every later use takes
        hp = (hd - alpha.double()[:, None] * self.P[None, :]).float()
        UH = self._pad((hg * self.u.view(self.NG, self.G)).sum(-1))
        HSQ = self._pad((hp.view(B, self.NG, self.G) ** 2).sum(-1))
        c = (self.t * (hd @ self.mu64)).float() if self.t != 0 else torch.zeros(B, device=hb.device)
        Z, VAR = K.int4_head(self._pad(hb), UH, HSQ, self.Q, self.S, self._pad(c))
        corr = alpha[:, None] * self.r[None, :]
        Z, VAR = Z[:B] - corr, VAR[:B]
        hn = hf.norm(dim=1, keepdim=True)
        ext = 1.01 * U32 * hn * self.wt_norm[None] if self.t != 0 else 0.0
        b = self.csub * INFL * torch.sqrt(VAR) + self.gam * hn * self.nsum4[None] \
            + 2 * U32 * (Z.abs() + c.abs()[:, None]) + ext \
            + 3 * U32 * alpha.abs()[:, None] * self.r.abs()[None, :] + 2 * U32 * corr.abs()
        return Z, b, c, hf, hn

    def step(self, hb):
        if self.active is None:
            return super().step(hb)
        B = hb.shape[0]
        Z, b, c, hf, hn = self.first_pass(hb)
        H16 = self._pad(hb)
        h1 = hf.abs().sum(1, keepdim=True)
        L, U = Z - b, Z + b
        read8 = torch.zeros(self.V, device=hb.device, dtype=torch.bool)
        n8 = 0
        while True:                                            # INT8 stage, as CertHead.step
            need = ((U >= L.amax(1, keepdim=True)).any(0)) & ~read8
            idx = need.nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            read8[idx] = True
            n8 += idx.numel()
            z8 = K.gather_gemv(H16, self.Q8, idx, self.S8)[:B] + c[:, None]
            e8 = 1.01 * U32 * hn * self.wt_norm[idx][None] if self.t != 0 else 0.0
            b8 = torch.minimum(self.emax8[idx][None] * h1, self.enorm8[idx][None] * hn) \
                + self.gam * hn * self.nsum8[idx][None] + 2 * U32 * (z8.abs() + c.abs()[:, None]) + e8
            L[:, idx] = torch.maximum(L[:, idx], z8 - b8)
            U[:, idx] = torch.minimum(U[:, idx], z8 + b8)
        readR = torch.zeros(self.V, device=hb.device, dtype=torch.bool)
        nR = 0
        while True:                                            # BF16 stage
            need = ((U >= L.amax(1, keepdim=True)).any(0)) & ~readR
            idx = need.nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            readR[idx] = True
            nR += idx.numel()
            zr = K.gather_gemv(H16, self.Wb, idx)[:B]
            L[:, idx] = zr
            U[:, idx] = zr
        tok = torch.where(readR[None], L, torch.full_like(L, -math.inf)).argmax(1)
        return tok, n8, nR


@triton.jit
def _first_pass_corr(H, UH, HSQ, Q, S, L, U, LMAX, C, HN, NSUM, WTN, RV, ALPHA, V, csub, gam, ext_on,
                     B: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BLOCK_V: tl.constexpr):
    """l4_graph._first_pass with the correction: z = acc + c - alpha r, and two more slack terms."""
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
    al = tl.load(ALPHA + b)
    rv = tl.load(RV + rows, mask=rmask, other=0.0)
    corr = al[:, None] * rv[None, :]
    z = acc + c[:, None] - corr
    nsum = tl.load(NSUM + rows, mask=rmask, other=0.0)
    wtn = tl.load(WTN + rows, mask=rmask, other=0.0)
    bnd = csub * tl.sqrt(var) + gam * hn[:, None] * nsum[None, :] \
        + 2 * 5.960464477539063e-08 * (tl.abs(z) + tl.abs(c)[:, None]) \
        + ext_on * 1.01 * 5.960464477539063e-08 * hn[:, None] * wtn[None, :] \
        + 3 * 5.960464477539063e-08 * tl.abs(al)[:, None] * tl.abs(rv)[None, :] \
        + 2 * 5.960464477539063e-08 * tl.abs(corr)
    lo_v = z - bnd
    out = b[:, None] * V + rows[None, :]
    tl.store(L + out, lo_v, mask=rmask[None, :])
    tl.store(U + out, z + bnd, mask=rmask[None, :])
    m = tl.max(tl.where(rmask[None, :], lo_v, float("-inf")), axis=1)
    tl.atomic_max(LMAX + b, m, mask=b < B)


class GraphStepCorr(l4_graph.GraphStep):
    """l4_graph.GraphStep with the corrected first pass. The head's active direction is fixed at construction."""

    def __init__(self, head, B, **kw):
        assert head.active is not None
        super().__init__(head, B, **kw)
        self.P, self.r = head.P, head.r
        self.ALPHA = torch.zeros(16, device=head.Wb.device, dtype=torch.float32)

    def _body(self):
        h, B, V, D = self.h, self.B, self.h.V, self.h.D
        cf = self.cfg
        hf = self.hb.float()
        hd = hf.double()
        hg = hf.view(B, h.NG, h.G)
        self.H16[:B].copy_(self.hb)
        self.ALPHA[:B].copy_((hd @ self.P).float())
        hp = (hd - self.ALPHA[:B].double()[:, None] * self.P[None, :]).float()
        self.UH[:B].copy_((hg * h.u.view(h.NG, h.G)).sum(-1))
        self.HSQ[:B].copy_((hp.view(B, h.NG, h.G) ** 2).sum(-1))
        if h.t != 0:
            self.C[:B].copy_((h.t * (hd @ h.mu64)).float())
        self.HN[:B].copy_(hf.norm(dim=1))
        self.H1[:B].copy_(hf.abs().sum(1))
        self.LMAX.fill_(float("-inf"))
        self.CNT.zero_()
        self.ZR.fill_(float("-inf"))
        ext = 1.0 if h.t != 0 else 0.0
        _first_pass_corr[(triton.cdiv(V, cf["block_v"]),)](
            self.H16, self.UH, self.HSQ, h.Q, h.S, self.L, self.U, self.LMAX, self.C, self.HN, h.nsum4, h.wt_norm,
            self.r, self.ALPHA, V, h.csub * INFL, h.gam, ext, B=B, D=D, G=128, BLOCK_V=cf["block_v"],
            num_warps=4, num_stages=3)
        l4_graph._select_all[(triton.cdiv(V, cf["block_sel"]),)](
            self.U, self.LMAX, self.IDX8, self.CNT[0:1], V, self.cap8, B=B, BLOCK=cf["block_sel"], num_warps=4)
        l4_graph._int8_stage[(triton.cdiv(self.cap8, cf["block_n"]),)](
            self.H16, h.Q8, h.S8, h.emax8, h.enorm8, h.nsum8, h.wt_norm, self.IDX8, self.CNT[0:1], self.L, self.U,
            self.LMAX, self.C, self.HN, self.H1, V, self.cap8, h.gam, ext,
            B=B, D=D, BLOCK_N=cf["block_n"], BLOCK_K=cf["block_k"], num_warps=4)
        l4_graph._select_from[(triton.cdiv(self.cap8, cf["block_sel"]),)](
            self.U, self.LMAX, self.IDX8, self.CNT[0:1], self.cap8, self.IDXR, self.CNT[1:2], self.capr, V,
            B=B, BLOCK=cf["block_sel"], num_warps=4)
        l4_graph._bf16_stage[(triton.cdiv(self.capr, cf["block_n"]),)](
            self.H16, h.Wb, self.IDXR, self.CNT[1:2], self.capr, self.ZR, B=B, D=D,
            BLOCK_N=cf["block_n"], BLOCK_K=cf["block_k"], num_warps=4)
        z = self.ZR[:B]
        valid = self.ids[None, :] < self.CNT[1].clamp(max=self.capr)
        z = torch.where(valid, z, torch.full_like(z, float("-inf")))
        best = z.amax(1, keepdim=True)
        tokid = torch.where(valid & (z == best), self.IDXR.long()[None, :], torch.full_like(z, V, dtype=torch.long))
        self.out[:B].copy_(tokid.amin(1))
        self.out[B:].copy_(self.CNT.long())
