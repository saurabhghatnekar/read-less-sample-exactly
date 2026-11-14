"""Certified greedy step on GPU: fused dithered-INT4 first pass (l4_kernels.int4_head) with the sub-Gaussian bound,
an INT8 stage and a BF16 stage that read only the rows still in question (l4_kernels.gather_gemv).

The head copies are exactly the B1 ones (cert_core quantizers on W_t from a3_lib.shifted); build() checks that the
packed codes dequantize to cert_core's Wq bit for bit.
"""
from __future__ import annotations

import math

import torch

import a3_lib
import cert_core as cc
import l4_kernels as K
import subg_lib

U32 = cc.U32


class CertHead:
    def __init__(self, Wb, t):
        """Wb: BF16 head [V, D] on the GPU (the reference). t: mean-row shift for the low-bit copies."""
        self.Wb, self.t = Wb, t
        V, D = Wb.shape
        self.V, self.D, self.G, self.NG = V, D, 128, D // 128
        dev = Wb.device
        W = Wb.float().cpu()                                  # build on the CPU: the copies do not fit an L4 at once
        mu64 = a3_lib.mean_row(W)
        self.mu64 = mu64
        Wt = a3_lib.shifted(W, mu64, t)
        d4 = cc.make_int4_dither(Wt)
        s = d4.scale                                          # [V, NG], FP16 values held as fp32
        u = d4.dither                                         # [D]
        codes = torch.round(Wt.view(V, self.NG, self.G) / s[..., None] + u.view(1, self.NG, self.G)).clamp(-8, 7)
        deq = ((codes - u.view(1, self.NG, self.G)) * s[..., None]).view(V, D)
        assert torch.equal(deq, d4.Wq), "packed codes do not reproduce cert_core's dithered head"
        self.Q = K.pack_int4(codes.view(V, D).to(torch.int16))
        self.S = s.half().contiguous()
        self.u = u
        # Slack for our kernel's arithmetic and for the reference product: gamma_{2D} ||h|| (||W|| + ||(|q|+1) s||).
        absq = (codes.abs() + 1) * s[..., None]
        self.nsum4 = W.norm(dim=1) + absq.view(V, D).norm(dim=1)
        del codes, deq, absq
        d8 = cc.make_int8_row(Wt)
        s8 = (Wt.abs().amax(1, keepdim=True) / 127).half().float().clamp_min(1e-12)
        q8 = torch.round(Wt / s8).clamp(-127, 127)
        assert torch.equal(q8 * s8, d8.Wq), "INT8 codes do not reproduce cert_core's INT8 head"
        self.Q8, self.S8 = q8.to(torch.int8).contiguous(), s8.squeeze(1).contiguous()
        self.emax8, self.enorm8 = d8.emax.squeeze(1), d8.enorm.squeeze(1)
        self.nsum8 = W.norm(dim=1) + d8.Wq.norm(dim=1)
        self.wt_norm = Wt.norm(dim=1)
        del W, Wt, d4, d8, q8
        for k in ("mu64", "Q", "S", "u", "nsum4", "Q8", "S8", "emax8", "enorm8", "nsum8", "wt_norm"):
            setattr(self, k, getattr(self, k).to(dev))
        torch.cuda.empty_cache()
        self.csub = subg_lib.subg_c()
        self.gam = 2 * D * U32 / (1 - 2 * D * U32)

    def first_pass_bytes(self):
        return self.Q.numel() + 2 * self.S.numel()

    def _pad(self, x):
        out = torch.zeros((16,) + x.shape[1:], device=x.device, dtype=x.dtype)
        out[: x.shape[0]] = x
        return out

    def step(self, hb):
        """hb: BF16 states [B, D], B <= 16. Returns (tokens [B], rows read in INT8, rows read in BF16).
        Rows read are the union over the batch, since a row is fetched once for all states."""
        B = hb.shape[0]
        hf = hb.float()
        hg = hf.view(B, self.NG, self.G)
        H16 = self._pad(hb)
        UH = self._pad((hg * self.u.view(self.NG, self.G)).sum(-1))
        HSQ = self._pad((hg * hg).sum(-1))
        c = (self.t * (hf.double() @ self.mu64)).float() if self.t != 0 else torch.zeros(B, device=hb.device)
        Z, VAR = K.int4_head(H16, UH, HSQ, self.Q, self.S, self._pad(c))
        Z, VAR = Z[:B], VAR[:B]
        hn = hf.norm(dim=1, keepdim=True)
        h1 = hf.abs().sum(1, keepdim=True)
        ext = 1.01 * U32 * hn * self.wt_norm[None] if self.t != 0 else 0.0
        b = self.csub * torch.sqrt(VAR) + self.gam * hn * self.nsum4[None] + 2 * U32 * (Z.abs() + c.abs()[:, None]) + ext
        L, U = Z - b, Z + b
        V = self.V
        read8 = torch.zeros(V, device=hb.device, dtype=torch.bool)
        n8 = 0
        while True:                                            # INT8 stage, repeated until no new row is needed
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
        readR = torch.zeros(V, device=hb.device, dtype=torch.bool)
        nR = 0
        while True:                                            # BF16 stage: exact rows
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
