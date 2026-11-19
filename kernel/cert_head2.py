"""The head copies for the unified kernel (fused2.py): the certificate of the byte runs, stored as a kernel reads it.

  first pass   dithered INT4 g128 of W_t = W - t 1 mu^T, packed two per byte, FP16 group steps (cert_core.make_int4_dither)
  INT8 stage   dithered INT8 g128 of W_t, FP16 group steps (mix_lib.make_mixed_dither with every group in INT8)
  K2           one FP32 number per row and copy, r = (W~ - W_t) . p, for a unit direction p (optional)
  exact        the BF16 rows of W

Bound constant: the sub-Gaussian one, sqrt(ln(2/delta)/6), times 1 + 1e-6 for the rounding of the bound's own
arithmetic (as hdir_lib.corrected). The quantised values equal cert_core's and mix_lib's bit for bit (test_cert_head2.py).
"""
import math

import torch

G = 128
U32 = 2.0 ** -24


class CertHead2:
    def __init__(self, W_bf16, t=0.0, p=None, seed=1234, delta=1e-15, chunk=16384):
        V, d = W_bf16.shape
        dev = W_bf16.device
        self.V, self.d, self.ng, self.t, self.dev = V, d, d // G, float(t), dev
        self.W = W_bf16.contiguous()
        self.mu64 = self.W.double().mean(0)                       # a3_lib.mean_row
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.u = (torch.rand(d, generator=gen, dtype=torch.float64) - 0.5).float().to(dev)
        self.C = math.sqrt(math.log(2.0 / delta) / 6.0) * (1 + 1e-6)
        self.gamma = d * U32 / (1 - d * U32)
        self.p = None if p is None else (p.double() / p.double().norm()).to(dev)
        f32 = dict(device=dev, dtype=torch.float32)
        self.packed = torch.empty(V, d // 2, dtype=torch.uint8, device=dev)
        self.s4 = torch.empty(V, self.ng, dtype=torch.float16, device=dev)
        self.q8 = torch.empty(V, d, dtype=torch.int8, device=dev)
        self.s8 = torch.empty(V, self.ng, dtype=torch.float16, device=dev)
        self.w_norm, self.wt_norm, self.wq4_norm, self.wq8_norm = (torch.empty(V, **f32) for _ in range(4))
        self.r4, self.r8 = torch.zeros(V, **f32), torch.zeros(V, **f32)
        ug = self.u.view(1, self.ng, G)
        for r0 in range(0, V, chunk):
            w = self.W[r0:r0 + chunk].float()
            n = w.shape[0]
            wt = (w.double() - self.t * self.mu64).float() if self.t != 0 else w          # a3_lib.shifted
            wg = wt.view(n, self.ng, G)
            amax = wg.abs().amax(-1, keepdim=True)
            s_h = (amax / 6.5).half()
            s = s_h.float().clamp_min(1e-12)
            q = torch.round(wg / s + ug)
            assert q.max() <= 7 and q.min() >= -8, "dither clipped"     # strict: the clamp below must change nothing
            q = torch.clamp(q, -8, 7)
            wq4 = ((q - ug) * s).view(n, d)
            qq = (q + 8).to(torch.uint8).view(n, d)
            self.packed[r0:r0 + n] = qq[:, 0::2] | (qq[:, 1::2] << 4)
            self.s4[r0:r0 + n] = s_h.squeeze(-1)
            s8_h = (amax / 126.5).half()
            s8 = s8_h.float().clamp_min(1e-12)
            q8 = torch.round(wg / s8 + ug)
            assert q8.abs().max() <= 127, "INT8 dither clipped"
            wq8 = ((q8 - ug) * s8).view(n, d)
            self.q8[r0:r0 + n] = q8.to(torch.int8).view(n, d)
            self.s8[r0:r0 + n] = s8_h.squeeze(-1)
            self.w_norm[r0:r0 + n] = w.norm(dim=1)
            self.wt_norm[r0:r0 + n] = wt.norm(dim=1)
            self.wq4_norm[r0:r0 + n] = wq4.norm(dim=1)
            self.wq8_norm[r0:r0 + n] = wq8.norm(dim=1)
            if self.p is not None:
                self.r4[r0:r0 + n] = ((wq4.double() - wt.double()) @ self.p).float()
                self.r8[r0:r0 + n] = ((wq8.double() - wt.double()) @ self.p).float()

    # dequantised copies, for the tests and the reference implementation
    def dequant4(self, r0, r1):
        pk = self.packed[r0:r1]
        n = pk.shape[0]
        q = torch.stack((pk & 15, pk >> 4), -1).view(n, self.ng, G).float() - 8
        return ((q - self.u.view(1, self.ng, G)) * self.s4[r0:r1].float().clamp_min(1e-12)[:, :, None]).view(n, self.d)

    def dequant8(self, r0, r1):
        q = self.q8[r0:r1].float().view(-1, self.ng, G)
        return ((q - self.u.view(1, self.ng, G)) * self.s8[r0:r1].float().clamp_min(1e-12)[:, :, None]).view(-1, self.d)

    def state_inputs(self, h):
        """Per-state numbers the kernels take. h: [B, d] float32 holding BF16 values."""
        B = h.shape[0]
        hd = h.double()
        csh = (self.t * (hd @ self.mu64)).float() if self.t != 0 else torch.zeros(B, device=h.device)
        if self.p is not None:
            alpha = hd @ self.p
            hp = (hd - alpha[:, None] * self.p[None, :]).float()
            alpha = alpha.float()
        else:
            alpha, hp = torch.zeros(B, device=h.device), h
        hg = h.view(B, self.ng, G)
        ug = self.u.view(self.ng, G)
        return dict(csh=csh.contiguous(), alpha=alpha.contiguous(),
                    h2sq=(hp.view(B, self.ng, G).norm(dim=-1) ** 2).contiguous(),      # of h - alpha p: the bound's state
                    cu=(hg * ug).sum(-1).contiguous(), uh=(hg.norm(dim=-1) * ug.norm(dim=1)).contiguous(),
                    hn=h.norm(dim=1).contiguous())

    def bytes_first(self):
        return self.V * (self.d // 2 + 2 * self.ng + (4 if self.p is not None else 0))

    def bytes_int8_row(self):
        return self.d + 2 * self.ng + 4 + (4 if self.p is not None else 0)
