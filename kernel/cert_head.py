"""Three-pass certified LM head in plain PyTorch: the timing baseline for the Triton kernel.

  Pass 1  every row, dithered INT4 g128 -> approximate logits z~ and the Hoeffding bound (+ FP32 slack).
  Pass 2  rows the rule still needs, re-read in INT8 (one FP16 scale per row) -> tighter intervals. Repeat until stable.
  Pass 3  rows still needed, read in BF16 and computed in FP32. Repeat until stable. The token is the rule's choice
          among the exactly read rows, which the stable state guarantees is the reference token.

Weights are stored the way a kernel would read them: INT4 packed two per byte with FP16 group scales, INT8 with an
FP16 row scale, BF16 rows. The quantised values equal cert_core.make_int4_dither and make_int8_row bit for bit.

Within a batch each later pass reads the union of the rows any state needs, and every state uses those reads.
Intersecting a state's interval with another valid interval only tightens it, so this stays sound.

Rules: greedy and Gumbel (cert_core's need functions and noise). Top-k and min-p are not wired in yet.
"""
import math
import time

import torch

G = 128
U32 = 2.0 ** -24


def softcap(z, cap):
    return z if cap is None else cap * torch.tanh(z / cap)


class CertHead:
    def __init__(self, W_bf16, seed=1234, delta=1e-15, chunk=16384):
        V, d = W_bf16.shape
        dev = W_bf16.device
        self.V, self.d, self.ng, self.chunk, self.dev = V, d, d // G, chunk, dev
        self.W = W_bf16.contiguous()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.u = (torch.rand(d, generator=gen, dtype=torch.float64) - 0.5).float().to(dev)
        self.C = math.sqrt(math.log(2.0 / delta) / 2.0)
        self.gamma = d * U32 / (1 - d * U32)
        f32 = dict(device=dev, dtype=torch.float32)
        self.packed = torch.empty(V, d // 2, dtype=torch.uint8, device=dev)
        self.s4 = torch.empty(V, self.ng, dtype=torch.float16, device=dev)
        self.q8 = torch.empty(V, d, dtype=torch.int8, device=dev)
        self.s8 = torch.empty(V, dtype=torch.float16, device=dev)
        self.w_norm, self.wq4_norm, self.wq8_norm = (torch.empty(V, **f32) for _ in range(3))
        self.emax8, self.enorm8 = torch.empty(V, **f32), torch.empty(V, **f32)
        ug = self.u.view(1, self.ng, G)
        for r0 in range(0, V, chunk):
            w = self.W[r0:r0 + chunk].float()
            n = w.shape[0]
            wg = w.view(n, self.ng, G)
            # INT4 with subtractive dither (same ops, same order as cert_core.make_int4_dither)
            s_h = (wg.abs().amax(-1, keepdim=True) / 6.5).half()
            s = s_h.float().clamp_min(1e-12)
            q = torch.round(wg / s + ug)
            assert q.max() <= 7 and q.min() >= -8, "dither would clip"
            wq4 = ((q - ug) * s).view(n, d)
            qq = (q + 8).to(torch.uint8).view(n, d)
            self.packed[r0:r0 + n] = qq[:, 0::2] | (qq[:, 1::2] << 4)
            self.s4[r0:r0 + n] = s_h.squeeze(-1)
            # INT8, one scale per row (cert_core.make_int8_row)
            s8_h = (w.abs().amax(1, keepdim=True) / 127).half()
            s8 = s8_h.float().clamp_min(1e-12)
            q8 = torch.clamp(torch.round(w / s8), -127, 127)
            wq8 = q8 * s8
            self.q8[r0:r0 + n] = q8.to(torch.int8)
            self.s8[r0:r0 + n] = s8_h.squeeze(-1)
            e8 = wq8 - w
            self.emax8[r0:r0 + n], self.enorm8[r0:r0 + n] = e8.abs().amax(1), e8.norm(dim=1)
            self.w_norm[r0:r0 + n] = w.norm(dim=1)
            self.wq4_norm[r0:r0 + n] = wq4.norm(dim=1)
            self.wq8_norm[r0:r0 + n] = wq8.norm(dim=1)
        self.s4_eff = self.s4.float().clamp_min(1e-12)
        self.s2 = self.s4_eff ** 2

    # bytes each pass reads (weights and per-row metadata), for bandwidth arithmetic
    def bytes_pass1(self):
        return self.V * (self.d // 2 + 2 * self.ng + 8)          # packed INT4, FP16 scales, two FP32 norms

    def bytes_int8_row(self):
        return self.d + 2 + 12                                    # INT8 row, FP16 scale, emax/enorm/norm

    def bytes_bf16_row(self):
        return 2 * self.d

    # ---------------------------------------------------------------------------------------------- passes
    def dequant4(self, r0, r1):
        p = self.packed[r0:r1]
        n = p.shape[0]
        q = torch.stack((p & 15, p >> 4), -1).view(n, self.ng, G).float() - 8
        return ((q - self.u.view(1, self.ng, G)) * self.s4_eff[r0:r1, :, None]).view(n, self.d)

    def pass1(self, h):
        """h: [B, d] float32 holding BF16 values. Returns z~ [B, V] and the bound b [B, V]."""
        B = h.shape[0]
        z = torch.empty(B, self.V, device=h.device)
        for r0 in range(0, self.V, self.chunk):
            r1 = min(r0 + self.chunk, self.V)
            z[:, r0:r1] = h @ self.dequant4(r0, r1).T
        return z

    def bound1(self, h):
        h2sq = h.view(h.shape[0], self.ng, G).pow(2).sum(-1)
        hn = h.norm(dim=1, keepdim=True)
        return self.C * torch.sqrt(h2sq @ self.s2.T) + self.gamma * hn * (self.w_norm + self.wq4_norm)

    def read_int8(self, h, cols):
        w = self.q8[cols].float() * self.s8[cols].float().clamp_min(1e-12)[:, None]
        z = h @ w.T
        h1 = h.abs().sum(1, keepdim=True)
        hn = h.norm(dim=1, keepdim=True)
        b = torch.minimum(h1 * self.emax8[cols], hn * self.enorm8[cols]) \
            + self.gamma * hn * (self.w_norm[cols] + self.wq8_norm[cols])
        return z - b, z + b

    def read_exact(self, h, cols):
        z = h @ self.W[cols].float().T
        return z, z

    @staticmethod
    def refine(L, U, need_fn, cfg, g, read, read_fn):
        """Read the union of needed rows until the rule needs nothing new. read: [V] bool, updated in place.
        Returns the list of column tensors read and their (lo, hi) values."""
        got = []
        while True:
            need = need_fn(L, U, cfg, g) & ~read[None, :]
            cols = need.any(0).nonzero().squeeze(1)
            if cols.numel() == 0:
                return got
            lo, hi = read_fn(cols)
            L[:, cols] = torch.maximum(L[:, cols], lo)
            U[:, cols] = torch.minimum(U[:, cols], hi)
            read[cols] = True
            got.append((cols, lo))

    def certify(self, h, g, cfg, cc, sync=lambda: None, p1=None):
        """Returns token [B], rows read in INT8 (union), rows read exactly (union), stage times in seconds.
        p1: optional callable h -> (z~, bound) replacing the PyTorch pass 1 (e.g. int4_pass1.Pass1)."""
        need_fn = cc.RULES[cfg["rule"]]
        t = [time.perf_counter()]
        if p1 is None:
            z = self.pass1(h)
        else:
            z, b = p1(h)
        sync(); t.append(time.perf_counter())
        if p1 is None:
            b = self.bound1(h)
        L, U = z - b, z + b
        sync(); t.append(time.perf_counter())
        read8 = torch.zeros(self.V, dtype=torch.bool, device=h.device)
        self.refine(L, U, need_fn, cfg, g, read8, lambda c: self.read_int8(h, c))
        sync(); t.append(time.perf_counter())
        readR = torch.zeros(self.V, dtype=torch.bool, device=h.device)
        got = self.refine(L, U, need_fn, cfg, g, readR, lambda c: self.read_exact(h, c))
        s = torch.full_like(L, -float("inf"))
        for cols, zc in got:
            sc = softcap(zc, cfg["cap"])
            s[:, cols] = sc if cfg["rule"] == "greedy" else sc / cfg["T"] + g[:, cols]
        tok = s.argmax(1)
        sync(); t.append(time.perf_counter())
        times = dict(pass1_read=t[1] - t[0], pass1_bound=t[2] - t[1], int8=t[3] - t[2], bf16=t[4] - t[3])
        return tok, int(read8.sum()), int(readR.sum()), times

    def baseline(self, h, g, cfg):
        """Plain BF16 head: one cuBLAS product, soft-cap, Gumbel noise added, argmax."""
        z = (h.bfloat16() @ self.W.T).float()
        c = softcap(z, cfg["cap"])
        return (c if cfg["rule"] == "greedy" else c / cfg["T"] + g).argmax(1)

    def reference(self, h, g, cfg, cc):
        z = torch.empty(h.shape[0], self.V, device=h.device)
        for r0 in range(0, self.V, self.chunk):
            z[:, r0:r0 + self.chunk] = h @ self.W[r0:r0 + self.chunk].float().T
        return cc.decide(z, cfg, g)
