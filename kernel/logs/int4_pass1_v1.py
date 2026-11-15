"""Triton kernel for pass 1 of the certified head: dithered INT4 g128 product and its Hoeffding bound.

For each row r and state b:
    z~[b, r] = sum_i w~[r, i] h[b, i],   w~ = (q - u) * s   (q packed INT4, u the shared dither, s the FP16 group scale)
    bound[b, r] = C * sqrt(sum_g ||h_g||^2 s_g^2) + gamma * ||h|| (||w|| + ||w~||)

w~ is formed in FP32 with the same operations as cert_core.make_int4_dither, so it is bit-identical to the
reference's dequantised head; the dot product accumulates in FP32 (no TF32), which the slack term covers for any
summation order. Weights are read once per row block; the batch rides along.

Packing for the kernel ("split" layout): in each 128-weight group, byte j holds weight j in the low nibble and
weight j + 64 in the high nibble, stored as q + 8. `repack` converts CertHead.packed (adjacent pairs) to it.
"""
import torch
import triton
import triton.language as tl

GROUP = 128
HALF = 64


def repack(ch, chunk=16384):
    """CertHead.packed (byte j = weights 2j, 2j+1) -> split layout, [V, d/2] uint8."""
    out = torch.empty_like(ch.packed)
    for r0 in range(0, ch.V, chunk):
        p = ch.packed[r0:r0 + chunk]
        n = p.shape[0]
        q = torch.stack((p & 15, p >> 4), -1).view(n, ch.ng, GROUP)
        out[r0:r0 + n] = (q[:, :, :HALF] | (q[:, :, HALF:] << 4)).view(n, ch.d // 2)
    return out


_CONFIGS = [triton.Config({"BLOCK_V": bv}, num_warps=nw, num_stages=ns)
            for bv in (32, 64, 128) for nw in (4, 8) for ns in (2, 3, 4)]


@triton.autotune(configs=_CONFIGS, key=["V", "D"])
@triton.jit
def _pass1_gemv(P, S4, U, H, H2SQ, WN, WQN, HN, Z, BND, V, C, GAMMA,
                D: tl.constexpr, NG: tl.constexpr, BLOCK_V: tl.constexpr):
    """Batch of one."""
    rows = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    rmask = rows < V
    j = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_V,), tl.float32)
    bacc = tl.zeros((BLOCK_V,), tl.float32)
    for g in range(NG):
        p = tl.load(P + rows[:, None] * (D // 2) + g * 64 + j[None, :], mask=rmask[:, None], other=0)
        s = tl.maximum(tl.load(S4 + rows * NG + g, mask=rmask, other=1.0).to(tl.float32), 1e-12)
        ulo = tl.load(U + g * 128 + j)
        uhi = tl.load(U + g * 128 + 64 + j)
        wlo = ((p & 15).to(tl.float32) - 8.0 - ulo[None, :]) * s[:, None]
        whi = ((p >> 4).to(tl.float32) - 8.0 - uhi[None, :]) * s[:, None]
        hlo = tl.load(H + g * 128 + j)
        hhi = tl.load(H + g * 128 + 64 + j)
        acc += tl.sum(wlo * hlo[None, :], axis=1) + tl.sum(whi * hhi[None, :], axis=1)
        bacc += tl.load(H2SQ + g) * s * s
    wn = tl.load(WN + rows, mask=rmask, other=0.0)
    wqn = tl.load(WQN + rows, mask=rmask, other=0.0)
    hn = tl.load(HN)
    tl.store(Z + rows, acc, mask=rmask)
    tl.store(BND + rows, C * tl.sqrt(bacc) + GAMMA * hn * (wn + wqn), mask=rmask)


@triton.autotune(configs=_CONFIGS, key=["V", "D", "BLOCK_B"])
@triton.jit
def _pass1_batched(P, S4, U, H, H2SQ, WN, WQN, HN, Z, BND, B, V, C, GAMMA,
                   D: tl.constexpr, NG: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_B: tl.constexpr):
    """Batch of 2 to BLOCK_B states; FP32 dot products on CUDA cores (input_precision="ieee", no TF32)."""
    rows = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    rmask = rows < V
    bi = tl.arange(0, BLOCK_B)
    bmask = bi < B
    j = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)
    bacc = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)
    for g in range(NG):
        p = tl.load(P + rows[:, None] * (D // 2) + g * 64 + j[None, :], mask=rmask[:, None], other=0)
        s = tl.maximum(tl.load(S4 + rows * NG + g, mask=rmask, other=1.0).to(tl.float32), 1e-12)
        ulo = tl.load(U + g * 128 + j)
        uhi = tl.load(U + g * 128 + 64 + j)
        wlo = ((p & 15).to(tl.float32) - 8.0 - ulo[None, :]) * s[:, None]
        whi = ((p >> 4).to(tl.float32) - 8.0 - uhi[None, :]) * s[:, None]
        hlo = tl.load(H + bi[:, None] * D + g * 128 + j[None, :], mask=bmask[:, None], other=0.0)
        hhi = tl.load(H + bi[:, None] * D + g * 128 + 64 + j[None, :], mask=bmask[:, None], other=0.0)
        acc = tl.dot(hlo, tl.trans(wlo), acc, input_precision="ieee")
        acc = tl.dot(hhi, tl.trans(whi), acc, input_precision="ieee")
        h2 = tl.load(H2SQ + bi * NG + g, mask=bmask, other=0.0)
        bacc += h2[:, None] * (s * s)[None, :]
    wn = tl.load(WN + rows, mask=rmask, other=0.0)
    wqn = tl.load(WQN + rows, mask=rmask, other=0.0)
    hn = tl.load(HN + bi, mask=bmask, other=0.0)
    off = bi[:, None] * V + rows[None, :]
    m = bmask[:, None] & rmask[None, :]
    tl.store(Z + off, acc, mask=m)
    tl.store(BND + off, C * tl.sqrt(bacc) + GAMMA * hn[:, None] * (wn + wqn)[None, :], mask=m)


class Pass1:
    """Holds the split-layout weights for one CertHead; call it with h [B, d] float32 to get (z~, bound)."""

    def __init__(self, ch):
        self.ch = ch
        self.P = repack(ch)

    def __call__(self, h):
        ch = self.ch
        B, d = h.shape
        assert d == ch.d and d % GROUP == 0 and B <= 64
        h = h.contiguous().float()
        h2sq = h.view(B, ch.ng, GROUP).pow(2).sum(-1).contiguous()
        hn = h.norm(dim=1).contiguous()
        Z = torch.empty(B, ch.V, device=h.device, dtype=torch.float32)
        BND = torch.empty_like(Z)
        grid = lambda meta: (triton.cdiv(ch.V, meta["BLOCK_V"]),)
        if B == 1:
            _pass1_gemv[grid](self.P, ch.s4, ch.u, h, h2sq, ch.w_norm, ch.wq4_norm, hn, Z, BND,
                              ch.V, ch.C, ch.gamma, D=d, NG=ch.ng)
        else:
            bb = max(16, triton.next_power_of_2(B))
            _pass1_batched[grid](self.P, ch.s4, ch.u, h, h2sq, ch.w_norm, ch.wq4_norm, hn, Z, BND, B,
                                 ch.V, ch.C, ch.gamma, D=d, NG=ch.ng, BLOCK_B=bb)
        return Z, BND
