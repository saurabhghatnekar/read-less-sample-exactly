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


def _cfgs(bbs, bvs, gpis, limit):
    return [triton.Config({"BB": bb, "BLOCK_V": bv, "GPI": gpi}, num_warps=nw, num_stages=3)
            for bb in bbs for bv in bvs for gpi in gpis for nw in (4, 8) if bb * bv * gpi * 64 <= limit]


@triton.jit
def _pass1(P, S4, U, H, H2SQ, WN, WQN, HN, Z, BND, B, V, C, GAMMA,
           D: tl.constexpr, NG: tl.constexpr, BB: tl.constexpr, BLOCK_V: tl.constexpr, GPI: tl.constexpr):
    """Program (bt, rb): states bt*BB .. +BB, rows rb*BLOCK_V .. +BLOCK_V, GPI groups (GPI*64 bytes per row) per step.
    Batch tiles are the fastest grid axis, so tiles sharing a row block run together and share it through L2.
    Products and sums are FP32 on CUDA cores; w~ is formed exactly as in cert_core."""
    bt = tl.program_id(0)
    rb = tl.program_id(1)
    rows = rb * BLOCK_V + tl.arange(0, BLOCK_V)
    rmask = rows < V
    bi = bt * BB + tl.arange(0, BB)
    bmask = bi < B
    k = tl.arange(0, GPI * 64)
    kg = k // 64
    kj = k % 64
    gi = tl.arange(0, GPI)
    acc = tl.zeros((BB, BLOCK_V), tl.float32)
    bacc = tl.zeros((BB, BLOCK_V), tl.float32)
    for g0 in range(0, NG, GPI):
        p = tl.load(P + rows[:, None] * (D // 2) + g0 * 64 + k[None, :], mask=rmask[:, None], other=0)
        s = tl.load(S4 + rows[:, None] * NG + g0 + kg[None, :], mask=rmask[:, None], other=1.0)
        s = tl.maximum(s.to(tl.float32), 1e-12)
        elo = (g0 + kg) * 128 + kj
        ulo = tl.load(U + elo)
        uhi = tl.load(U + elo + 64)
        wlo = ((p & 15).to(tl.float32) - 8.0 - ulo[None, :]) * s
        whi = ((p >> 4).to(tl.float32) - 8.0 - uhi[None, :]) * s
        hlo = tl.load(H + bi[:, None] * D + elo[None, :], mask=bmask[:, None], other=0.0)
        hhi = tl.load(H + bi[:, None] * D + elo[None, :] + 64, mask=bmask[:, None], other=0.0)
        acc += tl.sum(wlo[None, :, :] * hlo[:, None, :] + whi[None, :, :] * hhi[:, None, :], axis=2)
        sg = tl.load(S4 + rows[:, None] * NG + g0 + gi[None, :], mask=rmask[:, None], other=1.0)
        sg = tl.maximum(sg.to(tl.float32), 1e-12)
        h2 = tl.load(H2SQ + bi[:, None] * NG + g0 + gi[None, :], mask=bmask[:, None], other=0.0)
        bacc += tl.sum(h2[:, None, :] * (sg * sg)[None, :, :], axis=2)
    wn = tl.load(WN + rows, mask=rmask, other=0.0)
    wqn = tl.load(WQN + rows, mask=rmask, other=0.0)
    hn = tl.load(HN + bi, mask=bmask, other=0.0)
    off = bi[:, None] * V + rows[None, :]
    m = bmask[:, None] & rmask[None, :]
    tl.store(Z + off, acc, mask=m)
    tl.store(BND + off, C * tl.sqrt(bacc) + GAMMA * hn[:, None] * (wn + wqn)[None, :], mask=m)


_pass1_b1 = triton.autotune(configs=_cfgs((1,), (16, 32, 64, 128), (1, 2, 4), 16384), key=["V", "D"])(_pass1)
_pass1_bn = triton.autotune(configs=_cfgs((2, 4, 8), (16, 32, 64), (1, 2), 32768), key=["B", "V", "D"])(_pass1)


class Pass1:
    """Holds the split-layout weights for one CertHead; call it with h [B, d] float32 to get (z~, bound)."""

    def __init__(self, ch):
        self.ch = ch
        self.P = repack(ch)
        assert all(ch.ng % gpi == 0 for gpi in (1, 2, 4))

    def __call__(self, h):
        ch = self.ch
        B, d = h.shape
        assert d == ch.d and d % GROUP == 0
        h = h.contiguous().float()
        h2sq = h.view(B, ch.ng, GROUP).pow(2).sum(-1).contiguous()
        hn = h.norm(dim=1).contiguous()
        Z = torch.empty(B, ch.V, device=h.device, dtype=torch.float32)
        BND = torch.empty_like(Z)
        grid = lambda meta: (triton.cdiv(B, meta["BB"]), triton.cdiv(ch.V, meta["BLOCK_V"]))
        kern = _pass1_b1 if B == 1 else _pass1_bn
        kern[grid](self.P, ch.s4, ch.u, h, h2sq, ch.w_norm, ch.wq4_norm, hn, Z, BND, B,
                   ch.V, ch.C, ch.gamma, D=d, NG=ch.ng)
        return Z, BND
