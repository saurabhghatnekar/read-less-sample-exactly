"""Fused first pass for the certified head: one read of a dithered INT4 g128 head gives approximate logits
and the sub-Gaussian bound's variance term for every row.

Layout (V rows, D hidden, G = 128, NG = D / G groups):
  Q   uint8  [V, D/2]  code q in [-8, 7] stored as q + 8; byte k of a group holds dims 2k (low nibble) and 2k+1 (high)
  S   fp16   [V, NG]   group step s
  H   bf16   [16, D]   states, zero-padded to 16 rows (B <= 16); BF16 values are exact in the dot
  UH  fp32   [16, NG]  sum_{j in g} u_j h_j  (dither correction, shared pattern u)
  HSQ fp32   [16, NG]  ||h_g||^2
Outputs Z fp32 [16, V] = sum_g s_g (q_g . h_g - uh_g) + c, and VAR fp32 [16, V] = sum_g s_g^2 ||h_g||^2.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_V": bv}, num_warps=w, num_stages=s)
             for bv in (32, 64, 128) for w in (4, 8) for s in (2, 3)],
    key=["V", "D"],
)
@triton.jit
def _int4_head_kernel(H, UH, HSQ, Q, S, Z, VAR, C, V, D: tl.constexpr, G: tl.constexpr, BLOCK_V: tl.constexpr):
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
        lo = ((qb & 0xF).to(tl.int8) - 8).to(tl.bfloat16)          # [BLOCK_V, G/2], dims 2k
        hi = ((qb >> 4).to(tl.int8) - 8).to(tl.bfloat16)           # dims 2k + 1
        he = tl.load(H + b[:, None] * D + g * G + 2 * k2[None, :])   # [16, G/2] bf16
        ho = tl.load(H + b[:, None] * D + g * G + 2 * k2[None, :] + 1)
        part = tl.dot(he, tl.trans(lo)) + tl.dot(ho, tl.trans(hi))  # [16, BLOCK_V] fp32, products exact
        s = tl.load(S + rows * NG + g, mask=rmask, other=0.0).to(tl.float32)
        uh = tl.load(UH + b * NG + g)
        hsq = tl.load(HSQ + b * NG + g)
        acc += (part - uh[:, None]) * s[None, :]
        var += hsq[:, None] * (s * s)[None, :]
    c = tl.load(C + b)
    out = b[:, None] * V + rows[None, :]
    tl.store(Z + out, acc + c[:, None], mask=rmask[None, :])
    tl.store(VAR + out, var, mask=rmask[None, :])


def int4_head(H16, UH, HSQ, Q, S, C):
    """H16 bf16 [16, D]; returns Z, VAR fp32 [16, V]."""
    V, D2 = Q.shape
    D = 2 * D2
    Z = torch.empty(16, V, device=Q.device, dtype=torch.float32)
    VAR = torch.empty_like(Z)
    grid = lambda m: (triton.cdiv(V, m["BLOCK_V"]),)
    _int4_head_kernel[grid](H16, UH, HSQ, Q, S, Z, VAR, C, V, D=D, G=128)
    return Z, VAR


def pack_int4(codes):
    """codes int [V, D] in [-8, 7] -> uint8 [V, D/2] (even dim low nibble)."""
    u = (codes + 8).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w)
             for bn in (16, 32, 64) for bk in (128, 256) for w in (4, 8)],
    key=["D", "IS_INT8"],
)
@triton.jit
def _gather_gemv_kernel(H, W, IDX, SC, OUT, N, D: tl.constexpr, IS_INT8: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """OUT[b, i] = (W[IDX[i]] . H[b]) (* SC[IDX[i]] for INT8), FP32 accumulation; reads only the listed rows."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs < N
    rows = tl.load(IDX + offs, mask=m, other=0).to(tl.int64)
    b = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(W + rows[:, None] * D + k[None, :], mask=m[:, None], other=0).to(tl.bfloat16)
        h = tl.load(H + b[:, None] * D + k[None, :])
        acc += tl.dot(h, tl.trans(w))
    if IS_INT8:
        acc *= tl.load(SC + rows, mask=m, other=0.0).to(tl.float32)[None, :]
    tl.store(OUT + b[:, None] * N + offs[None, :], acc, mask=m[None, :])


def gather_gemv(H16, W, idx, scale=None):
    """H16 bf16 [16, D]; W int8 or bf16 [V, D]; idx int64 [N]. Returns fp32 [16, N]."""
    N, D = idx.numel(), W.shape[1]
    out = torch.empty(16, N, device=W.device, dtype=torch.float32)
    if N == 0:
        return out
    grid = lambda m: (triton.cdiv(N, m["BLOCK_N"]),)
    _gather_gemv_kernel[grid](H16, W, idx, scale if scale is not None else W, out, N, D=D,
                              IS_INT8=W.dtype == torch.int8)
    return out
