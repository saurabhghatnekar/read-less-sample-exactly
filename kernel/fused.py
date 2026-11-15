"""Certified head with all three passes on the GPU: a fixed sequence of launches, no host syncs, CUDA-graph ready.

Rules: greedy and Gumbel (race rules). Score s_r = phi(z_r) for greedy, phi(z_r) / T + g_r for Gumbel, where phi is
the soft-cap (or the identity) and g the Gumbel noise. A row is needed while its upper score can reach the largest
lower score of any row. That maximum only grows as intervals tighten, so the needed set only shrinks: one INT8
round over the rows needed after pass 1, then one exact round over the INT8 rows still needed, is already the
stable state that cert_core's refine-until-stable loop reaches.

  K1  pass1      every row: INT4 product (tensor cores, see int4_pass1.py), bound, and per row block the largest
                 lower score -> M1 per state
  K2  select     every row: upper score >= M1 -> appended to the state's own list (atomic counter)
  K3  int8       listed rows only: INT8 product and bound, intervals intersected, largest lower score -> M2
  K4  exact      listed rows whose upper score >= M2: BF16 row x h in FP32, score, per-program best (score, row)
  then a few small torch reductions pick the token (ties go to the lowest row index, as torch.argmax does).

Each state has its own row lists, so one state's long list does not make the others read more.
The Gumbel noise g [B, V] is an input, generated outside (as in the PyTorch baseline).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from int4_pass1 import GROUP, repack

NEG = float("-inf")


@triton.jit
def _score(x, g, CAP, T, HAS_CAP: tl.constexpr, HAS_G: tl.constexpr):
    if HAS_CAP:
        x = CAP * libdevice.tanh(x / CAP)
    if HAS_G:
        x = x / T + g
    return x


@triton.autotune(configs=[triton.Config({"BLOCK_V": bv}, num_warps=nw, num_stages=ns)
                          for bv in (32, 64, 128, 256) for nw in (4, 8) for ns in (3, 4)],
                 key=["V", "D", "BLOCK_B", "HAS_CAP", "HAS_G"])
@triton.jit
def _k1_pass1(P, S4, H16, CU, UH, H2SQ, WN, WQN, HN, G, Z, BND, PM, B, V, NBLK, C, GAMMA, GAMMA2, CAP, T,
              D: tl.constexpr, NG: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_B: tl.constexpr,
              HAS_CAP: tl.constexpr, HAS_G: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    rmask = rows < V
    bi = tl.arange(0, BLOCK_B)
    bmask = bi < B
    j = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)
    bacc = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)
    uacc = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)
    for g in range(NG):
        p = tl.load(P + rows[:, None] * (D // 2) + g * 64 + j[None, :], mask=rmask[:, None], other=136)
        qlo = ((p & 15).to(tl.float32) - 8.0).to(tl.bfloat16)
        qhi = ((p >> 4).to(tl.float32) - 8.0).to(tl.bfloat16)
        hlo = tl.load(H16 + bi[:, None] * D + g * 128 + j[None, :], mask=bmask[:, None], other=0.0)
        hhi = tl.load(H16 + bi[:, None] * D + g * 128 + 64 + j[None, :], mask=bmask[:, None], other=0.0)
        dq = tl.dot(hlo, tl.trans(qlo))
        dq = tl.dot(hhi, tl.trans(qhi), dq)
        s = tl.maximum(tl.load(S4 + rows * NG + g, mask=rmask, other=1.0).to(tl.float32), 1e-12)
        cu = tl.load(CU + bi * NG + g, mask=bmask, other=0.0)
        acc += (dq - cu[:, None]) * s[None, :]
        h2 = tl.load(H2SQ + bi * NG + g, mask=bmask, other=0.0)
        bacc += h2[:, None] * (s * s)[None, :]
        uh = tl.load(UH + bi * NG + g, mask=bmask, other=0.0)
        uacc += uh[:, None] * s[None, :]
    wn = tl.load(WN + rows, mask=rmask, other=0.0)
    wqn = tl.load(WQN + rows, mask=rmask, other=0.0)
    hn = tl.load(HN + bi, mask=bmask, other=0.0)[:, None]
    off = bi[:, None] * V + rows[None, :]
    m = bmask[:, None] & rmask[None, :]
    b = C * tl.sqrt(bacc) + GAMMA * hn * wn[None, :] + GAMMA2 * (hn * wqn[None, :] + 2.0 * uacc)
    tl.store(Z + off, acc, mask=m)
    tl.store(BND + off, b, mask=m)
    if HAS_G:
        gv = tl.load(G + off, mask=m, other=0.0)
    else:
        gv = 0.0
    sl = tl.where(m, _score(acc - b, gv, CAP, T, HAS_CAP, HAS_G), float("-inf"))
    tl.store(PM + bi * NBLK + pid, tl.max(sl, axis=1), mask=bmask)


@triton.jit
def _k2_select(Z, BND, G, M1, CNT, LIST, V, CAP, T, BLOCK: tl.constexpr,
               HAS_CAP: tl.constexpr, HAS_G: tl.constexpr):
    b = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    rm = rows < V
    z = tl.load(Z + b * V + rows, mask=rm, other=0.0)
    bd = tl.load(BND + b * V + rows, mask=rm, other=0.0)
    if HAS_G:
        gv = tl.load(G + b * V + rows, mask=rm, other=0.0)
    else:
        gv = 0.0
    su = _score(z + bd, gv, CAP, T, HAS_CAP, HAS_G)
    flag = (su >= tl.load(M1 + b)) & rm
    fi = flag.to(tl.int32)
    n = tl.sum(fi, axis=0)
    if n > 0:
        base = tl.atomic_add(CNT + b, n)
        pos = base + tl.cumsum(fi, axis=0) - 1
        tl.store(LIST + b * V + pos, rows, mask=flag)


@triton.jit
def _k3_int8(Q8, S8, EMAX, ENORM, WN, WQN, H, H1, HN, Z, BND, G, LIST, CNT, L8, U8, PM2, V, GAMMA, CAP, T,
             NPROG: tl.constexpr, D: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr,
             HAS_CAP: tl.constexpr, HAS_G: tl.constexpr):
    b = tl.program_id(0)
    p = tl.program_id(1)
    n = tl.load(CNT + b)
    h1 = tl.load(H1 + b)
    hn = tl.load(HN + b)
    run = tl.full((BLOCK_R,), float("-inf"), tl.float32)
    for start in range(p * BLOCK_R, n, NPROG * BLOCK_R):
        idx = start + tl.arange(0, BLOCK_R)
        im = idx < n
        rows = tl.load(LIST + b * V + idx, mask=im, other=0)
        s8 = tl.maximum(tl.load(S8 + rows, mask=im, other=1.0).to(tl.float32), 1e-12)
        acc = tl.zeros((BLOCK_R,), tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            q = tl.load(Q8 + rows[:, None].to(tl.int64) * D + k[None, :], mask=im[:, None], other=0).to(tl.float32)
            hv = tl.load(H + b * D + k)
            acc += tl.sum((q * s8[:, None]) * hv[None, :], axis=1)
        e = tl.minimum(h1 * tl.load(EMAX + rows, mask=im, other=0.0), hn * tl.load(ENORM + rows, mask=im, other=0.0)) \
            + GAMMA * hn * (tl.load(WN + rows, mask=im, other=0.0) + tl.load(WQN + rows, mask=im, other=0.0))
        z4 = tl.load(Z + b * V + rows, mask=im, other=0.0)
        b4 = tl.load(BND + b * V + rows, mask=im, other=0.0)
        lo = tl.maximum(z4 - b4, acc - e)
        hi = tl.minimum(z4 + b4, acc + e)
        tl.store(L8 + b * V + idx, lo, mask=im)
        tl.store(U8 + b * V + idx, hi, mask=im)
        if HAS_G:
            gv = tl.load(G + b * V + rows, mask=im, other=0.0)
        else:
            gv = 0.0
        run = tl.maximum(run, tl.where(im, _score(lo, gv, CAP, T, HAS_CAP, HAS_G), float("-inf")))
    tl.store(PM2 + b * NPROG + p, tl.max(run, axis=0))


@triton.jit
def _k4_exact(W16, H, G, LIST, CNT, U8, M2, OUTV, OUTI, NR, V, CAP, T,
              NPROG: tl.constexpr, D: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr,
              HAS_CAP: tl.constexpr, HAS_G: tl.constexpr):
    b = tl.program_id(0)
    p = tl.program_id(1)
    n = tl.load(CNT + b)
    m2 = tl.load(M2 + b)
    bestv = float("-inf")
    besti = 2147483647
    nr = 0
    for start in range(p * BLOCK_R, n, NPROG * BLOCK_R):
        idx = start + tl.arange(0, BLOCK_R)
        im = idx < n
        rows = tl.load(LIST + b * V + idx, mask=im, other=0)
        if HAS_G:
            gv = tl.load(G + b * V + rows, mask=im, other=0.0)
        else:
            gv = 0.0
        need = im & (_score(tl.load(U8 + b * V + idx, mask=im, other=0.0), gv, CAP, T, HAS_CAP, HAS_G) >= m2)
        nr += tl.sum(need.to(tl.int32), axis=0)
        acc = tl.zeros((BLOCK_R,), tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            w = tl.load(W16 + rows[:, None].to(tl.int64) * D + k[None, :], mask=need[:, None], other=0.0)
            hv = tl.load(H + b * D + k)
            acc += tl.sum(w.to(tl.float32) * hv[None, :], axis=1)
        sc = tl.where(need, _score(acc, gv, CAP, T, HAS_CAP, HAS_G), float("-inf"))
        v = tl.max(sc, axis=0)
        i = tl.min(tl.where(need & (sc == v), rows, 2147483647), axis=0)
        take = (v > bestv) | ((v == bestv) & (i < besti))
        bestv = tl.where(take, v, bestv)
        besti = tl.where(take, i, besti)
    tl.store(OUTV + b * NPROG + p, bestv)
    tl.store(OUTI + b * NPROG + p, besti)
    tl.store(NR + b * NPROG + p, nr)


class FusedCert:
    """All passes on the GPU for one CertHead. call(h, g, cfg) -> (token [B] int64, INT8 rows [B], exact rows [B]).
    h: [B, d] float32 holding BF16 values (B <= 64). g: [B, V] float32 Gumbel noise (ignored for greedy)."""

    BLOCK_SEL, BLOCK_R, BLOCK_K = 1024, 16, 256

    def __init__(self, ch):
        assert ch.d % self.BLOCK_K == 0
        self.ch = ch
        self.P = repack(ch)
        n = 2 * ch.d
        self.gamma2 = n * 2 * 2.0 ** -24 / (1 - n * 2 * 2.0 ** -24)
        self.ug = ch.u.view(ch.ng, GROUP)
        self.un = self.ug.norm(dim=1)
        self.nblk = triton.cdiv(ch.V, 32)          # enough partial-max slots for the smallest BLOCK_V
        self._buf = {}

    def _buffers(self, B, dev):
        if B not in self._buf:
            V = self.ch.V
            f = dict(device=dev, dtype=torch.float32)
            nprog = max(1, 256 // B)
            self._buf[B] = dict(
                nprog=nprog, Z=torch.empty(B, V, **f), BND=torch.empty(B, V, **f), PM=torch.empty(B, self.nblk, **f),
                CNT=torch.empty(B, device=dev, dtype=torch.int32), LIST=torch.empty(B, V, device=dev, dtype=torch.int32),
                L8=torch.empty(B, V, **f), U8=torch.empty(B, V, **f), PM2=torch.empty(B, nprog, **f),
                OUTV=torch.empty(B, nprog, **f), OUTI=torch.empty(B, nprog, device=dev, dtype=torch.int32),
                NR=torch.empty(B, nprog, device=dev, dtype=torch.int32))
        return self._buf[B]

    def __call__(self, h, g, cfg):
        ch = self.ch
        B, d = h.shape
        V = ch.V
        bf = self._buffers(B, h.device)
        nprog = bf["nprog"]
        has_cap = cfg["cap"] is not None
        has_g = cfg["rule"] == "gumbel"
        cap = float(cfg["cap"] or 1.0)
        T = float(cfg.get("T", 1.0))
        hg = h.view(B, ch.ng, GROUP)
        h2sq = hg.pow(2).sum(-1)
        hn = h.norm(dim=1)
        h1 = h.abs().sum(1)
        cu = (hg * self.ug).sum(-1)
        uh = hg.norm(dim=-1) * self.un
        h16 = h.bfloat16()
        gg = g if has_g else bf["Z"]               # any float buffer; not read when HAS_G is false
        bf["PM"].fill_(NEG)
        bf["CNT"].zero_()
        bb = max(16, triton.next_power_of_2(B))
        _k1_pass1[lambda meta: (triton.cdiv(V, meta["BLOCK_V"]),)](
            self.P, ch.s4, h16, cu, uh, h2sq, ch.w_norm, ch.wq4_norm, hn, gg, bf["Z"], bf["BND"], bf["PM"],
            B, V, self.nblk, ch.C, ch.gamma, self.gamma2, cap, T, D=d, NG=ch.ng, BLOCK_B=bb,
            HAS_CAP=has_cap, HAS_G=has_g)
        m1 = bf["PM"].amax(1)
        _k2_select[(B, triton.cdiv(V, self.BLOCK_SEL))](
            bf["Z"], bf["BND"], gg, m1, bf["CNT"], bf["LIST"], V, cap, T, BLOCK=self.BLOCK_SEL,
            HAS_CAP=has_cap, HAS_G=has_g, num_warps=4)
        _k3_int8[(B, nprog)](
            ch.q8, ch.s8, ch.emax8, ch.enorm8, ch.w_norm, ch.wq8_norm, h, h1, hn, bf["Z"], bf["BND"], gg,
            bf["LIST"], bf["CNT"], bf["L8"], bf["U8"], bf["PM2"], V, ch.gamma, cap, T,
            NPROG=nprog, D=d, BLOCK_R=self.BLOCK_R, BLOCK_K=self.BLOCK_K, HAS_CAP=has_cap, HAS_G=has_g, num_warps=4)
        m2 = torch.maximum(m1, bf["PM2"].amax(1))
        _k4_exact[(B, nprog)](
            ch.W, h, gg, bf["LIST"], bf["CNT"], bf["U8"], m2, bf["OUTV"], bf["OUTI"], bf["NR"], V, cap, T,
            NPROG=nprog, D=d, BLOCK_R=self.BLOCK_R, BLOCK_K=self.BLOCK_K, HAS_CAP=has_cap, HAS_G=has_g, num_warps=4)
        v = bf["OUTV"]
        best = v.amax(1, keepdim=True)
        tok = torch.where(v == best, bf["OUTI"], torch.full_like(bf["OUTI"], 2147483647)).amin(1).long()
        return tok, bf["CNT"], bf["NR"].sum(1)


class Graphed:
    """CUDA-graph replay of a callable f(h, g) with static inputs; call warm-up first so autotuning is done."""

    def __init__(self, f, h, g, warmup=3):
        self.h, self.g = h.clone(), g.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                f(self.h, self.g)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = f(self.h, self.g)

    def __call__(self, h, g):
        self.h.copy_(h)
        self.g.copy_(g)
        self.graph.replay()
        return self.out
