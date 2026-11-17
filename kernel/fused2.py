"""The unified certified head: fused.py's fixed sequence of launches with the certificate of the byte runs.

Differences from fused.py:
  first pass  the sub-Gaussian bound; copies of W_t = W - t 1 mu^T with the exact add-back c = t (h . mu); K2's
              correction z -= alpha r with the bound taken on h - alpha p
  INT8 stage  a dithered INT8 g128 copy with the same bound (T1's d8 stage) in place of the worst-case INT8 rows
The select and exact kernels, the tensor-core product and its slack are fused.py's.

Slack, on top of the sub-Gaussian term C sqrt(sum_g ||h'_g||^2 s_g^2):
  gamma ||h|| ||w||                                   the FP32 reference
  gamma2 (||h|| ||w~|| + 2 sum_g s_g ||u_g|| ||h_g||)   the copy's product (tensor cores; see int4_pass1.py)
  1.01 u ||h|| ||w_t|| + 2 u (|z| + |c|)               the rounding of W_t and of the add-back (a3_lib.intervals)
  3 u |alpha| |r| + 2 u |alpha r|                      K2 (hdir_lib.corrected)
"""
import time

import torch
import triton
import triton.language as tl

from fused import NEG, Graphed, _k2_select, _k4_exact, _score  # noqa: F401
from int4_pass1 import GROUP, repack

U32 = 2.0 ** -24


@triton.autotune(configs=[triton.Config({"BLOCK_V": bv}, num_warps=nw, num_stages=ns)
                          for bv in (32, 64, 128, 256) for nw in (4, 8) for ns in (3, 4)],
                 key=["V", "D", "BLOCK_B", "HAS_CAP", "HAS_G", "HAS_K2"])
@triton.jit
def _k1_pass1(P, S4, H16, CU, UH, H2SQ, WN, WQN, WTN, RV, HN, CSH, ALPHA, G, Z, BND, PM, B, V, NBLK, C, GAMMA, GAMMA2,
              UX, CAP, T, D: tl.constexpr, NG: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_B: tl.constexpr,
              HAS_CAP: tl.constexpr, HAS_G: tl.constexpr, HAS_K2: tl.constexpr):
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
    wtn = tl.load(WTN + rows, mask=rmask, other=0.0)
    hn = tl.load(HN + bi, mask=bmask, other=0.0)[:, None]
    csh = tl.load(CSH + bi, mask=bmask, other=0.0)[:, None]
    off = bi[:, None] * V + rows[None, :]
    m = bmask[:, None] & rmask[None, :]
    z = acc + csh
    b = C * tl.sqrt(bacc) + GAMMA * hn * wn[None, :] + GAMMA2 * (hn * wqn[None, :] + 2.0 * uacc) \
        + 1.01 * UX * hn * wtn[None, :] + 2.0 * UX * (tl.abs(z) + tl.abs(csh))
    if HAS_K2:
        al = tl.load(ALPHA + bi, mask=bmask, other=0.0)[:, None]
        rv = tl.load(RV + rows, mask=rmask, other=0.0)[None, :]
        corr = al * rv
        z = z - corr
        b = b + 3.0 * UX * tl.abs(al) * tl.abs(rv) + 2.0 * UX * tl.abs(corr)
    tl.store(Z + off, z, mask=m)
    tl.store(BND + off, b, mask=m)
    if HAS_G:
        gv = tl.load(G + off, mask=m, other=0.0)
    else:
        gv = 0.0
    sl = tl.where(m, _score(z - b, gv, CAP, T, HAS_CAP, HAS_G), float("-inf"))
    tl.store(PM + bi * NBLK + pid, tl.max(sl, axis=1), mask=bmask)


@triton.jit
def _k3_d8(Q8, S8, CU, UH, H2SQ, WN, WQN, WTN, R8, H, HN, CSH, ALPHA, Z, BND, G, LIST, CNT, L8, U8, PM2, V, C, GAMMA, GAMMA2,
           UX, CAP, T, NPROG: tl.constexpr, D: tl.constexpr, NG: tl.constexpr, BLOCK_R: tl.constexpr,
           HAS_CAP: tl.constexpr, HAS_G: tl.constexpr, HAS_K2: tl.constexpr):
    b = tl.program_id(0)
    p = tl.program_id(1)
    n = tl.load(CNT + b)
    hn = tl.load(HN + b)
    csh = tl.load(CSH + b)
    al = tl.load(ALPHA + b)
    k = tl.arange(0, 128)
    run = tl.full((BLOCK_R,), float("-inf"), tl.float32)
    for start in range(p * BLOCK_R, n, NPROG * BLOCK_R):
        idx = start + tl.arange(0, BLOCK_R)
        im = idx < n
        rows = tl.load(LIST + b * V + idx, mask=im, other=0).to(tl.int64)
        acc = tl.zeros((BLOCK_R,), tl.float32)
        bacc = tl.zeros((BLOCK_R,), tl.float32)
        uacc = tl.zeros((BLOCK_R,), tl.float32)
        for g in range(NG):
            q = tl.load(Q8 + rows[:, None] * D + g * 128 + k[None, :], mask=im[:, None], other=0).to(tl.float32)
            hv = tl.load(H + b * D + g * 128 + k)
            s = tl.maximum(tl.load(S8 + rows * NG + g, mask=im, other=1.0).to(tl.float32), 1e-12)
            acc += (tl.sum(q * hv[None, :], axis=1) - tl.load(CU + b * NG + g)) * s
            bacc += tl.load(H2SQ + b * NG + g) * s * s
            uacc += tl.load(UH + b * NG + g) * s
        wn = tl.load(WN + rows, mask=im, other=0.0)
        wqn = tl.load(WQN + rows, mask=im, other=0.0)
        wtn = tl.load(WTN + rows, mask=im, other=0.0)
        z8 = acc + csh
        e = C * tl.sqrt(bacc) + GAMMA * hn * wn + GAMMA2 * (hn * wqn + 2.0 * uacc) \
            + 1.01 * UX * hn * wtn + 2.0 * UX * (tl.abs(z8) + tl.abs(csh))
        if HAS_K2:
            r8 = tl.load(R8 + rows, mask=im, other=0.0)
            corr = al * r8
            z8 = z8 - corr
            e = e + 3.0 * UX * tl.abs(al) * tl.abs(r8) + 2.0 * UX * tl.abs(corr)
        z4 = tl.load(Z + b * V + rows, mask=im, other=0.0)
        b4 = tl.load(BND + b * V + rows, mask=im, other=0.0)
        lo = tl.maximum(z4 - b4, z8 - e)
        hi = tl.minimum(z4 + b4, z8 + e)
        tl.store(L8 + b * V + idx, lo, mask=im)
        tl.store(U8 + b * V + idx, hi, mask=im)
        if HAS_G:
            gv = tl.load(G + b * V + rows, mask=im, other=0.0)
        else:
            gv = 0.0
        run = tl.maximum(run, tl.where(im, _score(lo, gv, CAP, T, HAS_CAP, HAS_G), float("-inf")))
    tl.store(PM2 + b * NPROG + p, tl.max(run, axis=0))


class FusedCert2:
    """All passes on the GPU for one CertHead2. call(h, g, cfg) -> (token [B], INT8 rows [B], exact rows [B])."""

    BLOCK_SEL, BLOCK_R, BLOCK_K = 1024, 16, 256

    def __init__(self, ch):
        assert ch.d % self.BLOCK_K == 0
        self.ch = ch
        self.P = repack(ch)
        n = 2 * ch.d
        self.gamma2 = n * 2 * 2.0 ** -24 / (1 - n * 2 * 2.0 ** -24)
        self.nblk = triton.cdiv(ch.V, 32)
        self.has_k2 = ch.p is not None
        self._buf = {}

    def _buffers(self, B, dev):
        if B not in self._buf:
            V = self.ch.V
            f = dict(device=dev, dtype=torch.float32)
            i = dict(device=dev, dtype=torch.int32)
            nprog = max(1, 256 // B)
            self._buf[B] = dict(nprog=nprog, Z=torch.empty(B, V, **f), BND=torch.empty(B, V, **f),
                                PM=torch.empty(B, self.nblk, **f), CNT=torch.empty(B, **i), LIST=torch.empty(B, V, **i),
                                L8=torch.empty(B, V, **f), U8=torch.empty(B, V, **f), PM2=torch.empty(B, nprog, **f),
                                OUTV=torch.empty(B, nprog, **f), OUTI=torch.empty(B, nprog, **i), NR=torch.empty(B, nprog, **i))
        return self._buf[B]

    def __call__(self, h, g, cfg, prof=None):
        ch = self.ch
        B, d = h.shape
        V = ch.V
        bf = self._buffers(B, h.device)
        nprog = bf["nprog"]
        has_cap = cfg["cap"] is not None
        has_g = cfg["rule"] == "gumbel"
        cap = float(cfg["cap"] or 1.0)
        T = float(cfg.get("T", 1.0))

        def mark(name):
            if prof is not None:
                torch.cuda.synchronize()
                now = time.perf_counter()
                prof[name] = prof.get(name, 0.0) + (now - mark.t) * 1e3
                mark.t = now

        if prof is not None:
            torch.cuda.synchronize()
            mark.t = time.perf_counter()
        si = ch.state_inputs(h)
        h16 = h.bfloat16()
        gg = g if has_g else bf["Z"]
        bf["PM"].fill_(NEG)
        bf["CNT"].zero_()
        bb = max(16, triton.next_power_of_2(B))
        mark("prep")
        _k1_pass1[lambda meta: (triton.cdiv(V, meta["BLOCK_V"]),)](
            self.P, ch.s4, h16, si["cu"], si["uh"], si["h2sq"], ch.w_norm, ch.wq4_norm, ch.wt_norm, ch.r4, si["hn"],
            si["csh"], si["alpha"], gg, bf["Z"], bf["BND"], bf["PM"], B, V, self.nblk, ch.C, ch.gamma, self.gamma2, U32,
            cap, T, D=d, NG=ch.ng, BLOCK_B=bb, HAS_CAP=has_cap, HAS_G=has_g, HAS_K2=self.has_k2)
        m1 = bf["PM"].amax(1)
        mark("pass1")
        _k2_select[(B, triton.cdiv(V, self.BLOCK_SEL))](
            bf["Z"], bf["BND"], gg, m1, bf["CNT"], bf["LIST"], V, cap, T, BLOCK=self.BLOCK_SEL,
            HAS_CAP=has_cap, HAS_G=has_g, num_warps=4)
        mark("select")
        _k3_d8[(B, nprog)](
            ch.q8, ch.s8, si["cu"], si["uh"], si["h2sq"], ch.w_norm, ch.wq8_norm, ch.wt_norm, ch.r8, h, si["hn"], si["csh"],
            si["alpha"], bf["Z"], bf["BND"], gg, bf["LIST"], bf["CNT"], bf["L8"], bf["U8"], bf["PM2"], V, ch.C, ch.gamma,
            self.gamma2, U32, cap, T, NPROG=nprog, D=d, NG=ch.ng, BLOCK_R=self.BLOCK_R, HAS_CAP=has_cap, HAS_G=has_g,
            HAS_K2=self.has_k2, num_warps=4)
        m2 = torch.maximum(m1, bf["PM2"].amax(1))
        mark("int8")
        _k4_exact[(B, nprog)](
            ch.W, h, gg, bf["LIST"], bf["CNT"], bf["U8"], m2, bf["OUTV"], bf["OUTI"], bf["NR"], V, cap, T,
            NPROG=nprog, D=d, BLOCK_R=self.BLOCK_R, BLOCK_K=self.BLOCK_K, HAS_CAP=has_cap, HAS_G=has_g, num_warps=4)
        v = bf["OUTV"]
        best = v.amax(1, keepdim=True)
        tok = torch.where(v == best, bf["OUTI"], torch.full_like(bf["OUTI"], 2147483647)).amin(1).long()
        mark("exact")
        return tok, bf["CNT"], bf["NR"].sum(1)
