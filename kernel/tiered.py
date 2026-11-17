"""Memory tiers for the unified head (tier_protocol.json): the rare higher-precision rows stay in host memory.

  layout B   INT4 and INT8 copies on the GPU, the BF16 head in host memory
  layout A   only the INT4 copy on the GPU; the INT8 copy and the BF16 head in host memory

The GPU part is graph-captured and ends in one small tensor that is read back once. The host finishes the step on
the CPU from its own copies, so no weight row is ever copied to the GPU:
  B  GPU: first pass, select, INT8 kernel -> the rows that still need the exact value (up to KX per state), their
     noise, and the state. Host: the BF16 rows x the state, and the token.
  A  GPU: first pass, select -> the shortlist (up to KS rows per state) with its first-pass intervals and noise, and
     the state with its per-state numbers. Host: the INT8-stage intervals for those rows with fused2's formula,
     then the BF16 rows, and the token.
A state with more rows than the fixed slots takes the slow path (rows read back in full); it is counted.
Greedy and Gumbel. The per-row numbers (norms, K2's r) are kept on both sides: 16 to 24 bytes a row.

History, kept because the protocol was hashed before any of it ran. Version 1 ran the kernels without graph capture
and read back three times a step: 1.2 ms on Qwen3-0.6B where the all-GPU step takes 0.24. Version 2 graph-captured
the GPU part but still copied the fetched rows to the GPU and finished there: 0.84 ms. This version finishes on the
host. All three returned the reference token on every smoke state.
"""
import torch
import triton

from fused2 import NEG, U32, FusedCert2, Graphed, _k1_pass1, _k2_select, _k3_d8

G = 128


def _score(x, g, cfg):
    if cfg["cap"] is not None:
        x = cfg["cap"] * torch.tanh(x / cfg["cap"])
    return x / cfg["T"] + g if cfg["rule"] == "gumbel" else x


class Tiered(FusedCert2):
    KX, KS = 256, 1024

    def __init__(self, ch, layout):
        super().__init__(ch)
        assert layout in ("A", "B")
        self.layout = layout
        self.W_host = ch.W.cpu()
        if layout == "A":
            self.q8_host, self.s8_host = ch.q8.cpu(), ch.s8.cpu()
            self.wn_host, self.wq8n_host, self.wtn_host, self.r8_host = (x.cpu() for x in (ch.w_norm, ch.wq8_norm, ch.wt_norm, ch.r8))
        self.last, self.slow_steps, self._graphs = {}, 0, {}

    def gpu_bytes(self):
        ch = self.ch
        b = self.P.numel() + 2 * ch.s4.numel() + 4 * (ch.w_norm.numel() + ch.wt_norm.numel() + ch.wq4_norm.numel())
        if ch.p is not None:
            b += 4 * ch.r4.numel()
        if self.layout == "B":
            b += ch.q8.numel() + 2 * ch.s8.numel() + 4 * ch.wq8_norm.numel() + (4 * ch.r8.numel() if ch.p is not None else 0)
        return b

    # ---- the GPU part: fixed shapes, no host sync, graph-captured
    def _front(self, h, g, cfg):
        ch = self.ch
        B, d = h.shape
        V = ch.V
        bf = self._buffers(B, h.device)
        has_cap, has_g = cfg["cap"] is not None, cfg["rule"] == "gumbel"
        cap, T = float(cfg["cap"] or 1.0), float(cfg["T"])
        si = ch.state_inputs(h)
        gg = g if has_g else bf["Z"]
        bf["PM"].fill_(NEG)
        bf["CNT"].zero_()
        _k1_pass1[lambda meta: (triton.cdiv(V, meta["BLOCK_V"]),)](
            self.P, ch.s4, h.bfloat16(), si["cu"], si["uh"], si["h2sq"], ch.w_norm, ch.wq4_norm, ch.wt_norm, ch.r4, si["hn"],
            si["csh"], si["alpha"], gg, bf["Z"], bf["BND"], bf["PM"], B, V, self.nblk, ch.C, ch.gamma, self.gamma2, U32,
            cap, T, D=d, NG=ch.ng, BLOCK_B=max(16, triton.next_power_of_2(B)), HAS_CAP=has_cap, HAS_G=has_g,
            HAS_K2=self.has_k2)
        m1 = bf["PM"].amax(1)
        _k2_select[(B, triton.cdiv(V, self.BLOCK_SEL))](
            bf["Z"], bf["BND"], gg, m1, bf["CNT"], bf["LIST"], V, cap, T, BLOCK=self.BLOCK_SEL,
            HAS_CAP=has_cap, HAS_G=has_g, num_warps=4)
        lst = bf["LIST"][:, :self.KS].clamp(0, V - 1).long()            # entries past CNT are stale; the host ignores them
        gl = g.gather(1, lst) if has_g else torch.zeros_like(lst, dtype=torch.float32)
        cnt = bf["CNT"][:, None].float()
        if self.layout == "A":
            return torch.cat([cnt, lst.float(), bf["Z"].gather(1, lst), bf["BND"].gather(1, lst), gl, h, si["h2sq"], si["cu"],
                              si["uh"], si["csh"][:, None], si["alpha"][:, None], si["hn"][:, None], m1[:, None]], 1)
        nprog = bf["nprog"]
        _k3_d8[(B, nprog)](
            ch.q8, ch.s8, si["cu"], si["uh"], si["h2sq"], ch.w_norm, ch.wq8_norm, ch.wt_norm, ch.r8, h, si["hn"], si["csh"],
            si["alpha"], bf["Z"], bf["BND"], gg, bf["LIST"], bf["CNT"], bf["L8"], bf["U8"], bf["PM2"], V, ch.C, ch.gamma,
            self.gamma2, U32, cap, T, NPROG=nprog, D=d, NG=ch.ng, BLOCK_R=self.BLOCK_R, HAS_CAP=has_cap, HAS_G=has_g,
            HAS_K2=self.has_k2, num_warps=4)
        m2 = torch.maximum(m1, bf["PM2"].amax(1))
        valid = torch.arange(self.KS, device=h.device)[None, :] < bf["CNT"][:, None]
        need = valid & (_score(bf["U8"][:, :self.KS], gl, cfg) >= m2[:, None])
        idx = torch.topk(need.float(), self.KX, dim=1).indices
        return torch.cat([cnt, need.sum(1, keepdim=True).float(), lst.gather(1, idx).float(), gl.gather(1, idx), h, m2[:, None]], 1)

    # ---- the host part, on the CPU
    def _cpu_exact(self, hb, rows, gv, cfg):
        sc = _score(self.W_host[rows].float() @ hb, gv, cfg)
        return rows[sc == sc.max()].min()

    def _cpu_int8(self, hb, rows, z4, b4, gv, st, cfg):
        """fused2's INT8-stage interval for the shortlisted rows; returns the rows that still need the exact value."""
        ch = self.ch
        h2sq, cu, uh, csh, alpha, hn, m1 = st
        q = self.q8_host[rows].float().view(-1, ch.ng, G)
        s = self.s8_host[rows].float().clamp_min(1e-12)
        z8 = (((q * hb.view(1, ch.ng, G)).sum(-1) - cu) * s).sum(1) + csh
        e = ch.C * torch.sqrt((h2sq * s * s).sum(1)) + ch.gamma * hn * self.wn_host[rows] \
            + self.gamma2 * (hn * self.wq8n_host[rows] + 2.0 * (uh * s).sum(1)) \
            + 1.01 * U32 * hn * self.wtn_host[rows] + 2.0 * U32 * (z8.abs() + csh.abs())
        if self.has_k2:
            corr = alpha * self.r8_host[rows]
            z8 = z8 - corr
            e = e + 3.0 * U32 * alpha.abs() * self.r8_host[rows].abs() + 2.0 * U32 * corr.abs()
        lo, hi = torch.maximum(z4 - b4, z8 - e), torch.minimum(z4 + b4, z8 + e)
        m2 = torch.maximum(m1, _score(lo, gv, cfg).max())
        need = _score(hi, gv, cfg) >= m2
        return rows[need], gv[need]

    def __call__(self, h, g, cfg):
        ch = self.ch
        B, d = h.shape
        cfg = dict(cfg, T=float(cfg.get("T", 1.0)))
        has_g = cfg["rule"] == "gumbel"
        key = (B, cfg["rule"], cfg["cap"], cfg["T"])
        if key not in self._graphs:
            self._front(h, g, cfg)                                      # autotune and warm up before capture
            self._graphs[key] = Graphed(lambda hh, gg: self._front(hh, gg, cfg), h, g)
        pk = self._graphs[key](h, g).cpu()                              # the one read-back of the fast path
        cnt = pk[:, 0].long().tolist()
        bf = self._buffers(B, h.device)
        toks, n_exact = [], []
        KS, KX, ng = self.KS, self.KX, ch.ng
        for b in range(B):
            if self.layout == "B":
                nx = int(pk[b, 1])
                hb = pk[b, 2 + 2 * KX:2 + 2 * KX + d]
                if cnt[b] > KS or nx > KX:                               # slow path: the whole shortlist is read back
                    self.slow_steps += 1
                    lst = bf["LIST"][b, :cnt[b]].long()
                    gv = (g[b, lst] if has_g else torch.zeros(cnt[b], device=h.device)).cpu()
                    u8, lst = bf["U8"][b, :cnt[b]].cpu(), lst.cpu()
                    keep = _score(u8, gv, cfg) >= pk[b, -1]
                    rows, gv = lst[keep], gv[keep]
                else:
                    rows, gv = pk[b, 2:2 + nx].long(), pk[b, 2 + KX:2 + KX + nx]
            else:
                o = 1 + 4 * KS
                hb = pk[b, o:o + d]
                st = (pk[b, o + d:o + d + ng], pk[b, o + d + ng:o + d + 2 * ng], pk[b, o + d + 2 * ng:o + d + 3 * ng],
                      pk[b, o + d + 3 * ng], pk[b, o + d + 3 * ng + 1], pk[b, o + d + 3 * ng + 2], pk[b, o + d + 3 * ng + 3])
                n = cnt[b]
                if n > KS:
                    self.slow_steps += 1
                    lst = bf["LIST"][b, :n].long()
                    z4, b4 = bf["Z"][b, lst].cpu(), bf["BND"][b, lst].cpu()
                    gv = (g[b, lst] if has_g else torch.zeros(n, device=h.device)).cpu()
                    lst = lst.cpu()
                else:
                    lst, z4, b4, gv = pk[b, 1:1 + n].long(), pk[b, 1 + KS:1 + KS + n], pk[b, 1 + 2 * KS:1 + 2 * KS + n], \
                        pk[b, 1 + 3 * KS:1 + 3 * KS + n]
                rows, gv = self._cpu_int8(hb, lst, z4, b4, gv, st, cfg)
            toks.append(self._cpu_exact(hb, rows, gv if has_g else 0.0, cfg))
            n_exact.append(int(rows.numel()))
        self.last = dict(int8_rows=cnt, exact_rows=n_exact)
        return torch.stack(toks)
