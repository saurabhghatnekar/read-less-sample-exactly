"""CPU test for cert_head2.py: the stored copies equal the byte runs' quantizers bit for bit, and the per-state inputs
give the byte runs' intervals when put together in plain PyTorch.

  python3 -I test_cert_head2.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp01_repro"))
import a3_lib, cert_core as cc, hdir_lib, mix_lib, pipe_lib  # noqa: E402
from cert_head2 import CertHead2, U32  # noqa: E402
from test_a3 import fake_head, fake_states  # noqa: E402

torch.manual_seed(0)
V, d, B = 4096, 512, 32
W = fake_head(V, d)
mu = a3_lib.mean_row(W)
Pall, _ = hdir_lib.fit_directions(mu, fake_states(256, d, mu), kmax=4)
p = Pall[:, 1]
h = fake_states(B, d, mu)
w_norm = W.norm(dim=1)
for t, pp in ((0.0, None), (0.5, None), (1.5, p)):
    ch = CertHead2(W.bfloat16(), t=t, p=pp, chunk=1000)
    assert torch.equal(ch.mu64, mu)
    vs, Wt = pipe_lib.make_variants(W, mu, t, dith8=True)
    wq4 = torch.cat([ch.dequant4(r0, min(r0 + 1000, V)) for r0 in range(0, V, 1000)])
    wq8 = torch.cat([ch.dequant8(r0, min(r0 + 1000, V)) for r0 in range(0, V, 1000)])
    assert torch.equal(wq4, vs["i4"].Wq), "INT4 copy differs from cert_core.make_int4_dither"
    assert torch.equal(wq8, vs["d8"].Wq), "INT8 copy differs from mix_lib.make_mixed_dither"
    assert torch.equal(ch.s4.float().clamp_min(1e-12), vs["i4"].scale) and torch.equal(ch.s8.float().clamp_min(1e-12), vs["d8"].scale)
    assert torch.allclose(ch.wt_norm, vs["i4"].wt_norm) and torch.allclose(ch.wq4_norm, vs["i4"].wq_norm)
    si = ch.state_inputs(h)
    for name, wq, sc, r, wqn in (("i4", wq4, ch.s4, ch.r4, ch.wq4_norm), ("d8", wq8, ch.s8, ch.r8, ch.wq8_norm)):
        sets = None
        if pp is not None:
            rr = hdir_lib.err_proj(vs[name].Wq, Wt, pp[:, None])
            assert torch.allclose(r, rr[:, 0], atol=1e-7)
            sets = (pp[:, None], rr)
        z_ref, b_ref = pipe_lib.intervals(vs[name], "subg", sets, h, w_norm, mu, None)
        # the kernel's formula, in plain PyTorch, from the stored copies and the per-state inputs
        s = sc.float().clamp_min(1e-12)
        z = h @ wq.T + si["csh"][:, None]
        hn = si["hn"][:, None]
        b = ch.C * torch.sqrt(si["h2sq"] @ (s ** 2).T) + ch.gamma * hn * (ch.w_norm + wqn)[None, :] \
            + 1.01 * U32 * hn * ch.wt_norm[None, :] + 2 * U32 * (z.abs() + si["csh"].abs()[:, None])
        if pp is not None:
            corr = si["alpha"][:, None] * r[None, :]
            z = z - corr
            b = b + 3 * U32 * si["alpha"].abs()[:, None] * r.abs()[None, :] + 2 * U32 * corr.abs()
        assert float((z - z_ref).abs().max()) < 2e-4 * float(b_ref.median()), (name, float((z - z_ref).abs().max()))
        ratio = b / b_ref
        assert float(ratio.min()) > 1 - 1e-5 and float(ratio.max()) < 1.01, (name, float(ratio.min()), float(ratio.max()))
        assert ((h @ W.T - z).abs() <= b).all()
    print(f"t={t:g}, K2={'yes' if pp is not None else 'no'}: copies equal the byte runs' bit for bit; intervals equal "
          f"pipe_lib's (logits to 2e-4 of a bound, bounds within [1 - 1e-5, 1.01]); every reference logit inside")
print("ALL PASS")
