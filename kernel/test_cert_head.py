"""CPU test: quantised weights equal cert_core's, and certified tokens equal the FP32 reference.

  python3 -I test_cert_head.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp01_repro"))
import cert_core as cc  # noqa: E402
from cert_head import CertHead  # noqa: E402
from test_local import fake_head, fake_states  # noqa: E402

torch.manual_seed(0)
V, d, B = 4096, 512, 64
W = fake_head(V, d)
h = fake_states(B, d)
ch = CertHead(W.bfloat16(), chunk=1000)                  # chunk not a multiple of V on purpose

ref4 = cc.make_int4_dither(W)
ref8 = cc.make_int8_row(W)
wq4 = torch.cat([ch.dequant4(r0, min(r0 + 1000, V)) for r0 in range(0, V, 1000)])
assert torch.equal(wq4, ref4.Wq), "INT4 dequant differs from cert_core"
assert torch.equal(ch.q8.float() * ch.s8.float().clamp_min(1e-12)[:, None], ref8.Wq), "INT8 differs"
assert torch.allclose(ch.emax8, ref8.emax.squeeze(1)) and torch.allclose(ch.enorm8, ref8.enorm.squeeze(1))
print("weights match cert_core")

fails = 0
keys = torch.arange(B, dtype=torch.int64) * 977
g = cc.gumbel(keys, V, "cpu")
for cap in (None, 30.0):
    for cfg in (dict(rule="greedy", cap=cap), dict(rule="gumbel", T=1.0, cap=cap), dict(rule="gumbel", T=0.6, cap=cap)):
        ref = ch.reference(h, g, cfg, cc)
        for bs in (1, 8, 64):
            toks, S, R = [], [], []
            for i in range(0, B, bs):
                tok, s, r, _ = ch.certify(h[i:i + bs], g[i:i + bs], cfg, cc)
                toks.append(tok); S.append(s); R.append(r)
            bad = (torch.cat(toks) != ref).sum().item()
            fails += bad > 0
            print(f"cap {cap} {cfg['rule']} T={cfg.get('T')} batch {bs:2d}: mismatches {bad}, "
                  f"union rows INT8 median {sorted(S)[len(S) // 2]}, BF16 median {sorted(R)[len(R) // 2]}")
print("ALL PASS" if not fails else f"{fails} FAILURES")
sys.exit(fails)
