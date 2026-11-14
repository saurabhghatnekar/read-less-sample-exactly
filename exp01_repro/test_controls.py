"""CPU test of the C6 planted faults on synthetic data.  python3 -I test_controls.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cert_core as cc  # noqa: E402
import controls_lib as cl  # noqa: E402
from test_local import fake_head, fake_states  # noqa: E402

torch.manual_seed(0)
V, d, B = 4096, 512, 64
W, h = fake_head(V, d), fake_states(B, d)
variants = {v.name: cc.prepare(v) for v in (cc.make_int8_row(W), cc.make_int4_g(W), cc.make_int4_dither(W),
                                            cc.make_int4_rot(W), cc.make_nvfp4(W))}
keys = torch.arange(B, dtype=torch.int64) * 977
fails = 0
for cap in (None, 30.0):
    rules = {"greedy": dict(rule="greedy", cap=cap), "gumbel_1.0": dict(rule="gumbel", T=1.0, cap=cap),
             "top50": dict(rule="topk", k=50, T=1.0, cap=cap), "minp_0.05": dict(rule="minp", p=0.05, T=1.0, cap=cap)}
    tot = cl.merge({}, cl.planted_chunk(cc, variants, W, W.norm(dim=1), h, keys, rules, n_fp64=16))
    checks, ok = cl.verdict(tot, list(rules), list(variants))
    print("cap", cap, checks)
    print("   fp64 max gap/slack", {v: round(tot[f"f/{v}/max_ratio"], 4) for v in variants})
    fails += not ok
print("ALL PASS" if not fails else f"{fails} FAILURES")
sys.exit(fails)
