"""CPU check of the shared-error probe on a synthetic head. Run: python3 -I test_shared_err.py"""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, cert_core as cc, shared_err_lib as se  # noqa: E402
torch.manual_seed(0)
V, d = 4096, 512
W = (torch.randn(V, d) * 0.03 + 0.03 * torch.randn(d)).bfloat16().float()
h = (torch.randn(64, d) * 2).bfloat16().float()
mu = a3_lib.mean_row(W)
vs = {}
for t in (0.0, 1.0):
    Wt = a3_lib.shifted(W, mu, t)
    for f in (cc.make_int8_row, cc.make_int4_g, cc.make_int4_dither):
        v = cc.prepare(f(Wt)); vs[f"t{t:g}/{v.name}"] = (Wt, v)
out = se.probe(h, vs)
for k, r in out.items():
    assert all(r[f"r{x}"]["identity_err"] < 1e-3 for x in ("mean", 1, 4, 16)), r
    assert all(r[f"r{x}"]["mean_ratio"] <= 1.0 + 1e-6 or x == "mean" for x in (1, 4, 16)), r
print(json.dumps(out, indent=1)[:1500])
print("shared-error probe self-test passed")
