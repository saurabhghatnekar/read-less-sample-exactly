"""CPU smoke test for crosscheck_lib on synthetic data.  python3 -I test_crosscheck.py"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert_core as cc  # noqa: E402
import crosscheck_lib as xl  # noqa: E402

spec = importlib.util.spec_from_file_location("vz_core", os.path.join(HERE, "..", "vizuara_certified_sampling", "cert_core.py"))
vz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vz)

torch.manual_seed(0)
V, d, N = 4000, 1024, 24
W = torch.randn(V, d) * 0.02 * torch.exp(torch.randn(V, d) * 0.4)
H = torch.randn(N, d) * torch.exp(torch.randn(N, d) * 0.5) * 3
keys = torch.tensor([cc.state_key(i, j, 1) for i, j in zip(range(N), range(N))], dtype=torch.int64)
hd = xl.Heads(cc, vz, W)
for cap in (None, 30.0):
    out = xl.chunk_stats(cc, vz, hd, W, H, keys, cap)
    bad = {k: v.sum().item() for k, v in out.items() if k.endswith("/mismatch") and v.sum() > 0}
    agree = {k: v.mean().item() for k, v in out.items() if k.startswith("refagree")}
    print("cap", cap, "mismatches", bad or "none", "ref agreement", agree)
    for k in sorted(out):
        if k.endswith("/bytes"):
            print(f"  {k:55s} median {out[k].median().item():.4f}")
    assert all(a == 1.0 for a in agree.values())
    assert not {k for k in bad if "/A/" not in k and "B/" in k}, bad
print("crosscheck smoke OK")
