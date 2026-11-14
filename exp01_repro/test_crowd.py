"""CPU check for K1. Run: python3 -I test_crowd.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, cert_core as cc, crowd_lib, subg_lib  # noqa: E402
from test_a3 import fake_head, fake_states  # noqa: E402

torch.manual_seed(0)
V, d, B = 4096, 512, 32
W = fake_head(V, d); mu = a3_lib.mean_row(W)
w_norm = W.norm(dim=1)
for cap in (None, 30.0):
    h = fake_states(B, d) * 0.3
    copies = {"t0": a3_lib.build(W, mu, 0.0)["int4d_g128"], "ts": a3_lib.build(W, mu, 0.5)["int4d_g128"]}
    rec = crowd_lib.run_chunk(W, mu, w_norm, copies, h, cap)
    z = h @ W.T
    c = cc.softcap(z, cap)
    cfg = dict(rule="greedy", cap=cap)
    g = torch.zeros(B, V)
    for i in range(B):
        row = sorted(c[i].tolist(), reverse=True)
        for k in crowd_lib.KS:
            if k <= V:
                assert abs(float(rec[f"gap_k{k}"][i]) - (torch.tensor(row[0]) - torch.tensor(row[k - 1])).item()) < 1e-6
        for w in crowd_lib.WIDTHS:
            thr = (torch.tensor(row[0], dtype=torch.float32) - w).item()
            assert int(rec[f"n_within_{w:g}"][i]) == sum(x >= thr for x in row), (cap, i, w)
    assert "gap_k10000" not in rec                                  # V = 4096 here
    for tag, v in copies.items():
        zt, bounds, slack = a3_lib.intervals(v, h, w_norm, mu)
        for bn, b in (("hoeff", bounds["hoeff"]), ("subg", subg_lib.subg_bound(v, h, slack))):
            if tag != "t0" and bn == "hoeff":
                continue
            _, R, tok = cc.certify(z, zt, b, cfg, g, None)        # single stage: the rows it reads
            assert torch.equal(rec[f"{tag}/{bn}/need_actual"], R.sum(1).float()), (tag, bn)
            assert torch.equal(tok, c.argmax(1))
            ex = rec[f"{tag}/{bn}/need_exact"]
            assert (ex >= 1).all()
            # exact centres, direct count: rows with z + b >= max(z - b), through the cap
            L, U = cc.softcap(z - b, cap), cc.softcap(z + b, cap)
            assert torch.equal(ex, (U >= L.amax(1, keepdim=True)).sum(1).float())
    print(f"cap={cap}: gaps and counts match brute force; need_actual equals certify's read set "
          f"(median {float(rec['t0/subg/need_actual'].median()):.0f} actual, "
          f"{float(rec['t0/subg/need_exact'].median()):.0f} exact centres)")
print("K1 self-tests passed")
