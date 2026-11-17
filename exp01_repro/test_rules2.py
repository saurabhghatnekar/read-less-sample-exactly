"""CPU check for T1's rules and the pipeline harness. Run: python3 -I test_rules2.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, cert_core as cc, hdir_lib, pipe_lib, rules2_lib  # noqa: E402
from test_a3 import fake_head, fake_states, RULES_G, RULES_T  # noqa: E402

torch.manual_seed(0)

# 1. top-p membership bounds are sound: for values drawn inside the intervals, "in" rows are members and "out" rows are not
bad = n_in = n_out = 0
for trial in range(300):
    V = 14
    mid = torch.randn(1, V, dtype=torch.float64) * 2.5
    wid = torch.rand(1, V, dtype=torch.float64) * (0.0 if trial % 5 == 0 else 1.5) * (torch.rand(1, V) < 0.7)
    aL, aU = mid - wid, mid + wid
    p = [0.5, 0.9, 0.99][trial % 3]
    cin, cout, _, _ = rules2_lib.topp_membership(aL, aU, p)
    for _ in range(20):
        a = aL + (aU - aL) * torch.rand(1, V, dtype=torch.float64)
        w = torch.softmax(a, 1)[0]
        member = torch.tensor([float(w[a[0] > a[0, i]].sum()) < p for i in range(V)])
        bad += int((cin[0] & ~member).sum()) + int((cout[0] & member).sum())
    n_in += int(cin.sum())
    n_out += int(cout.sum())
assert bad == 0 and n_in > 300 and n_out > 300, (bad, n_in, n_out)
print(f"top-p membership: sound on 6,000 draws ({n_in} certain-in and {n_out} certain-out decisions)")

# 2. the top-p reference decision against a plain loop
for cap in (None, 30.0):
    z = torch.randn(6, 200) * 3
    g = cc.gumbel(torch.arange(6, dtype=torch.int64) * 31, 200, "cpu")
    cfg = dict(rule="topp", p=0.9, T=1.0, cap=cap)
    got = rules2_lib.decide(z, cfg, g)
    for b in range(6):
        c = cc.softcap(z[b], cap).double()
        w = torch.softmax(c, 0)
        member = [float(w[c > c[i]].sum()) < 0.9 for i in range(200)]
        s = cc.softcap(z[b], cap) + g[b]
        want = max((i for i in range(200) if member[i]), key=lambda i: float(s[i]))
        assert int(got[b]) == want
print("top-p reference decision equals a plain loop")

# 3. the pipeline harness on a synthetic head: every arm and rule returns the reference token
V, d, B = 4096, 512, 48
W = fake_head(V, d)
mu = a3_lib.mean_row(W)
w_norm = W.norm(dim=1)
Pall, _ = hdir_lib.fit_directions(mu, fake_states(256, d, mu), kmax=4)
p1 = Pall[:, 1:2]
G8 = 2
for cap in (None, 30.0):
    for mode, base in (("greedy", RULES_G), ("t1", RULES_T)):
        rules = {k: dict(x, cap=cap) for k, x in base.items()}
        if mode == "t1":
            rules["top50_lean"] = dict(rule="topk_lean", k=50, T=1.0, cap=cap)
            rules["topp_0.9"] = dict(rule="topp", p=0.9, T=1.0, cap=cap)
        h = fake_states(B, d, mu)
        h[: B // 2, G8 * 128:(G8 + 1) * 128] *= 6
        h = h.bfloat16().float()
        rows = [dict(pid=i // 8, group=("tamil", "english")[i >= B // 2], step=i % 8, mode=mode) for i in range(B)]
        vs, Wt = pipe_lib.make_variants(W, mu, 0.5, groups8=(G8,), dith8=True)
        r = {k: hdir_lib.err_proj(vs[k].Wq, Wt, p1) for k in ("i4", "mix", "d8")}
        src = {"i4": dict(v=vs["i4"], kind="subg", sets=None),
               "i4k": dict(v=vs["i4"], kind="subg", sets=(p1, r["i4"])),
               "mixk": dict(v=vs["mix"], kind="subg", sets=(p1, r["mix"]), row_bytes=204),
               "i8": dict(v=vs["i8"], kind="det", row_bytes=d + 4),
               "d8k": dict(v=vs["d8"], kind="subg", sets={g: (p1, r["d8"]) for g in ("tamil", "english")},
                           row_bytes=d + 2 * (d // 128) + 8)}
        arms = {"none": dict(first="i4", stages=["i8"]),
                "k2": dict(first="i4k", stages=["i8"], first_extra=4),
                "k2_d8": dict(first="i4k", stages=["d8k"], first_extra=4),
                "staged_d8": dict(first="i4k", stages=["mixk", "d8k"], first_extra=4)}
        rec, st = pipe_lib.run_chunk(W, mu, w_norm, h, rows, rules, src, arms)
        assert sum(st["viol"].values()) == 0 and sum(st["mismatch"].values()) == 0, (st["viol"], st["mismatch"])
        # the arm k2 is hdir_lib's path, to the row
        vs_old = a3_lib.build(W, mu, 0.5)
        old, _ = hdir_lib.run_chunk_sets(W, mu, w_norm, vs_old, {"k2": (p1, hdir_lib.err_proj(vs_old["int4d_g128"].Wq, Wt, p1))},
                                         h, rows, {k: v for k, v in rules.items() if v["rule"] not in ("topk_lean", "topp")})
        for rn, cfg in rules.items():
            if cfg["rule"] in ("topk_lean", "topp"):
                continue
            assert torch.equal(rec[f"k2/{rn}/S_i8"], old[f"t0.5/k2/{rn}/S"]) and torch.equal(rec[f"k2/{rn}/R"], old[f"t0.5/k2/{rn}/R"])
            assert torch.allclose(rec[f"k2/{rn}/bytes"], old[f"t0.5/k2/{rn}/bytes"], atol=2e-6)
            assert torch.equal(rec[f"none/{rn}/S_i8"], old[f"t0.5/none/{rn}/S"])
        # planted fault: the dithered INT8 stage with numbers from other rows
        bad_src = dict(src, d8k=dict(v=vs["d8"], kind="subg", sets=(p1, r["d8"].roll(1, 0)), row_bytes=1))
        _, sb = pipe_lib.run_chunk(W, mu, w_norm, h, rows, rules, bad_src, {"x": arms["k2_d8"]})
        assert sb["viol"]["d8k"] > 0
        msg = f"cap={cap} {mode}: 0 violations, 0 mismatches in 4 arms x {len(rules)} rules; k2 = K2's path; planted fault caught"
        if mode == "t1":
            a, b = rec["k2/top50/R"].sum(), rec["k2/top50_lean/R"].sum()
            c, e = rec["k2_d8/top50/R"].sum(), rec["k2/topp_0.9/R"].sum()
            ratio = float((rec["bmed/d8k"] / rec["bmed/i8"]).median())
            msg += (f"; BF16 rows top-50 {a:.0f}, lean {b:.0f}, with dithered INT8 {c:.0f}; top-p {e:.0f}; "
                    f"dithered INT8 bound / worst-case INT8 bound {ratio:.2f}")
        print(msg)
print("T1 self-tests passed")
