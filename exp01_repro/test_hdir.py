"""CPU check for K2. Run: python3 -I test_hdir.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, cert_core as cc, hdir_lib, subg_lib  # noqa: E402
from test_a3 import fake_head, fake_states, RULES_G, RULES_T  # noqa: E402

torch.manual_seed(0)
V, d, B = 4096, 512, 48
W = fake_head(V, d); mu = a3_lib.mean_row(W)
w_norm = W.norm(dim=1)
Hcal = fake_states(256, d, mu)
Pall, info = hdir_lib.fit_directions(mu, Hcal)
P = hdir_lib.split(Pall)
assert set(P) == {"mu", "pca1", "pca4", "pca16", "pca64"} and P["pca16"].shape == (d, 16)
assert torch.allclose(Pall[:, 1:].T @ Pall[:, 1:], torch.eye(64, dtype=torch.float64), atol=1e-9)
for cap in (None, 30.0):
    for mode, rules in (("greedy", RULES_G), ("t1", {"gumbel_1.0": RULES_T["gumbel_1.0"]})):
        rules = {k: dict(x, cap=cap) for k, x in rules.items()}
        h = fake_states(B, d, mu)                            # half the states lean on the mean row
        rows = [dict(pid=i // 8, group="english", step=i % 8, mode=mode) for i in range(B)]
        for t in (0.0, 0.5):
            vs = a3_lib.build(W, mu, t)
            v = vs["int4d_g128"]
            Wt = a3_lib.shifted(W, mu, t)
            rall = hdir_lib.err_proj(v.Wq, Wt, Pall)
            R = {n: (rall[:, :1] if n == "mu" else rall[:, 1:1 + P[n].shape[1]]) for n in P}
            rec, st = hdir_lib.run_chunk(W, mu, w_norm, vs, P, R, h, rows, rules)
            tag = f"t{t:g}"
            # 1. no correction reproduces subg_lib (B1) bit for bit
            ref, _ = subg_lib.run_chunk(W, mu, h, rows, rules, grid=(t,))
            for rn in rules:
                for q in ("S", "R", "bytes"):
                    assert torch.equal(rec[f"{tag}/none/{rn}/{q}"], ref[f"{tag}/int4d_g128/subg/casc/{rn}/{q}"]), (rn, q)
            # 2. no violations, no mismatches
            assert sum(st["viol"].values()) == 0, st["viol"]
            assert sum(st["mismatch"].values()) == 0, st["mismatch"]
            # 3. the corrected error is (h - P alpha) . e_v, computed directly in float64
            zt, _, slack = a3_lib.intervals(v, h, w_norm, mu)
            E = v.Wq.double() - Wt.double()
            for n in ("mu", "pca16"):
                ztc, b, rho = hdir_lib.corrected(v, P[n], R[n], h, zt, slack)
                hp = h.double() - (h.double() @ P[n]) @ P[n].T
                direct = hp @ E.T                             # remainder of the error
                base = (zt.double() - h.double() @ E.T)       # the logit the copy would give with no error
                assert float((ztc.double() - base - direct).abs().max()) < 1e-2, n
                assert ((ztc - h @ W.T).abs() <= b).all()
                assert float(rho.min()) >= -1e-9 and float(rho.max()) <= 1 + 1e-9
            # 4. states that lean on mu lose most of their length, and their bound narrows
            lean = torch.arange(B) % 2 == 0
            assert float(rec[f"{tag}/mu/rho"][lean].median()) > 0.5
            ratio = rec[f"{tag}/mu/b_med"][lean] / rec[f"{tag}/none/b_med"][lean]
            assert float(ratio.median()) < 0.8, float(ratio.median())
            # 5. planted fault: r taken from rolled rows breaks the bound
            Rbad = {n: x.roll(1, 0) for n, x in R.items()}
            _, sb = hdir_lib.run_chunk(W, mu, w_norm, vs, {"mu": P["mu"]}, {"mu": Rbad["mu"]}, h, rows, rules)
            assert sb["viol"][f"{tag}/mu"] > 0
            rn = next(iter(rules))
            print(f"cap={cap} {mode} t={t:g}: none = B1; 0 violations, 0 mismatches; lean states rho "
                  f"{float(rec[f'{tag}/mu/rho'][lean].median()):.2f}, bound ratio {float(ratio.median()):.2f}; "
                  f"mean bytes none {float(rec[f'{tag}/none/{rn}/bytes'].mean()):.3f}, "
                  f"mu {float(rec[f'{tag}/mu/{rn}/bytes'].mean()):.3f}, "
                  f"pca16 {float(rec[f'{tag}/pca16/{rn}/bytes'].mean()):.3f}; planted fault caught")
print("K2 self-tests passed", info)

# ---- hdir_addendum_robust.json: five rules and per-group directions
GROUPS = ("english", "hindi")
Pg = {}
for i, gname in enumerate(GROUPS):                          # one fit per group, from different states
    Pa, _ = hdir_lib.fit_directions(mu, fake_states(128, d, mu if i == 0 else None) + 0.1 * i, kmax=4)
    Pg[gname] = Pa[:, 1:5]
for cap in (None, 30.0):
    for mode, rules in (("greedy", RULES_G), ("t1", RULES_T)):
        rules = {k: dict(x, cap=cap) for k, x in rules.items()}
        h = fake_states(B, d, mu)
        for t in (0.0, 0.5):
            vs = a3_lib.build(W, mu, t)
            v = vs["int4d_g128"]
            Wt = a3_lib.shifted(W, mu, t)
            rg = {gname: hdir_lib.err_proj(v.Wq, Wt, Pg[gname]) for gname in GROUPS}
            rmu = hdir_lib.err_proj(v.Wq, Wt, P["mu"])
            tag = f"t{t:g}"
            # mixed groups in one chunk
            rows = [dict(pid=i // 8, group=GROUPS[(i // 8) % 2], step=i % 8, mode=mode) for i in range(B)]
            sets = {"mu": (P["mu"], rmu), "lolo4": {gn: (Pg[gn], rg[gn]) for gn in GROUPS},
                    "lolo1": {gn: (Pg[gn][:, :1], rg[gn][:, :1]) for gn in GROUPS}}
            rec, st = hdir_lib.run_chunk_sets(W, mu, w_norm, vs, sets, h, rows, rules)
            assert sum(st["viol"].values()) == 0 and sum(st["mismatch"].values()) == 0, (st["viol"], st["mismatch"])
            ref, _ = subg_lib.run_chunk(W, mu, h, rows, rules, grid=(t,))
            old, _ = hdir_lib.run_chunk(W, mu, w_norm, vs, {"mu": P["mu"]}, {"mu": rmu}, h, rows, rules)
            for rn in rules:
                for qn in ("S", "R", "bytes"):
                    assert torch.equal(rec[f"{tag}/none/{rn}/{qn}"], ref[f"{tag}/int4d_g128/subg/casc/{rn}/{qn}"])
                    assert torch.equal(rec[f"{tag}/mu/{rn}/{qn}"], old[f"{tag}/mu/{rn}/{qn}"])
            # a chunk of one group equals the fixed-set path with that group's directions
            rows1 = [dict(r, group="hindi") for r in rows]
            a, _ = hdir_lib.run_chunk_sets(W, mu, w_norm, vs, {"x": {"hindi": (Pg["hindi"], rg["hindi"])}}, h, rows1, rules)
            b_, _ = hdir_lib.run_chunk_sets(W, mu, w_norm, vs, {"x": (Pg["hindi"], rg["hindi"])}, h, rows1, rules)
            assert all(torch.equal(a[k], b_[k]) for k in a)
        print(f"cap={cap} {mode}: five-rule path reproduces B1 and K2; per-group directions: 0 violations, 0 mismatches")
print("K2 addendum self-tests passed")
