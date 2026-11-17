"""CPU check for M2 (stage_rule.json, control S0_cpu). Run: python3 -I test_stage.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, cert_core as cc, hdir_lib, mix_lib, stage_lib  # noqa: E402
from test_a3 import fake_head, fake_states, RULES_G, RULES_T  # noqa: E402

torch.manual_seed(0)
V, d, B, G8 = 4096, 512, 48, 2
W = fake_head(V, d)
mu = a3_lib.mean_row(W)
w_norm = W.norm(dim=1)
Pall, _ = hdir_lib.fit_directions(mu, fake_states(256, d, mu), kmax=4)
p = Pall[:, 1:2]
GROUPS = ("tamil", "english")
sl = slice(G8 * 128, (G8 + 1) * 128)
for cap in (None, 30.0):
    for mode, rules in (("greedy", RULES_G), ("t1", RULES_T)):
        rules = {k: dict(x, cap=cap) for k, x in rules.items()}
        h = fake_states(B, d, mu)
        h[: B // 2, sl] *= 6                                   # half the states have their energy in the group
        h = h.bfloat16().float()
        rows = [dict(pid=i // 8, group=GROUPS[i >= B // 2], step=i % 8, mode=mode) for i in range(B)]
        t = 0.5
        vs = a3_lib.build(W, mu, t)
        Wt = a3_lib.shifted(W, mu, t)
        v = vs["int4d_g128"]
        m = cc.prepare(mix_lib.make_mixed_dither(Wt, (G8,)))
        m.t, m.wt_norm = t, v.wt_norm
        sp = {g: (p, hdir_lib.err_proj(v.Wq, Wt, p)) for g in GROUPS}
        sm = {g: (p, hdir_lib.err_proj(m.Wq, Wt, p)) for g in GROUPS}
        # 1. the group stage given the first-pass intervals themselves changes nothing
        same, st = stage_lib.run_chunk_staged(W, mu, w_norm, v, v, vs["int8_row"], sp, sp, h, rows, rules)
        assert sum(st["viol"].values()) == 0 and sum(st["mismatch"].values()) == 0
        for rn in rules:
            for q in ("S", "R"):
                assert torch.equal(same[f"staged/{rn}/{q}"], same[f"k2/{rn}/{q}"]), (rn, q)
        # 2. k2 here is hdir_lib's lolo path
        old, _ = hdir_lib.run_chunk_sets(W, mu, w_norm, vs, {"k2": sp}, h, rows, rules)
        for rn in rules:
            for q in ("S", "R", "bytes"):
                assert torch.equal(same[f"k2/{rn}/{q}"], old[f"t0.5/k2/{rn}/{q}"]), (rn, q)
        # 3. the real staged certificate: exact tokens, and fewer INT8 rows on the heavy states
        rec, st = stage_lib.run_chunk_staged(W, mu, w_norm, v, m, vs["int8_row"], sp, sm, h, rows, rules)
        assert sum(st["viol"].values()) == 0 and sum(st["mismatch"].values()) == 0, st
        heavy = torch.arange(B) < B // 2
        for rn in rules:
            assert float(rec[f"staged/{rn}/S"].sum()) <= float(rec[f"k2/{rn}/S"].sum()), rn
            gap = rec[f"staged/{rn}/bytes"] - rec[f"staged/{rn}/bytes_kept_sum"]
            assert torch.allclose(gap, rec[f"staged/{rn}/Sg"] * 66 / (2.0 * V * d), atol=1e-7)
        # 4. planted fault: group-stage numbers taken from other rows
        bad = {g: (p, sm[g][1].roll(1, 0)) for g in GROUPS}
        _, sb = stage_lib.run_chunk_staged(W, mu, w_norm, v, m, vs["int8_row"], sp, bad, h, rows, rules)
        assert sb["viol"]["group"] > 0
        rn = "top50" if "top50" in rules else next(iter(rules))
        print(f"cap={cap} {mode}: identity stage changes nothing; k2 = K2's path; staged: 0 violations, 0 mismatches; "
              f"{rn} heavy states INT8 rows {float(rec[f'k2/{rn}/S'][heavy].mean()):.0f} -> "
              f"{float(rec[f'staged/{rn}/S'][heavy].mean()):.0f} (group rows {float(rec[f'staged/{rn}/Sg'][heavy].mean()):.0f}), "
              f"bytes {float(rec[f'k2/{rn}/bytes'][heavy].mean()):.4f} -> {float(rec[f'staged/{rn}/bytes'][heavy].mean()):.4f}; "
              f"planted fault caught")
print("M2 self-tests passed")
