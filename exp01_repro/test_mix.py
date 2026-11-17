"""CPU check for M1 (mix_rule.json, control M0_cpu). Run: python3 -I test_mix.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, cert_core as cc, hdir_lib, mix_lib, subg_lib  # noqa: E402
from test_a3 import fake_head, fake_states, RULES_G, RULES_T  # noqa: E402

torch.manual_seed(0)
V, d, B, G8 = 4096, 512, 48, 2
W = fake_head(V, d)
mu = a3_lib.mean_row(W)
w_norm = W.norm(dim=1)

# 1. with no INT8 group the quantizer is cert_core's, bit for bit
a, b = cc.make_int4_dither(W), mix_lib.make_mixed_dither(W, ())
assert torch.equal(a.Wq, b.Wq) and torch.equal(a.scale, b.scale) and torch.equal(a.dither, b.dither)
assert a.bytes_per_weight == b.bytes_per_weight and a.name == b.name
print("no INT8 group: identical to cert_core.make_int4_dither")

# 2. with one, the other groups are unchanged, the codes fit INT8, the error is within half the step
m = mix_lib.make_mixed_dither(W, (G8,))
sl = slice(G8 * 128, (G8 + 1) * 128)
other = torch.ones(d, dtype=torch.bool)
other[sl] = False
assert torch.equal(m.Wq[:, other], a.Wq[:, other]) and torch.equal(m.scale[:, [0, 1, 3]], a.scale[:, [0, 1, 3]])
s8 = m.scale[:, G8:G8 + 1]
codes = torch.round(m.Wq[:, sl] / s8 + m.dither[sl])
assert float(codes.abs().max()) <= 127
assert float(((m.Wq[:, sl] - W[:, sl]).abs() / s8).max()) <= 0.5 + 1e-3
ratio = float((a.scale[:, G8] / m.scale[:, G8]).median())
assert 19 < ratio < 20, ratio                                  # 126.5 / 6.5 = 19.46
assert abs(m.bytes_per_weight - (0.5 + 2 / 128 + 64 / d)) < 1e-12
print(f"one INT8 group: other groups unchanged, codes within 127, error within half a step, step {ratio:.1f} times smaller")

# 3. the cascade with the mixed head: no violations, no mismatches, and the bound narrows as predicted
Pall, _ = hdir_lib.fit_directions(mu, fake_states(256, d, mu), kmax=4)
p = Pall[:, 1:2]
GROUPS = ("tamil", "english")
for cap in (None, 30.0):
    for mode, rules in (("greedy", RULES_G), ("t1", RULES_T)):
        rules = {k: dict(x, cap=cap) for k, x in rules.items()}
        h = fake_states(B, d, mu)
        h[: B // 2, sl] *= 6                                   # half the states have their energy in the group
        h = h.bfloat16().float()
        rows = [dict(pid=i // 8, group=GROUPS[i >= B // 2], step=i % 8, mode=mode) for i in range(B)]
        heavy = torch.arange(B) < B // 2
        for t in (0.0, 0.5):
            vs = a3_lib.build(W, mu, t)
            Wt = a3_lib.shifted(W, mu, t)
            v = vs["int4d_g128"]
            mm = cc.prepare(mix_lib.make_mixed_dither(Wt, (G8,)))
            mm.t, mm.wt_norm = t, v.wt_norm
            q, share, inside = mix_lib.group_direction(h[heavy], p[:, 0], G8)
            assert abs(float(q @ p[:, 0])) < 1e-9 and abs(float(q.norm()) - 1) < 1e-9
            P2 = torch.cat([p, q[:, None]], 1)
            out = {}
            for head, var in (("plain", v), ("mixed", mm)):
                r = hdir_lib.err_proj(var.Wq, Wt, P2)
                sets = {"k2": {g: (p, r[:, :1]) for g in GROUPS}}
                if head == "plain":
                    sets["k2_dir"] = {g: (P2, r) for g in GROUPS}
                rec, st = hdir_lib.run_chunk_sets(W, mu, w_norm, {"int4d_g128": var, "int8_row": vs["int8_row"]}, sets,
                                                  h, rows, rules)
                assert sum(st["viol"].values()) == 0 and sum(st["mismatch"].values()) == 0, (head, st)
                out[head] = rec
            tag = f"t{t:g}"
            ref, _ = subg_lib.run_chunk(W, mu, h, rows, rules, grid=(t,))
            for rn in rules:
                for qn in ("S", "R", "bytes"):
                    assert torch.equal(out["plain"][f"{tag}/none/{rn}/{qn}"], ref[f"{tag}/int4d_g128/subg/casc/{rn}/{qn}"])
            # the variance term with the group's step replaced, against the measured bound ratio (slack aside)
            hg2 = (h.double() ** 2).view(B, -1, 128).sum(-1)
            pred = torch.sqrt((hg2 @ (mm.scale.double() ** 2).T) / (hg2 @ (v.scale.double() ** 2).T)).median(1).values
            got = (out["mixed"][f"{tag}/none/b_med"] / out["plain"][f"{tag}/none/b_med"]).double()
            assert float((got[heavy] - pred[heavy]).abs().max()) < 0.05, float((got[heavy] - pred[heavy]).abs().max())
            assert float(got[heavy].median()) < 0.7 and float(got[~heavy].median()) > 0.85
            rn = next(iter(rules))
            floor_gap = out["mixed"][f"{tag}/none/{rn}/bytes"].min() - out["plain"][f"{tag}/none/{rn}/bytes"].min()
            assert float(floor_gap) <= 64 / (2 * d) + 1e-6
        # planted fault: the bound computed with INT8 steps on a head that is INT4 everywhere must be violated
        bad = cc.prepare(cc.make_int4_dither(Wt))
        bad.t, bad.wt_norm, bad.scale = t, v.wt_norm, mm.scale
        _, sb = hdir_lib.run_chunk_sets(W, mu, w_norm, {"int4d_g128": bad, "int8_row": vs["int8_row"]}, {}, h, rows, rules)
        assert sum(sb["viol"].values()) > 0
        print(f"cap={cap} {mode}: 0 violations, 0 mismatches; plain none = B1; bound ratio on heavy states "
              f"{float(got[heavy].median()):.2f} (predicted {float(pred[heavy].median()):.2f}), others "
              f"{float(got[~heavy].median()):.2f}; direction inside the group holds {share:.2f} of its energy; "
              f"planted fault caught")

# 4. the heaviest group is found
S = mix_lib.variance_weights(a)
arg, top = mix_lib.largest_group(h[heavy], S)
assert int(arg.mode().values) == G8 and float(top.median()) > 0.5
print("M1 self-tests passed")
