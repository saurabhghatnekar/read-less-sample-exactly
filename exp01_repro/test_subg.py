"""CPU check for B1. Run: python3 -I test_subg.py"""
import math, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, subg_lib  # noqa: E402
from test_a3 import fake_head, fake_states, RULES_G, RULES_T  # noqa: E402
torch.manual_seed(0)
V, d, B = 4096, 512, 48
W = fake_head(V, d); mu = a3_lib.mean_row(W)
assert abs(subg_lib.subg_c() / (math.sqrt(math.log(2e15) / 2)) - 1 / math.sqrt(3)) < 1e-12
for cap in (None, 30.0):
    for mode, rules in (("greedy", RULES_G), ("t1", RULES_T)):
        rules = {k: dict(x, cap=cap) for k, x in rules.items()}
        h = fake_states(B, d, mu)
        rows = [dict(pid=i // 8, group="english", step=i % 8, mode=mode) for i in range(B)]
        rec, st = subg_lib.run_chunk(W, mu, h, rows, rules)
        ref, _ = a3_lib.run_chunk(W, mu, h, rows, rules, grid=subg_lib.T_GRID)
        same = sum(int((rec[k] != ref[k]).sum()) for k in rec if "/hoeff/casc/" in k)
        assert same == 0, f"Hoeffding cascade differs from A3 on {same} values"
        assert sum(st["viol"].values()) == 0, st["viol"]
        ratio = float((rec["t0/int4d_g128/subg/meanbound"] / rec["t0/int4d_g128/hoeff/meanbound"]).median())
        sub = float(rec["t0/int4d_g128/subg/casc/" + next(iter(rules)) + "/bytes"].mean())
        hof = float(rec["t0/int4d_g128/hoeff/casc/" + next(iter(rules)) + "/bytes"].mean())
        print(f"cap={cap} {mode}: S1 match A3, S2 violations 0 (largest excess {max(st['viol_excess_max'].values()):.3f}), "
              f"bound ratio {ratio:.3f}, mean bytes hoeff {hof:.3f} -> subg {sub:.3f}")
print("B1 self-tests passed")
