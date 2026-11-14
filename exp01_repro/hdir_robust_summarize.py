"""K2 addendum tables. Usage: python3 -I hdir_robust_summarize.py <hdir_robust dir> <model name> <V>"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_summarize import boot_diff, load, q
from subg_summarize import status

SETS = ("none", "mu", "pca1", "lolo1", "pca4", "lolo4")
MODE = {"greedy": "greedy", "gumbel_0.6": "t1", "gumbel_1.0": "t1", "top50": "t1", "minp_0.05": "t1"}


def main(rdir, name, V):
    V = int(V)
    data = load(rdir)
    lolo = json.load(open(f"{rdir}/P_lolo.json"))["info"]
    ts = json.load(open(sorted(f for f in glob.glob(f"{rdir}/*.json") if not f.endswith("P_lolo.json"))[0]))["t_star"]
    tags = ("t0", f"t{ts:g}")
    rng = np.random.default_rng(0)
    print(f"# K2 addendum: five rules and leave-one-language-out directions, {name}\n")
    print("Left-out fits, |cosine| of the top eigenvector with K2's pca1: " +
          ", ".join(f"{g} {abs(v['cos_with_pca1']):.4f}" for g, v in lolo.items()) + ".\n")
    for mode, (arrs, rows, stats) in data.items():
        viol = sum(v for s in stats for v in s["viol"].values())
        exc = max(v for s in stats for v in s["viol_excess_max"].values())
        mism = sum(v for s in stats for v in s["mismatch"].values())
        print(f"{mode}: {len(rows)} states. H0 values differing from B1: {sum(s['h0_diff'] for s in stats)}. "
              f"H1 bound violations: {viol} (largest excess {exc:+.4f} logits). H2 token mismatches: {mism}.")
    print()
    for rn, md in MODE.items():
        arrs, rows, _ = data[md]
        pid = np.array([r[0] for r in rows])
        test = pid % 2 == 1
        for tg in tags:
            print(f"## {rn}, {tg}: sub-Gaussian + INT8\n")
            print("| Directions | Bound vs none, median | S median, p99 | R median, p99 | Status (all states) | "
                  "Test bytes (median, p99, mean) | p99 bytes, all states | Max bytes, all states | Δmean vs none [95%] |")
            print("|---|---|---|---|---|---|---|---|---|")
            base = arrs[f"{tg}/none/{rn}/bytes"]
            for s in SETS:
                k = f"{tg}/{s}"
                B, S, R = arrs[f"{k}/{rn}/bytes"], arrs[f"{k}/{rn}/S"], arrs[f"{k}/{rn}/R"]
                d = boot_diff(B[test], base[test], pid[test], rng)
                print(f"| {s} | {np.median(arrs[f'{k}/b_med'] / arrs[f'{tg}/none/b_med']):.3f} | "
                      f"{np.median(S):.0f}, {np.percentile(S, 99):.0f} | {np.median(R):.0f}, {np.percentile(R, 99):.0f} | "
                      f"{status(R, B, V)} | {q(B[test])} | {np.percentile(B, 99):.3f} | {B.max():.3f} | "
                      f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] |")
            print()
    arrs, rows, _ = data["greedy"]
    g = np.array([r[1] for r in rows])
    tg = tags[1]
    print(f"## Greedy, {tg}, by language group (all states): directions fitted with and without the group\n")
    print("| Group | Share of the state removed: pca1 | lolo1 | p99 bytes: none | pca1 | lolo1 | Mean bytes: pca1 | lolo1 | "
          "S p99: pca1 | lolo1 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(set(g)):
        m = g == x
        f = lambda s, fn: fn(arrs[f"{tg}/{s}/greedy/bytes"][m])
        print(f"| {x} | {np.median(arrs[f'{tg}/pca1/rho'][m]):.3f} | {np.median(arrs[f'{tg}/lolo1/rho'][m]):.3f} | "
              f"{f('none', lambda a: np.percentile(a, 99)):.3f} | {f('pca1', lambda a: np.percentile(a, 99)):.3f} | "
              f"{f('lolo1', lambda a: np.percentile(a, 99)):.3f} | {f('pca1', np.mean):.4f} | {f('lolo1', np.mean):.4f} | "
              f"{np.percentile(arrs[f'{tg}/pca1/greedy/S'][m], 99):.0f} | {np.percentile(arrs[f'{tg}/lolo1/greedy/S'][m], 99):.0f} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
