"""K2 tables and controls. Usage: python3 -I hdir_summarize.py <hdir dir> <model name> <V>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_summarize import boot_diff, load, q
from subg_summarize import status

SETS = ("none", "mu", "pca1", "pca4", "pca16", "pca64")
MODE = {"greedy": "greedy", "gumbel_1.0": "t1"}


def main(hdir, name, V):
    V = int(V)
    data = load(hdir)
    info = json.load(open(f"{hdir}/P.json"))
    ts = json.load(open(sorted(f for f in (os.path.join(hdir, x) for x in os.listdir(hdir))
                               if f.endswith(".json") and not f.endswith("P.json"))[0]))["t_star"]
    tags = ("t0", f"t{ts:g}")
    rng = np.random.default_rng(0)
    print(f"# K2 state-direction correction, {name}\n")
    print(f"Directions: {info['n_cal']} calibration states (greedy, even prompt ids). Share of their second moment in "
          f"the top 1, 4, 16, 64 eigenvectors: " + ", ".join(f"{x:.3f}" for x in info["share"]) +
          f". Cosine of mu with the top eigenvector: {info['cos_mu_pc1']:+.3f}. ||mu|| = {info['mu_norm']:.3f}.\n")
    for mode, (arrs, rows, stats) in data.items():
        viol = sum(v for s in stats for v in s["viol"].values())
        exc = max(v for s in stats for v in s["viol_excess_max"].values())
        mism = sum(v for s in stats for v in s["mismatch"].values())
        h0 = sum(s["h0_diff"] for s in stats)
        print(f"{mode}: {len(rows)} states. H0 values differing from B1: {h0}. H1 bound violations: {viol} "
              f"(largest excess {exc:+.4f} logits). H2 token mismatches: {mism}.")
    print()
    for rn, md in MODE.items():
        arrs, rows, _ = data[md]
        pid = np.array([r[0] for r in rows])
        test = pid % 2 == 1
        for tg in tags:
            print(f"## {rn}, {tg}: sub-Gaussian + INT8\n")
            print("| Directions | Share of the state removed, median | Median-row bound vs none, median | "
                  "Top-row bound vs none, median | S median, p99 | Status (all states) | "
                  "Test bytes (median, p99, mean) | p99 bytes, all states | Δmean vs none [95%] |")
            print("|---|---|---|---|---|---|---|---|---|")
            base = arrs[f"{tg}/none/{rn}/bytes"]
            for s in SETS:
                k = f"{tg}/{s}"
                B, S, R = arrs[f"{k}/{rn}/bytes"], arrs[f"{k}/{rn}/S"], arrs[f"{k}/{rn}/R"]
                d = boot_diff(B[test], base[test], pid[test], rng)
                print(f"| {s} | {np.median(arrs[f'{k}/rho'][test]):.3f} | "
                      f"{np.median(arrs[f'{k}/b_med'][test] / arrs[f'{tg}/none/b_med'][test]):.3f} | "
                      f"{np.median(arrs[f'{k}/b_top'][test] / arrs[f'{tg}/none/b_top'][test]):.3f} | "
                      f"{np.median(S):.0f}, {np.percentile(S, 99):.0f} | {status(R, B, V)} | {q(B[test])} | "
                      f"{np.percentile(B, 99):.3f} | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] |")
            print()
    arrs, rows, _ = data["greedy"]
    pid = np.array([r[0] for r in rows])
    g = np.array([r[1] for r in rows])
    tg = tags[1]
    print(f"## Greedy, {tg}, by language group (all states)\n")
    print("| Group | Share removed by mu, median | by pca16 | Mean bytes: none | mu | pca16 | p99 bytes: none | mu | pca16 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(set(g)):
        m = g == x
        print(f"| {x} | {np.median(arrs[f'{tg}/mu/rho'][m]):.3f} | {np.median(arrs[f'{tg}/pca16/rho'][m]):.3f} | " +
              " | ".join(f"{arrs[f'{tg}/{s}/greedy/bytes'][m].mean():.3f}" for s in ("none", "mu", "pca16")) + " | " +
              " | ".join(f"{np.percentile(arrs[f'{tg}/{s}/greedy/bytes'][m], 99):.3f}" for s in ("none", "mu", "pca16")) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
