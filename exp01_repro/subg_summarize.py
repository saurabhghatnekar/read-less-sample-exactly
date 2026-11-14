"""B1 tables and controls. Usage: python3 -I subg_summarize.py <subg dir> <a3 dir> <model name> <V>"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_summarize import MODE, boot_diff, load, q

T_ORDER = [0.0, 1.0, 0.5, 1.5]


def tg(t):
    return f"t{t:g}"


def status(R, B, V):
    if np.median(B) >= 0.5:
        return "DEAD"
    if np.median(R) <= 0.001 * V and np.percentile(R, 99) <= 0.01 * V and np.percentile(B, 99) < 0.5:
        return "GATES_PASS"
    return "UNDER_HALF"


def main(sdir, a3dir, name, V):
    V = int(V)
    data, a3 = load(sdir), load(a3dir)
    rng = np.random.default_rng(0)
    print(f"# B1 sub-Gaussian bound, {name}\n")
    for mode, (arrs, rows, stats) in data.items():
        viol = {t: sum(s["viol"].get(t, 0) for s in stats) for t in stats[0]["viol"]}
        exc = {t: max(s["viol_excess_max"][t] for s in stats) for t in stats[0]["viol"]}
        mism = sum(v for s in stats for k, v in s["mismatch"].items())
        se = {t: sum(s["dither_se"][t] for s in stats) / sum(s["dither_n"][t] for s in stats) for t in stats[0]["dither_se"]}
        aa, ar, _ = a3[mode]
        assert [tuple(r) for r in ar] == [tuple(r) for r in rows]
        diff = sum(int((arrs[k] != aa[k]).sum()) for k in arrs if "/hoeff/casc/" in k)
        ratio = np.median(arrs["t0/int4d_g128/subg/meanbound"] / arrs["t0/int4d_g128/hoeff/meanbound"])
        print(f"{mode}: {len(rows)} states. S1 values differing from A3: {diff}. S2 sub-Gaussian violations {viol}, "
              f"largest excess {max(exc.values()):+.4f}. Dithered-token mismatches {mism}. S4 {{" +
              ", ".join(f"{k}: {v:.4f}" for k, v in se.items()) + f"}}. Bound ratio subg/hoeff {ratio:.4f}")
    print()
    for cert, label in (("subg/casc", "sub-Gaussian + INT8"), ("subg/single", "sub-Gaussian, single stage"),
                        ("hoeff/casc", "Hoeffding + INT8 (A3 headline)")):
        print(f"## {label}\n")
        print("| Rule | t* | Status (all states, t*) | Test bytes at t* (median, p99, mean) | R median, p99 | "
              "Δmean vs Hoeffding+INT8 at t=0 [95%] |")
        print("|---|---|---|---|---|---|")
        for rn, md in MODE.items():
            arrs, rows, _ = data[md]
            pid = np.array([r[0] for r in rows])
            val, test = pid % 2 == 0, pid % 2 == 1
            means = [arrs[f"{tg(t)}/int4d_g128/{cert}/{rn}/bytes"][val].mean() for t in T_ORDER]
            ts = T_ORDER[int(np.argmin(means))]
            k = f"{tg(ts)}/int4d_g128/{cert}/{rn}"
            B, R = arrs[k + "/bytes"], arrs[k + "/R"]
            base = arrs[f"t0/int4d_g128/hoeff/casc/{rn}/bytes"]
            d = boot_diff(B[test], base[test], pid[test], rng)
            print(f"| {rn} | {ts:g} | {status(R, B, V)} | {q(B[test])} | {np.median(R):.0f}, {np.percentile(R, 99):.0f} | "
                  f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] |")
        print()
    arrs, rows, _ = data["greedy"]
    g = np.array([r[1] for r in rows])
    print("## Sub-Gaussian + INT8, greedy, t = 1: bytes by language group (median, p99, mean)\n")
    B = arrs["t1/int4d_g128/subg/casc/greedy/bytes"]
    for x in sorted(set(g)):
        print(f"- {x}: {q(B[g == x])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
