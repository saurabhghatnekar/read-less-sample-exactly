"""F1 tables and controls. Usage: python3 -I flip_summarize.py <flip dir> <model name>"""
import json
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_summarize import MODE, boot_diff, load

COPIES = (("int4_g128", "INT4 round-to-nearest"), ("int4d_g128", "INT4 dithered"), ("int8_row", "INT8"))


def rate(x, pid, rng):
    m, lo, hi = boot_diff(x.astype(float), np.zeros(len(x)), pid, rng)
    return f"{100 * m:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]"


def any_per_prompt(x, pid):
    ups = np.unique(pid)
    hit = np.bincount(np.searchsorted(ups, pid), x.astype(float), minlength=len(ups)) > 0
    return f"{int(hit.sum())} of {len(ups)}"


def main(fdir, name):
    data = load(fdir)
    ts = json.load(open(sorted(glob.glob(f"{fdir}/*.json"))[0]))["t_star"]
    tags = ("t0", f"t{ts:g}")
    rng = np.random.default_rng(0)
    print(f"# F1 flip rate of uncertified heads, {name}\n")
    for mode, (arrs, rows, stats) in data.items():
        n = len(rows)
        f0 = sum(v for s in stats for v in s["f0_diff"].values())
        f0m = max(v for s in stats for v in s["f0_max"].values())
        eq = {k: sum(s["ref_eq_emitted"][k] for s in stats) / n for k in stats[0]["ref_eq_emitted"]}
        line = f"{mode}: {n} states, {len(set(r[0] for r in rows))} prompts. F0 largest-error values differing " \
               f"from A3: {f0} (largest difference {f0m:.2e} logits)."
        if mode == "greedy":
            line += f" F1 reference token equals the emitted token on {100 * eq['greedy']:.2f}% of states."
        print(line)
    print(f"\nShifts: t = 0 and t = {ts:g}. Flip rate per state, with a 95% interval from resampling prompts.\n")

    print("## Flip rate by rule and copy\n")
    print("| Rule | Copy | t = 0 | t = " + f"{ts:g}" + " |")
    print("|---|---|---|---|")
    for rn, md in MODE.items():
        arrs, rows, _ = data[md]
        pid = np.array([r[0] for r in rows])
        for vn, label in COPIES:
            print(f"| {rn} | {label} | " + " | ".join(rate(arrs[f"{tg}/{vn}/{rn}/flip"], pid, rng) for tg in tags) + " |")
    print()

    print("## Prompts with at least one flip (states follow the reference trajectory)\n")
    print("| Rule | Copy | t = 0 | t = " + f"{ts:g}" + " |")
    print("|---|---|---|---|")
    for rn in ("greedy", "gumbel_1.0"):
        arrs, rows, _ = data[MODE[rn]]
        pid = np.array([r[0] for r in rows])
        for vn, label in COPIES:
            print(f"| {rn} | {label} | " + " | ".join(any_per_prompt(arrs[f"{tg}/{vn}/{rn}/flip"], pid) for tg in tags) + " |")
    print()

    arrs, rows, _ = data["greedy"]
    pid = np.array([r[0] for r in rows])
    grp = np.array([r[1] for r in rows])
    print("## Greedy flip rate by language group\n")
    print("| Group | States | INT4 round-to-nearest, t = 0 | INT4 dithered, t = " + f"{ts:g}" + " | INT8, t = 0 |")
    print("|---|---|---|---|---|")
    for x in sorted(set(grp)):
        m = grp == x
        print(f"| {x} | {int(m.sum())} | {rate(arrs['t0/int4_g128/greedy/flip'][m], pid[m], rng)} | "
              f"{rate(arrs[f'{tags[1]}/int4d_g128/greedy/flip'][m], pid[m], rng)} | "
              f"{rate(arrs['t0/int8_row/greedy/flip'][m], pid[m], rng)} |")
    print()

    print("## How far off the flipped token is (gap in reference score, flipped states only)\n")
    print("| Rule | Copy, shift | Flips | Gap median | Gap p90 | Gap max | Share of flips with gap > 1 |")
    print("|---|---|---|---|---|---|---|")
    for rn in ("greedy", "gumbel_0.6", "gumbel_1.0"):
        a = data[MODE[rn]][0]
        for vn, label, tg in (("int4_g128", "INT4 round-to-nearest", "t0"), ("int4d_g128", "INT4 dithered", tags[1]),
                              ("int8_row", "INT8", "t0")):
            fl = a[f"{tg}/{vn}/{rn}/flip"].astype(bool)
            gp = a[f"{tg}/{vn}/{rn}/gap"][fl]
            if len(gp) == 0:
                print(f"| {rn} | {label}, {tg} | 0 | | | | |")
                continue
            print(f"| {rn} | {label}, {tg} | {len(gp)} | {np.median(gp):.3f} | {np.percentile(gp, 90):.3f} | "
                  f"{gp.max():.3f} | {100 * (gp > 1).mean():.1f}% |")
    print()

    print("## Truncation rules: kept set and emitted token\n")
    print("| Rule | Copy, shift | States whose kept set differs | Median set difference where it differs | "
          "Emitted token outside the reference set |")
    print("|---|---|---|---|---|")
    for rn in ("top50", "minp_0.05"):
        a, rows, _ = data[MODE[rn]]
        pid = np.array([r[0] for r in rows])
        for vn, label, tg in (("int4_g128", "INT4 round-to-nearest", "t0"), ("int4d_g128", "INT4 dithered", tags[1]),
                              ("int8_row", "INT8", "t0")):
            sd = a[f"{tg}/{vn}/{rn}/setdiff"]
            med = f"{np.median(sd[sd > 0]):.0f}" if (sd > 0).any() else ""
            print(f"| {rn} | {label}, {tg} | {rate(sd > 0, pid, rng)} | {med} | "
                  f"{rate(a[f'{tg}/{vn}/{rn}/outside'], pid, rng)} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
