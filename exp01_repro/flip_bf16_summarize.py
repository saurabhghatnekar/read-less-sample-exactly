"""flip_addendum_bf16.json tables. Usage: python3 -I flip_bf16_summarize.py <flip_bf16 dir> <flip dir> <model name>"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_summarize import MODE, load
from flip_summarize import any_per_prompt, rate


def main(bdir, fdir, name):
    data, f1 = load(bdir), load(fdir)
    rng = np.random.default_rng(0)
    print(f"# F1 addendum: the BF16 product against the FP32 reference, {name}\n")
    for mode, (arrs, rows, stats) in data.items():
        n = len(rows)
        assert [tuple(r) for r in rows] == [tuple(r) for r in f1[mode][1]]
        line = f"{mode}: {n} states. Reference tokens differing from the F1 run: " \
               f"{sum(v for s in stats for v in s['ref_diff'].values())}."
        if mode == "greedy":
            c1 = sum(s["c1_match"] for s in stats)
            c1e = sum(s["c1_match_exp01"] for s in stats)
            jobs = sum(s["c1_match"] != s["c1_match_exp01"] for s in stats)
            line += f" F1 corrected: BF16-product argmax equals the emitted token on {c1} states; exp01 stored " \
                    f"{c1e} ({'match in every job' if jobs == 0 else f'{jobs} jobs differ'}; {100 * c1 / n:.2f}%)."
        print(line)
    print("\nFlip rate per state against the reference, with a 95% interval from resampling prompts.\n")
    print("| Rule | BF16 product | INT8, t = 0 | INT4 dithered, t = 0 | INT4 round-to-nearest, t = 0 |")
    print("|---|---|---|---|---|")
    for rn, md in MODE.items():
        a, rows, _ = data[md]
        f = f1[md][0]
        pid = np.array([r[0] for r in rows])
        print(f"| {rn} | {rate(a[f'bf16_prod/{rn}/flip'], pid, rng)} | {rate(f[f't0/int8_row/{rn}/flip'], pid, rng)} | "
              f"{rate(f[f't0/int4d_g128/{rn}/flip'], pid, rng)} | {rate(f[f't0/int4_g128/{rn}/flip'], pid, rng)} |")
    a, rows, _ = data["greedy"]
    pid = np.array([r[0] for r in rows])
    grp = np.array([r[1] for r in rows])
    fl = a["bf16_prod/greedy/flip"].astype(bool)
    tie = a["bf16_prod/tie"].astype(bool)
    gp = a["bf16_prod/greedy/gap"][fl]
    print(f"\nGreedy: {int(fl.sum())} flips of {len(fl)}; prompts with at least one flip {any_per_prompt(fl, pid)}. "
          f"States with tied top BF16 logits: {int(tie.sum())} ({100 * tie.mean():.2f}%); flips that are ties: "
          f"{int((fl & tie).sum())} ({100 * (fl & tie).sum() / max(1, fl.sum()):.1f}% of flips).")
    if len(gp):
        print(f"Gap on flipped greedy states: median {np.median(gp):.3f}, p90 {np.percentile(gp, 90):.3f}, "
              f"max {gp.max():.3f} logits. Median largest |BF16 product - reference| over rows: "
              f"{np.median(a['bf16_prod/maxerr']):.3f} logits.")
    f = f1["greedy"][0]
    i8 = f["t0/int8_row/greedy/flip"].astype(bool)
    print(f"Overlap with INT8's greedy flips: {int((fl & i8).sum())} states flip under both "
          f"(BF16 product {int(fl.sum())}, INT8 {int(i8.sum())}).")
    print("\n| Group | States | BF16 product, greedy |")
    print("|---|---|---|")
    for x in sorted(set(grp)):
        m = grp == x
        print(f"| {x} | {int(m.sum())} | {rate(fl[m], pid[m], rng)} |")
    for rn in ("top50", "minp_0.05"):
        t = data["t1"][0]
        tp = np.array([r[0] for r in data["t1"][1]])
        print(f"\n{rn}: kept set differs on {rate(t[f'bf16_prod/{rn}/setdiff'] > 0, tp, rng)} of states; emitted token "
              f"outside the reference set on {rate(t[f'bf16_prod/{rn}/outside'], tp, rng)}.")


if __name__ == "__main__":
    main(*sys.argv[1:])
