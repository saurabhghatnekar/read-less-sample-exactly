"""A3 tables, controls and verdict. Usage: python3 -I a3_summarize.py <a3 results dir> [<exp01 results dir>]"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = json.load(open(os.path.join(HERE, "a3_decision_rule.json")))
GRID = RULE["grid"]
MODE = {"greedy": "greedy", "gumbel_0.6": "t1", "gumbel_1.0": "t1", "top50": "t1", "minp_0.05": "t1"}
CERTS = {"headline": "int4d_g128/hoeff/casc", "dither_single": "int4d_g128/hoeff/single",
         "rtn_res_casc": "int4_g128/res/casc", "int8": "int8_row/min/single"}
B = 10_000


def tg(t):
    return f"t{t:g}"


def load(d):
    data = {}
    for mode in ("greedy", "t1"):
        arrs, rows, stats = {}, [], []
        for f in sorted(glob.glob(f"{d}/{mode}_*.npz")):
            z = np.load(f)
            for k in z.files:
                arrs.setdefault(k, []).append(z[k])
            j = json.load(open(f[:-4] + ".json"))
            rows += j["rows"]
            stats.append(j["stats"])
        if rows:
            data[mode] = ({k: np.concatenate(v) for k, v in arrs.items()}, rows, stats)
    return data


def boot_diff(x, y, pid, rng):
    """Mean of (x - y) with a 95% interval from resampling prompts."""
    ups = np.unique(pid)
    idx = np.searchsorted(ups, pid)
    sx = np.bincount(idx, x - y, minlength=len(ups))
    n = np.bincount(idx, minlength=len(ups)).astype(float)
    draws = rng.integers(0, len(ups), (B, len(ups)))
    m = sx[draws].sum(1) / n[draws].sum(1)
    return float((x - y).mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def q(a):
    return f"{np.median(a):.3f}, {np.percentile(a, 99):.3f}, {a.mean():.3f}"


def main(a3dir, exp01dir=None):
    data = load(a3dir)
    rng = np.random.default_rng(0)
    name = os.path.basename(a3dir.rstrip("/"))
    print(f"# A3, {name}\n")

    # ---------------- controls
    ok = True
    for mode, (arrs, rows, stats) in data.items():
        c2 = sum(sum(s["c2_viol"].values()) for s in stats)
        c3 = sum(v for s in stats for k, v in s["c3_mismatch"].items() if "/hoeff/" not in k)
        se = {t: sum(s["dither_se"].get(tg(t), 0) for s in stats) / max(1, sum(s["dither_n"].get(tg(t), 0) for s in stats))
              for t in GRID}
        c6 = [(s["c6_shift_states"], s.get("c6_shift_n"), s["c6_eligible"], s["c6_eligible_missed"])
              for s in stats if s.get("c6_shift_states") is not None]
        c5ok = all(0.98 <= v <= 1.02 for v in se.values())
        c6ok = bool(c6) and all(m == 0 for *_, m in c6)
        # invalid_if in the rule names C2, C3, T0 and C6_shift; C5 is reported, with the out-of-band t values
        ok &= c2 == 0 and c3 == 0 and c6ok
        out5 = {f"{t:g}": round(v, 4) for t, v in se.items() if not 0.98 <= v <= 1.02}
        print(f"{mode}: {len(rows)} states. C2 violations {c2}, C3 mismatches {c3}, "
              f"C5 dither SE {min(se.values()):.4f}-{max(se.values()):.4f} "
              f"({'in band at every t' if c5ok else 'outside 0.98-1.02 at t=' + str(out5)}), "
              f"C6_shift (caught, of, eligible, eligible missed) {c6}")
    if exp01dir:
        ref = load(exp01dir)
        bad = 0
        for mode, (arrs, rows, _) in data.items():
            if mode not in ref:
                continue
            ea, er, _ = ref[mode]
            n = len(rows)   # a smoke run covers a prefix of the exp01 states
            assert [tuple(r) for r in er[:n]] == [tuple(r) for r in rows], "state order differs from exp01"
            ea = {k: v[:n] for k, v in ea.items()}
            for rn, md in MODE.items():
                if md != mode:
                    continue
                for c in ("int4d_g128/hoeff/casc", "int4d_g128/hoeff/single", "int4_g128/res/casc"):
                    for f in ("bytes", "S", "R"):
                        a, b = arrs[f"t0/{c}/{rn}/{f}"], ea[f"{c}/{rn}/{f}"]
                        bad += int((a != b).sum())
        print(f"T0 control: {bad} per-state values differ from exp01")
        ok &= bad == 0
    print("Controls:", "PASS" if ok else "FAIL (run invalid under the rule)", "\n")

    # ---------------- selection on validation (even pid), report on test (odd pid)
    verdict = {}
    for cname, cert in CERTS.items():
        print(f"## {cname} ({cert})\n")
        print("| Rule | t* | Test bytes t=0 (median, p99, mean) | t=1 | t* | Δmean t*−0 [95%] | Δmean t=1−0 [95%] |")
        print("|---|---|---|---|---|---|---|")
        for rn, md in MODE.items():
            if md not in data:
                continue
            arrs, rows, _ = data[md]
            pid = np.array([r[0] for r in rows])
            val, test = pid % 2 == 0, pid % 2 == 1
            means = [arrs[f"{tg(t)}/{cert}/{rn}/bytes"][val].mean() for t in GRID]
            ts = GRID[int(np.argmin(means))]   # argmin keeps the first of equal values
            b = {t: arrs[f"{tg(t)}/{cert}/{rn}/bytes"][test] for t in (0.0, 1.0, ts)}
            d1 = boot_diff(b[ts], b[0.0], pid[test], rng)
            d2 = boot_diff(b[1.0], b[0.0], pid[test], rng)
            print(f"| {rn} | {ts:g} | {q(b[0.0])} | {q(b[1.0])} | {q(b[ts])} | "
                  f"{d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] | {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}] |")
            if cname == "headline" and rn in ("greedy", "gumbel_1.0"):
                verdict[rn] = "HURTS" if d1[1] > 0 else ("HELPS" if ts != 0 and d1[2] < 0 else "NO_EFFECT")
        print()

    # ---------------- bound widths and sets across the whole grid (all states, greedy mode)
    if "greedy" in data:
        arrs, rows, _ = data["greedy"]
        print("## Greedy states across the grid (medians; bound widths relative to t=0)\n")
        print("| t | Hoeffding mean bound | residual mean bound (INT4 RTN) | Hoeffding / true max error | headline S median, p99 | R p99 | bytes p99 |")
        print("|---|---|---|---|---|---|---|")
        h0 = arrs["t0/int4d_g128/hoeff/meanbound"]
        r0 = arrs["t0/int4_g128/res/meanbound"]
        for t in sorted(GRID):
            k = tg(t)
            S = arrs[f"{k}/int4d_g128/hoeff/casc/greedy/S"]
            R = arrs[f"{k}/int4d_g128/hoeff/casc/greedy/R"]
            by = arrs[f"{k}/int4d_g128/hoeff/casc/greedy/bytes"]
            print(f"| {t:g} | {np.median(arrs[k + '/int4d_g128/hoeff/meanbound'] / h0):.3f} | "
                  f"{np.median(arrs[k + '/int4_g128/res/meanbound'] / r0):.3f} | "
                  f"{np.median(arrs[k + '/int4d_g128/hoeff/maxbound'] / arrs[k + '/int4d_g128/maxerr']):.2f} | "
                  f"{np.median(S):.0f}, {np.percentile(S, 99):.0f} | {np.percentile(R, 99):.0f} | {np.percentile(by, 99):.3f} |")
        print()
        groups = sorted({r[1] for r in rows})
        g = np.array([r[1] for r in rows])
        print("## Headline greedy bytes mean by language group\n")
        print("| t | " + " | ".join(groups) + " |\n|---|" + "---|" * len(groups))
        for t in sorted(GRID):
            by = arrs[f"{tg(t)}/int4d_g128/hoeff/casc/greedy/bytes"]
            print(f"| {t:g} | " + " | ".join(f"{by[g == x].mean():.3f}" for x in groups) + " |")
        print()

    v = set(verdict.values())
    final = "HURTS" if "HURTS" in v else ("HELPS" if v == {"HELPS"} else "NO_EFFECT")
    print(f"Verdict under a3_decision_rule.json: {final} ({verdict})" if ok else "Verdict withheld: controls failed")


if __name__ == "__main__":
    main(*sys.argv[1:])
