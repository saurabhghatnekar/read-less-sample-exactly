"""Summarize Exp01 reproduction results against the lab note. Usage: python3 -I summarize.py results/qwen3_8b"""
import glob
import json
import os
import sys

import numpy as np

rdir = sys.argv[1]
rule_file = sys.argv[2] if len(sys.argv) > 2 else "decision_rule.json"
LAB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), rule_file)))["lab_note_values"]
VOCAB = 262144 if "gemma" in rdir or "sarvam" in rdir else 151936


def status(R, B):
    if np.median(B) >= 0.5:
        return "DEAD"
    if np.median(R) <= 0.001 * VOCAB and np.percentile(R, 99) <= 0.01 * VOCAB and np.percentile(B, 99) < 0.5:
        return "GATES_PASS"
    return "UNDER_HALF"


def load(mode):
    arrs, groups, stats = {}, [], []
    for f in sorted(glob.glob(f"{rdir}/{mode}_*.npz")):
        a = np.load(f)
        for k in a.files:
            arrs.setdefault(k, []).append(a[k])
        j = json.load(open(f[:-4] + ".json"))
        groups += [r[1] for r in j["rows"]]
        stats.append(j["stats"])
    return {k: np.concatenate(v) for k, v in arrs.items()}, np.array(groups), stats


G, gG, sG = load("greedy")
T, gT, sT = load("t1")
GROUPS = ["english", "code", "hindi", "bengali", "tamil", "telugu"]
RULE_MODE = {"greedy": (G, gG), "gumbel_0.6": (T, gT), "gumbel_1.0": (T, gT), "top50": (T, gT), "minp_0.05": (T, gT)}

print(f"states: greedy {len(gG)}, T=1 {len(gT)}")
# controls
c2 = sum(v for s in sG + sT for k, v in s["c2_viol"].items() if k.split("/")[1] in ("wc", "res"))
c3 = sum(v for s in sG + sT for k, v in s["c3_mismatch"].items() if "/hoeff/" not in k)
c3h = sum(v for s in sG + sT for k, v in s["c3_mismatch"].items() if "/hoeff/" in k)
hv = sum(v for s in sG + sT for k, v in s["c2_viol"].items() if "/hoeff" in k)
c1 = sum(s["c1_match"] for s in sG) / max(1, sum(s["c1_n"] for s in sG))
nmse = sum(s["dither_se"] for s in sG + sT) / sum(s["dither_n"] for s in sG + sT)
rows_checked = sum(s["dither_n"] for s in sG + sT)
print(f"C1 alignment (BF16 product argmax == emitted greedy token): {c1:.4f}")
print(f"C2 wc/res violations: {c2}   C3 non-dither mismatches: {c3}")
print(f"dither: violations {hv}, mismatches {c3h}, normalized MSE {nmse:.4f} over {rows_checked:,} rows")

# main table: dithered INT4 + INT8 stage
print("\nDithered INT4 g128 + INT8 stage (Hoeffding). S, R: median, p99. Bytes: median, p99, mean | lab median, p99")
res = {}
for rn, (A, g) in RULE_MODE.items():
    k = f"int4d_g128/hoeff/casc/{rn}"
    S, R, B = A[k + "/S"], A[k + "/R"], A[k + "/bytes"]
    b8 = A[f"int8_row/wc/single/{rn}/bytes"]
    b8r = A[f"int8_row/res/single/{rn}/bytes"]
    b8m = np.minimum(b8, b8r)
    res[rn] = dict(med=float(np.median(B)), p99=float(np.percentile(B, 99)), status=status(R, B))
    print(f"  {rn:11s} S {np.median(S):7.0f} {np.percentile(S, 99):8.0f}  R {np.median(R):4.0f} {np.percentile(R, 99):6.0f}  "
          f"bytes {np.median(B):.3f} {np.percentile(B, 99):.3f} {B.mean():.3f}  | lab {LAB['dither_int8_bytes_median'][rn]:.3f} "
          f"{LAB['dither_int8_bytes_p99'][rn]:.3f}  | INT8 cert {np.median(b8m):.3f} {np.percentile(b8m, 99):.3f}")

# best deterministic single stage
print("\nBest single-stage deterministic 4-bit certificate, median bytes:")
for rn, (A, g) in RULE_MODE.items():
    best = min(((float(np.median(A[f"{v}/{b}/single/{rn}/bytes"])), f"{v}/{b}")
                for v in ("int4_g128", "int4d_g128", "int4rot_g128", "nvfp4") for b in ("wc", "res")))
    print(f"  {rn:11s} {best[0]:.3f} ({best[1]})")

# readouts on greedy states
print("\nReadouts on greedy states (medians) | lab")
wc_mean = G["int4d_g128/wc/meanbound"]; d_mean = G["int4d_g128/hoeff/meanbound"]
true4 = G["int4_g128/maxerr"]; true4d = G["int4d_g128/maxerr"]
ro = {
    "ratio_mean_wc_over_mean_dither": np.median(wc_mean / d_mean),
    "ratio_wc_over_true_max": np.median(G["int4_g128/wc/maxbound"] / true4),
    "ratio_dither_over_true_max": np.median(G["int4d_g128/hoeff/maxbound"] / true4d),
    "true_largest_error_int4": np.median(true4),
    "top_two_margin": np.median(G["margin"]),
}
for k, v in ro.items():
    print(f"  {k:32s} {v:8.3f} | {LAB[k]}")
print(f"  {'largest dithered bound':32s} {np.median(G['int4d_g128/hoeff/maxbound']):8.3f} | {LAB.get('largest_dithered_bound', 12.6)}")
print(f"  {'top logit':32s} {np.median(G['toplogit']):8.3f} | ~38 (smoke)")
print(f"  {'h concentration l1/(sqrt(d) l2)':32s} {np.median(G['h_conc']):8.3f} | ~0.70")

# by language
print("\nMedian S of dithered+INT8 by group (greedy) | bytes median at T=1:")
for grp in GROUPS:
    mg, mt = gG == grp, gT == grp
    if mg.any():
        labS = LAB.get("median_S_greedy_by_group", {}).get(grp, "")
        print(f"  {grp:8s} S {np.median(G['int4d_g128/hoeff/casc/greedy/S'][mg]):6.0f} | lab {labS:>4}   "
              f"bytes T=1 {np.median(T['int4d_g128/hoeff/casc/gumbel_1.0/bytes'][mt]):.3f}   "
              f"bytes p99 T=1 {np.percentile(T['int4d_g128/hoeff/casc/gumbel_1.0/bytes'][mt], 99):.3f}")

# verdict per decision_rule.json
checks = []
checks.append(("C2/C3 clean", c2 == 0 and c3 == 0))
for rn in RULE_MODE:
    checks.append((f"median {rn} within 0.01", abs(res[rn]["med"] - LAB["dither_int8_bytes_median"][rn]) <= 0.01))
for rn in ("greedy", "gumbel_0.6", "gumbel_1.0", "minp_0.05"):
    checks.append((f"p99 {rn} within 0.05", abs(res[rn]["p99"] - LAB["dither_int8_bytes_p99"][rn]) <= 0.05))
for k, v in ro.items():
    checks.append((f"{k} within 30%", abs(v - LAB[k]) <= 0.3 * LAB[k]))
for rn, st in LAB.get("statuses", {}).items():
    checks.append((f"status {rn}: ours {res[rn]['status']}, lab {st}", res[rn]["status"] == st))
print("\nDecision-rule checks:")
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
dead = not (c2 == 0 and c3 == 0) or any(abs(res[r]["med"] - LAB["dither_int8_bytes_median"][r]) > 0.05 for r in ("greedy", "gumbel_1.0"))
print("OUTCOME:", "NOT REPRODUCED" if dead else ("REPRODUCED" if all(ok for _, ok in checks) else "PARTIAL"))
