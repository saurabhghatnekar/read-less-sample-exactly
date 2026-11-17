"""M1, two things read off the per-state arrays (not a new run):

1. the gate status and the BF16 rows of each arm, per rule;
2. an ESTIMATE of a staged variant: K2's first pass on the plain head, then group 26 in INT8 only for the rows
   the rule still needs (130 bytes each: 128 codes and an FP16 step), then the later stages as the arm `both`
   reads them. bytes_staged = bytes_both - 64 / (2 d) + S_k2 * 130 / (2 V d).

  modal volume get exp01-repro mix/sarvam_30b results/mix/
  python3 -I mix_staged_estimate.py > logs/mix_staged_estimate.txt      # also writes logs/mix_staged_estimate.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from subg_summarize import status  # noqa: E402

V, D = 262144, 4096
ARMS = {"none": "plain/none", "k2": "plain/k2", "int8g": "mixed/none", "both": "mixed/k2", "k2_g26dir": "plain/k2_g26dir"}
RULES = {"greedy": "greedy", "gumbel_0.6": "t1", "gumbel_1.0": "t1", "top50": "t1", "minp_0.05": "t1"}
LANGS = ("bengali", "code", "english", "hindi", "tamil", "telugu", "all")


def q(x):
    return f"{np.median(x):.4f}, {np.percentile(x, 99):.4f}, {x.mean():.4f}"


def main():
    rdir = os.path.join(HERE, "results", "mix", "sarvam_30b")
    data = {}
    for mode in ("greedy", "t1"):
        arr, rows = {}, []
        for f in sorted(glob.glob(f"{rdir}/{mode}_*.npz")):
            z = np.load(f)
            rows += json.load(open(f[:-4] + ".json"))["rows"]
            for k in z.files:
                arr.setdefault(k, []).append(z[k])
        data[mode] = ({k: np.concatenate(v) for k, v in arr.items()}, np.array([r[1] for r in rows]))
    print("Gate status over all states, and BF16 rows (median, p99), per arm")
    for rn, md in RULES.items():
        arr, _ = data[md]
        print(f"  {rn}: " + "; ".join(
            f"{a} {status(arr[f'{k}/{rn}/R'], arr[f'{k}/{rn}/bytes'], V)} ({np.median(arr[f'{k}/{rn}/R']):.0f}, "
            f"{np.percentile(arr[f'{k}/{rn}/R'], 99):.0f})" for a, k in ARMS.items()))
    print("\nStaged variant, ESTIMATED. Bytes (median, p99, mean): k2 | both | staged; mean of staged - k2; share of states lower")
    out = {}
    for rn, md in RULES.items():
        arr, grp = data[md]
        st = arr[f"mixed/k2/{rn}/bytes"] - 64 / (2 * D) + arr[f"plain/k2/{rn}/S"] * 130 / (2.0 * V * D)
        for g in LANGS:
            mk = np.ones(len(grp), bool) if g == "all" else grp == g
            k2, bo, s = arr[f"plain/k2/{rn}/bytes"][mk], arr[f"mixed/k2/{rn}/bytes"][mk], st[mk]
            out[f"{rn}/{g}"] = dict(k2=[float(np.median(k2)), float(np.percentile(k2, 99)), float(k2.mean())],
                                    staged=[float(np.median(s)), float(np.percentile(s, 99)), float(s.mean())],
                                    staged_minus_k2_mean=float((s - k2).mean()), staged_lower=float((s < k2).mean()))
            print(f"  {rn:10s} {g:8s} {q(k2)} | {q(bo)} | {q(s)}; {np.mean(s - k2):+.4f}; {100 * np.mean(s < k2):.0f}%")
    json.dump(out, open(os.path.join(HERE, "logs", "mix_staged_estimate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
