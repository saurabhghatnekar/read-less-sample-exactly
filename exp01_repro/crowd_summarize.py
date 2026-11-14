"""K1 tables. Usage: python3 -I crowd_summarize.py <crowd results dir> <flip_bf16 results dir>
Both hold one folder per model (qwen3_8b, gemma4_e4b, sarvam_30b)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_summarize import load

MODELS = (("qwen3_8b", "Qwen3-8B"), ("gemma4_e4b", "Gemma 4 E4B"), ("sarvam_30b", "Sarvam-30B"))
KS = (2, 10, 100, 1000, 10000)
WIDTHS = (1, 2, 4, 8, 16, 32, 64)


def med(x):
    return float(np.median(x))


def row(label, vals, fmt="{:.2f}"):
    print(f"| {label} | " + " | ".join(fmt.format(v) if v is not None else "" for v in vals) + " |")


def main(cdir, bdir):
    D = {m: load(f"{cdir}/{m}")["greedy"] for m, _ in MODELS}
    print("# K1: crowding of the top reference logits (greedy states)\n")
    for m, name in MODELS:
        a, rows, stats = D[m]
        k0 = {k: sum(s["k0"][k] for s in stats) for k in ("a3", "subg")}
        print(f"{name}: {len(rows)} states. K0 shortlist values differing from A3 (Hoeffding): {k0['a3']}, "
              f"from B1 (sub-Gaussian): {k0['subg']}.")
    hdr = "| Median over greedy states | " + " | ".join(n for _, n in MODELS) + " |"
    sep = "|---|" + "---|" * len(MODELS)

    print("\n## Logit scale, offset and bounds\n")
    print(hdr); print(sep)
    A = [D[m][0] for m, _ in MODELS]
    row("Raw top logit", [med(a["top_raw"]) for a in A])
    row("Offset h . mu", [med(a["mu_dot_h"]) for a in A])
    row("Offset share of the raw top logit", [med(a["mu_dot_h"] / a["top_raw"]) for a in A])
    row("Top logit minus the median logit (capped)", [med(a["top"] - a["logit_median"]) for a in A])
    row("Standard deviation of the logits (capped)", [med(a["logit_std"]) for a in A])
    row("Top probability at T = 1", [med(a["p_top"]) for a in A], "{:.3f}")
    row("Entropy at T = 1 (nats)", [med(a["entropy"]) for a in A], "{:.3f}")
    row("Hoeffding bound, t = 0: median row", [med(a["t0/hoeff/b_med"]) for a in A])
    row("Hoeffding bound, t = 0: top row", [med(a["t0/hoeff/b_top"]) for a in A])
    row("Sub-Gaussian bound, t = 0: median row", [med(a["t0/subg/b_med"]) for a in A])
    row("Sub-Gaussian bound, B1 shift: median row", [med(a["ts/subg/b_med"]) for a in A])
    row("Bound in units of the logit spread (2 x median Hoeffding bound / (top - median logit))",
        [med(2 * a["t0/hoeff/b_med"] / (a["top"] - a["logit_median"])) for a in A])

    print("\n## Gaps below the top logit (capped logits)\n")
    print(hdr); print(sep)
    for k in KS:
        row(f"Top minus the {k}-th largest", [med(a[f"gap_k{k}"]) for a in A])
    print("\n## Tokens within a fixed distance of the top (capped logits)\n")
    print(hdr); print(sep)
    for w in WIDTHS:
        row(f"Within {w} logits", [med(a[f"n_within_{w}"]) for a in A], "{:.0f}")

    print("\n## Headroom in bound units: gap / (2 x median Hoeffding bound, t = 0)\n")
    print(hdr); print(sep)
    for k in KS:
        row(f"{k}-th largest", [med(a[f"gap_k{k}"] / (2 * a["t0/hoeff/b_med"])) for a in A])
    print("\nSame with the sub-Gaussian bound at the B1 shift\n")
    print(hdr); print(sep)
    for k in KS:
        row(f"{k}-th largest", [med(a[f"gap_k{k}"] / (2 * a["ts/subg/b_med"])) for a in A])

    print("\n## Shortlist: copy's own centres against exact centres\n")
    print(hdr); print(sep)
    for key, label in (("t0/hoeff", "Hoeffding, t = 0"), ("t0/subg", "sub-Gaussian, t = 0"),
                       ("ts/subg", "sub-Gaussian, B1 shift")):
        row(f"{label}: need_actual, median", [med(a[f"{key}/need_actual"]) for a in A], "{:.0f}")
        row(f"{label}: need_exact, median", [med(a[f"{key}/need_exact"]) for a in A], "{:.0f}")
        row(f"{label}: need_actual, p99", [float(np.percentile(a[f"{key}/need_actual"], 99)) for a in A], "{:.0f}")
        row(f"{label}: need_exact, p99", [float(np.percentile(a[f"{key}/need_exact"], 99)) for a in A], "{:.0f}")
        row(f"{label}: median of need_actual / need_exact",
            [med(a[f"{key}/need_actual"] / a[f"{key}/need_exact"]) for a in A])
        row(f"{label}: sum of need_actual / sum of need_exact",
            [float(a[f"{key}/need_actual"].sum() / a[f"{key}/need_exact"].sum()) for a in A])

    print("\n## By language group\n")
    groups = sorted(set(r[1] for r in D["sarvam_30b"][1]))
    for m, name in MODELS:
        a, rows, _ = D[m]
        g = np.array([r[1] for r in rows])
        print(f"{name}\n")
        print("| Group | need_actual (Hoeffding, t = 0), median | Within 16 logits, median | "
              "Top minus 1000-th, median | 2 x median Hoeffding bound, median | Entropy, median |")
        print("|---|---|---|---|---|---|")
        for x in groups:
            s = g == x
            print(f"| {x} | {med(a['t0/hoeff/need_actual'][s]):.0f} | {med(a['n_within_16'][s]):.0f} | "
                  f"{med(a['gap_k1000'][s]):.2f} | {med(2 * a['t0/hoeff/b_med'][s]):.2f} | {med(a['entropy'][s]):.3f} |")
        print()

    print("## Stock-BF16 ties against the raw top logit\n")
    print("| Model | Median raw top logit | Tie share, lower half by raw top logit | Tie share, upper half | "
          "Tie share, raw top below 64 | 64 to 128 | 128 and above |")
    print("|---|---|---|---|---|---|---|")
    for m, name in MODELS:
        a, rows, _ = D[m]
        b, brows, _ = load(f"{bdir}/{m}")["greedy"]
        assert [tuple(r) for r in rows] == [tuple(r) for r in brows]
        tie = b["bf16_prod/tie"].astype(bool)
        top = np.abs(a["top_raw"])
        lo = top <= np.median(top)

        def sh(mask):
            return f"{100 * tie[mask].mean():.2f}% ({int(mask.sum())})" if mask.any() else "none"
        print(f"| {name} | {np.median(a['top_raw']):.1f} | {sh(lo)} | {sh(~lo)} | {sh(top < 64)} | "
              f"{sh((top >= 64) & (top < 128))} | {sh(top >= 128)} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
