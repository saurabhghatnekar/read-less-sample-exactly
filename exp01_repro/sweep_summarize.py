"""K3 table. Usage: python3 -I sweep_summarize.py  (reads logs/sweep_<model>.json)"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"qwen3_8b": "Qwen3-8B", "gemma4_e4b": "Gemma 4 E4B", "sarvam_30b": "Sarvam-30B", "qwen3_4b": "Qwen3-4B",
         "qwen3_14b": "Qwen3-14B", "qwen3_30b_a3b": "Qwen3-30B-A3B (MoE)", "gemma4_e2b": "Gemma 4 E2B",
         "phi4_mini": "Phi-4-mini", "sarvam_m": "Sarvam-M", "sarvam_1": "Sarvam-1", "mistral_nemo": "Mistral-Nemo",
         "granite33_8b": "Granite 3.3 8B", "smollm3_3b": "SmolLM3-3B", "gpt_oss_20b": "gpt-oss-20b (MoE)"}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    R, failed = [], []
    for f in sorted(glob.glob(os.path.join(HERE, "logs", "sweep_*.json"))):
        if f.endswith("_smoke.json"):
            continue
        r = json.load(open(f))
        (failed if "failed" in r else R).append(r)
    R.sort(key=lambda r: -r["spectrum"]["top1"])
    print(f"# K3: one-direction share of the final hidden states, {len(R)} models\n")
    for r in failed:
        print(f"FAILED {r['name']}: {r['failed'][:200]}")
    print("Control A1 (BF16-product argmax equals the emitted token): " +
          ", ".join(f"{NAMES[r['name']]} {r['alignment']:.3f}" for r in R) + ".")
    certs = [r for r in R if r["certificate"]]
    print(f"Control A3: bound violations {sum(r['certificate']['violations'] for r in certs)}, token mismatches "
          f"{sum(r['certificate']['mismatches'] for r in certs)}, over {len(certs)} models with a certificate.\n")
    print("| Model | V | d | States' second moment in the top 1, 4, 16 directions | Share of the state along the top "
          "direction, median | Offset share of the top logit, median | Bound with pca1 vs none | "
          "Bytes none: median, p99, mean | Bytes pca1: median, p99, mean | S p99 none → pca1 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        c, s = r["certificate"], r["spectrum"]
        if c:
            b = lambda k: f"{c[k]['bytes']['median']:.3f}, {c[k]['bytes']['p99']:.3f}, {c[k]['bytes']['mean']:.3f}"
            tail = f"{c['pca1']['bound_vs_none']:.2f} | {b('none')} | {b('pca1')} | {c['none']['S']['p99']:,.0f} → {c['pca1']['S']['p99']:,.0f}"
        else:
            tail = "no certificate (d not a multiple of 128) | | |"
        print(f"| {NAMES[r['name']]} | {r['V']:,} | {r['d']:,} | {s['top1']:.2f}, {s['top4']:.2f}, {s['top16']:.2f} | "
              f"{r['rho_pca1']['median']:.2f} | {r['offset_share']['median']:+.2f} | {tail} |")
    t1 = np.array([r["spectrum"]["top1"] for r in R])
    off = np.array([r["offset_share"]["median"] for r in R])
    rho = np.array([r["rho_pca1"]["median"] for r in R])
    print(f"\nAcross the {len(R)} models: Spearman of the top-1 share with the offset share {spearman(t1, off):+.2f}, "
          f"with its absolute value {spearman(t1, np.abs(off)):+.2f}, with the median share of the state along the top "
          f"direction {spearman(t1, rho):+.2f}.")
    gain = np.array([r["certificate"]["none"]["bytes"]["p99"] - r["certificate"]["pca1"]["bytes"]["p99"] for r in certs])
    print("p99 bytes saved by pca1: " + ", ".join(f"{NAMES[r['name']]} {g:+.3f}" for r, g in zip(certs, gain)) + ".")
    print("Revisions: " + "; ".join(f"{r['meta']['repo']}@{r['meta']['rev'][:10]}" for r in R) + ".")


if __name__ == "__main__":
    main()
