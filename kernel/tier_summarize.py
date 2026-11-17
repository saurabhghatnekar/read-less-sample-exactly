"""Summarise results/tier_<model>_<gpu>.json (tier_protocol.json). Plain Python.

  python3 -I tier_summarize.py [gpu] > logs/tier_summary_a100.txt
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def st(x):
    s = sorted(x)
    return s[len(s) // 2], s[int(0.99 * (len(s) - 1))], s[-1]


def main(gpu="A100-80GB"):
    for model in ("qwen3_8b", "gemma4_e4b", "sarvam_30b", "qwen3_0.6b"):
        path = os.path.join(HERE, "results", f"tier_{model}_{gpu}.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        b = r["bytes"]
        h = b["bf16_head"]
        print(f"== {model} on {r['gpu']}: V {r['V']}, d {r['d']}; BF16 head {h / 1e9:.2f} GB")
        print(f"GPU bytes as a fraction of the BF16 head: all copies {b['layout_all'] / h:.3f}, layout B {b['layout_B'] / h:.3f}, "
              f"layout A {b['layout_A'] / h:.3f}; work buffers {b['work_buffers_per_state'] / h:.4f} per state in a batch")
        for rn, rr in r["rules"].items():
            for B, c in rr.items():
                print(f"\n{rn}, batch {B}: {c['batches']} steps; mismatches {c['mismatches']}; steps whose row counts differ "
                      f"from the all-GPU kernel {c['rows_differ']}; slow-path steps {c['slow_steps']}")
                ex, i8 = st(c["exact_rows_fetched"]), st(c["int8_rows_fetched"])
                print(f"  rows fetched from the host per step (median, p99, largest): BF16 {ex[0]}, {ex[1]}, {ex[2]}; "
                      f"INT8 (layout A) {i8[0]}, {i8[1]}, {i8[2]}")
                base = st(c["ms"]["stock_graph"])[0]
                print("  ms               median     p99  largest   median / stock head")
                for k, name in (("stock_graph", "stock BF16 head"), ("all", "all on the GPU"), ("B", "layout B"), ("A", "layout A")):
                    m, p, mx = st(c["ms"][k])
                    print(f"  {name:16s} {m:6.3f} {p:7.3f} {mx:8.3f}   {m / base:.2f}")
        print()


if __name__ == "__main__":
    main(*sys.argv[1:2])
