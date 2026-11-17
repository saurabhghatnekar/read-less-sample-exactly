"""Summarise results/fused_tail_<model>_L4.json (protocol: tail_protocol.json). Plain Python, no dependencies.

  python3 -I tail_summarize.py > logs/fused_tail_summary.txt
  python3 -I tail_summarize.py A100-80GB > logs/fused_tail_summary_a100.txt      # a100_protocol.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = {"1": 128, "8": 32, "32": 16}                    # batches the first run timed (fused_modal.py)
BINS = ((0, 100), (100, 1000), (1000, 10000), (10000, 30000), (30000, 10 ** 9))


def med(x):
    x = sorted(x)
    return x[len(x) // 2]


def stats(x):
    s = sorted(x)
    return dict(median=s[len(s) // 2], mean=sum(s) / len(s), p99=s[int(0.99 * (len(s) - 1))], max=s[-1])


def main(gpu="L4"):
    for model in ("qwen3_8b", "gemma4_e4b"):
        path = os.path.join(HERE, "results", f"fused_tail_{model}_{gpu}.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        oldp = os.path.join(HERE, "results", f"fused_{model}_{gpu}.json")
        old = json.load(open(oldp)) if os.path.exists(oldp) else None
        print(f"== {model} on {r['gpu']}, torch {r['torch']}, triton {r['triton']}, {r['reps']} repeats per batch ==")
        for rn, rr in r["rules"].items():
            for B, c in rr.items():
                per = {k: [med(x) for x in v] for k, v in c["ms"].items()}          # median of the repeats
                bg, fg = stats(per["base_graph"]), stats(per["fused_graph"])
                n = FIRST[B]
                rep = {k: med([x[0] for x in c["ms"][k][:n]]) for k in ("base_graph", "fused_graph")}
                mx = [max(x) for x in c["int8_rows"]]
                allrows = sorted(y for x in c["int8_rows"] for y in x)
                lose_med = sum(t > bg["median"] for t in per["fused_graph"])
                lose_own = sum(t > b for t, b in zip(per["fused_graph"], per["base_graph"]))
                worst = max(range(len(mx)), key=lambda i: per["fused_graph"][i])
                print(f"\n{rn}, batch {B}: {c['batches']} batches, {c['states']} states")
                print(f"  controls: mismatches eager {c['mismatches']}, graphed {c['mismatches_graph']}, "
                      f"BF16 head {c['mismatches_bf16_baseline']} (over {r['reps']} repeats); "
                      f"row counts differing across repeats: {c['rows_differ_across_reps']}")
                print("  ms              median    mean     p99     max")
                for name, s in (("BF16 graphed", bg), ("fused graphed", fg), ("BF16 eager", stats(per["base"])),
                                ("fused eager", stats(per["fused"]))):
                    print(f"  {name:14s} {s['median']:7.2f} {s['mean']:7.2f} {s['p99']:7.2f} {s['max']:7.2f}")
                print(f"  fused / BF16 (graphed): median {fg['median'] / bg['median']:.2f}, mean {fg['mean'] / bg['mean']:.2f}, "
                      f"p99 over BF16 median {fg['p99'] / bg['median']:.2f}, max over BF16 median {fg['max'] / bg['median']:.2f}")
                print(f"  batches slower than the BF16 graphed median: {lose_med} of {c['batches']}; "
                      f"slower than the same batch's BF16 graphed time: {lose_own}")
                if old is not None:
                    was = old["rules"][rn][B]["ms"]
                    print(f"  reproduction (first repeat, first {n} batches): BF16 graphed {rep['base_graph']:.2f} "
                          f"(first run {was['base_graph']:.2f}, x{rep['base_graph'] / was['base_graph']:.3f}); fused graphed "
                          f"{rep['fused_graph']:.2f} (first run {was['fused_graph']:.2f}, x{rep['fused_graph'] / was['fused_graph']:.3f})")
                print(f"  INT8 rows per state: median {med(allrows)}, p99 {allrows[int(0.99 * (len(allrows) - 1))]}, "
                      f"max {allrows[-1]}; exact rows max {max(y for x in c['exact_rows'] for y in x)}")
                print(f"  slowest batch: {per['fused_graph'][worst]:.2f} ms, its longest INT8 shortlist {mx[worst]} rows")
                print("  fused graphed ms by the longest INT8 shortlist in the batch:")
                for lo, hi in BINS:
                    t = [per["fused_graph"][i] for i in range(len(mx)) if lo <= mx[i] < hi]
                    if t:
                        label = f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
                        print(f"    {label:>12s} rows: {len(t):4d} batches, median {med(t):6.2f}, max {max(t):6.2f}")
        print()


if __name__ == "__main__":
    main(*sys.argv[1:2])
