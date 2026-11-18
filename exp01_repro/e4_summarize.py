"""Summarise logs/e4_<model>_<gpu>.json (e4_rule.json). Plain Python.

  python3 -I e4_summarize.py [gpu] [suffix] > logs/e4_summary.txt
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"qwen3_1.7b": "Qwen3-1.7B", "gemma4_e4b": "Gemma 4 E4B"}


def main(gpu="A10G", suffix=""):
    rows = []
    for model in NAMES:
        path = os.path.join(HERE, "logs", f"e4_{model}_{gpu}{suffix}.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        print(f"== {NAMES[model]} on {r['gpu']}: cache {r['cache']}; torch {r['torch']}, transformers {r['transformers']}")
        print(f"controls: head weights equal {r['control_head_weights_equal']}; state max abs diff "
              f"{r['control_state_max_abs_diff']}; soft-cap switched off {r['softcap_switched_off'] or 'none'}; "
              f"head replaced at {r['head_replaced']}; errors {r['errors'] or 'none'}")
        print(f"GPU memory: {r['gpu_mem_gb_after_load']:.1f} GB after load, {r['gpu_mem_gb_peak']:.1f} GB peak")
        for B, b in r["batches"].items():
            s, h, c, hc = b["step_ms"], b["head_ms"], b["check"], b["head_check"]
            st = s["stock"]["median"]
            mean2 = (st + s["stock_again"]["median"]) / 2
            rp = sorted(b["ratio_per_batch_all"])
            print(f"\nbatch {B}: {b['n_batches']} prompt batches, prompt tokens median {b['prompt_tokens']['median']}")
            print(f"  step ms (median; p99): stock {st:.3f}; {s['stock']['p99']:.3f} | certified {s['certified']['median']:.3f}; "
                  f"{s['certified']['p99']:.3f} | stock again {s['stock_again']['median']:.3f}; {s['stock_again']['p99']:.3f}")
            print(f"  head alone ms (median): stock {h['stock']['median']:.3f}, certified {h['certified']['median']:.3f}, "
                  f"ratio {b['head_ratio']:.3f}; INT8 rows per step median {hc['int8_rows_per_step']['median']}, "
                  f"p99 {hc['int8_rows_per_step']['p99']}")
            print(f"  head share {100 * b['head_share']:.1f}% | estimate by substitution {b['estimate_by_substitution']:.4f} | "
                  f"in loop {b['in_loop_ratio']:.4f} | against the mean of the two stock runs "
                  f"{s['certified']['median'] / mean2:.4f} | stock drift {b['stock_drift']:.4f}")
            print(f"  per prompt batch, certified / stock: min {rp[0]:.4f}, median {rp[len(rp) // 2]:.4f}, max {rp[-1]:.4f}")
            print(f"  check: {c['tokens']} tokens, {c['mismatches_vs_reference']} differ from the reference; "
                  f"{c['rows_identical_to_stock']} of {c['rows']} continuations identical to the stock model's | head alone: "
                  f"{hc['states']} states, certified differs on {hc['cert_mismatches']}, stock BF16 head on {hc['stock_mismatches']}")
            rows.append((NAMES[model], B, st, s["certified"]["median"], s["stock_again"]["median"], h["stock"]["median"],
                         h["certified"]["median"], b["head_share"], b["estimate_by_substitution"], b["in_loop_ratio"],
                         s["certified"]["median"] / mean2, b["stock_drift"]))
        print()
    print("| Model, batch | Stock step | Certified step | Stock again | Stock head | Certified head | Head's share | "
          "Estimate | In loop | In loop, vs mean of stock runs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, B, st, ce, sa, hs, hc, sh, es, il, il2, dr in rows:
        print(f"| {n}, {B} | {st:.2f} ms | {ce:.2f} ms | {sa:.2f} ms | {hs:.3f} ms | {hc:.3f} ms | {100 * sh:.1f}% | {es:.3f} | "
              f"{il:.3f} | {il2:.3f} |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
