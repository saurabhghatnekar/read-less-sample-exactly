"""E1 table. Usage: python3 -I e2e_summarize.py  (reads logs/e2e_*.json)"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda n: json.load(open(os.path.join(HERE, "logs", f"e2e_{n}.json")))
f3 = lambda d: f"{d['median']:.2f}, {d['p99']:.2f}, {d['mean']:.2f}"


def main():
    q, g, ss, sh = L("qwen3_8b"), L("gemma4_e4b"), L("sarvam_30b_stock"), L("sarvam_30b_head")
    print("# E1: the head's share of a greedy decode step (A100-80GB)\n")
    print(f"GPUs: Qwen3-8B {q['gpu']}, Gemma 4 E4B {g['gpu']}, Sarvam-30B stock loop {ss['gpu']} (torch {ss['torch']}, "
          f"transformers {ss['transformers']}), Sarvam-30B head {sh['gpu']} (torch {sh['torch']}).")
    print(f"T1 token mismatches against the FP32 reference: Qwen {q['B1']['mismatches'] + q['B16']['mismatches']}, "
          f"Gemma {g['B1']['mismatches'] + g['B16']['mismatches']}, Sarvam (head container) "
          f"{sh['B1']['mismatches'] + sh['B16']['mismatches']}; fallbacks "
          f"{sum(x[b]['fallbacks'] for x in (q, g, sh) for b in ('B1', 'B16'))}.")
    print(f"T2 Sarvam BF16 head in container (b) over the stock head in container (a), medians: "
          f"{sh['B1']['bf16_head_ms']['median'] / ss['B1']['head_ms']['median']:.3f} (batch 1), "
          f"{sh['B16']['bf16_head_ms']['median'] / ss['B16']['head_ms']['median']:.3f} (batch 16).")
    print(f"T3 loop tokens equal generate(): Qwen {q['T3_loop_equals_generate']}, Gemma {g['T3_loop_equals_generate']}.\n")
    print("| Model, batch | Timed steps | Body ms (median, p99, mean) | Stock head + argmax ms (median) | "
          "Certified head ms (median, p99, mean) | Head's share of the stock step | Certified head / stock head (median) | "
          "Certified step / stock step (totals) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in (("Qwen3-8B", q), ("Gemma 4 E4B", g)):
        for B in ("B1", "B16"):
            x = r[B]
            print(f"| {name}, {B[1:]} | {x['steps']:,} | {f3(x['body_ms'])} | {x['head_ms']['median'] + x['tail_ms']['median']:.2f} | "
                  f"{f3(x['cert_ms'])} | {100 * x['head_share']['of_totals']:.1f}% | {x['cert_over_head']['median']:.2f} | "
                  f"{x['end_to_end']['of_totals']:.4f} |")
    for B in ("B1", "B16"):
        x, y = ss[B], sh[B]
        stock, body = x["stock_ms"]["mean"], x["body_ms"]["mean"]
        print(f"| Sarvam-30B, {B[1:]} (two containers) | {x['steps']:,} | {f3(x['body_ms'])} | "
              f"{x['head_ms']['median'] + x['tail_ms']['median']:.2f} | {f3(y['cert_ms'])} | "
              f"{100 * x['head_share']['of_totals']:.1f}% | {y['cert_over_bf16']['median']:.2f} | "
              f"{(body + y['cert_ms']['mean']) / stock:.4f} |")
    print("\nCertified tokens that differ from the stock BF16 token (not errors: the stock head's ties): "
          f"Qwen {q['B1']['tokens_differing_from_stock']} of {q['B1']['steps']} (batch 1), "
          f"{q['B16']['tokens_differing_from_stock']} of {16 * q['B16']['steps']} (batch 16); "
          f"Gemma {g['B1']['tokens_differing_from_stock']} of {g['B1']['steps']}, "
          f"{g['B16']['tokens_differing_from_stock']} of {16 * g['B16']['steps']}.")
    for name, r, V, D in (("Qwen3-8B", q, q["V"], q["D"]), ("Gemma 4 E4B", g, g["V"], g["D"])):
        h = r["B1"]["head_ms"]["median"]
        print(f"{name}: the stock BF16 head moves {2 * V * D / 1e9:.2f} GB in {h:.2f} ms, {2 * V * D / h / 1e6:,.0f} GB/s; "
              f"context {r['B1']['ctx']['min']:.0f} to {r['B1']['ctx']['max']:.0f} tokens.")


if __name__ == "__main__":
    main()
