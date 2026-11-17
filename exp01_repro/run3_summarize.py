"""Tables for T1 (t1_rule.json), R1 (r1_rule.json) and G1 (g1_rule.json). Plain Python.

  python3 -I run3_summarize.py        # writes logs/t1_summary.txt, logs/r1_summary.txt, logs/g1_summary.txt
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, "logs")


def b3(x):
    return f"{x['median']:.4f}, {x['p99']:.4f}, {x['mean']:.4f}"


def t1(out):
    r = json.load(open(os.path.join(L, "t1_summary.json")))
    print("# T1: a dithered INT8 stage, a leaner top-k rule, top-p. Batch 128 (the run whose control is clean).", file=out)
    for model in ("qwen3_8b", "gemma4_e4b", "sarvam_30b"):
        m = r[model]
        print(f"\n## {model}: d8 bound / i8 bound, median: {m['bound_ratio']}", file=out)
        for mode, c in m["controls"].items():
            print(f"- {mode}: {c['states']} states; values differing from the stored run: {c['t0_diff']}; reference logits "
                  f"outside an interval: {sum(c['violations'].values())}; token mismatches: {sum(c['mismatches'].values())}", file=out)
        print("\n| Rule | Arm | Bytes (median, p99, mean) | BF16 rows (median, p99) | Rows per middle stage, p99 | Status |", file=out)
        print("|---|---|---|---|---|---|", file=out)
        for rn, tab in m["table"].items():
            for a, x in tab.items():
                st = ", ".join(f"{k[2:]} {v['p99']:.0f}" for k, v in x["stages"].items())
                print(f"| {rn} | {a} | {b3(x['bytes'])} | {x['R']['median']:.0f}, {x['R']['p99']:.0f} | {st} | {x['status']} |", file=out)
        if model == "sarvam_30b":
            print("\nTop-50, mean bytes by language:", file=out)
            for a, x in m["table"]["top50"].items():
                print(f"  {a}: " + ", ".join(f"{g} {v:.4f}" for g, v in x["mean_by_language"].items()), file=out)
    p = os.path.join(L, "t1_summary_batch64.json")
    if os.path.exists(p):
        o = json.load(open(p))
        d = max(abs(r[m]["table"][rn][a]["bytes"]["mean"] - o[m]["table"][rn][a]["bytes"]["mean"])
                for m in ("qwen3_8b", "gemma4_e4b", "sarvam_30b") for rn in r[m]["table"] for a in r[m]["table"][rn])
        bad = {m: {mode: c["t0_diff"] for mode, c in o[m]["controls"].items()} for m in ("qwen3_8b", "gemma4_e4b", "sarvam_30b")}
        print(f"\nFirst run, batch 64: values differing from the stored run {bad}; largest change of any mean between "
              f"the two runs {d:.6f}", file=out)


def r1(out):
    r = json.load(open(os.path.join(L, "r1_summary.json")))
    print("# R1: Sarvam-30B on a second prompt set and with four dither seeds", file=out)
    for key in sorted(k for k in r if k != "rule_sha"):
        m = r[key]
        print(f"\n## {key}: median bound with K2 / none {m['bound_k2_vs_none']:.3f}", file=out)
        for mode, c in m["controls"].items():
            print(f"- {mode}: {c['states']} states; violations {c['violations']}; mismatches {c['mismatches']}; "
                  f"jobs checked against M2 {c['r0_checked']}, values differing {c['r0_diff']}", file=out)
        print("\n| Rule | Arm | Bytes (median, p99, mean) | BF16 rows (median, p99) | Status |", file=out)
        print("|---|---|---|---|---|", file=out)
        for rn, tab in m["table"].items():
            for a, x in tab.items():
                print(f"| {rn} | {a} | {b3(x['bytes'])} | {x['R']['median']:.0f}, {x['R']['p99']:.0f} | {x['status']} |", file=out)
        g = m["table"]["greedy"]["k2"]
        print("\nGreedy with K2, p99 bytes by language: " + ", ".join(f"{k} {v:.4f}" for k, v in g["p99_by_language"].items()), file=out)


def g1(out):
    print("# G1: the method's parts on Sarvam-1, SmolLM3-3B and Phi-4-mini", file=out)
    for model in ("sarvam_1", "smollm3_3b", "phi4_mini"):
        p = os.path.join(L, f"g1_{model}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        s = r["selection"]
        print(f"\n## {model}: V {r['V']}, d {r['d']}", file=out)
        for mode, c in r["controls"].items():
            print(f"- {mode}: {c['states']} states ({c['odd_states']} from odd prompts); violations {c['violations']}; "
                  f"mismatches {c['mismatches']}", file=out)
        print(f"- selected on the even prompts (mean greedy bytes): shift {s['t_star']:g} from "
              f"{ {k: round(v, 4) for k, v in s['shift'].items()} }; K = {s['k_star']} from "
              f"{ {k: round(v, 4) for k, v in s['k'].items()} }; groups {s['groups_star']} from "
              f"{ {k: round(v, 4) for k, v in s['groups'].items()} }", file=out)
        print(f"- state energy in the top 1, 4, 16 directions: {s['top1_share']:.2f}, {s['top4_share']:.2f}, {s['top16_share']:.2f}; "
              f"largest group's share of the variance term {s['group_shares']['largest'][1]:.3f} (uniform {s['group_shares']['uniform']:.3f}); "
              f"median bound vs plain: shift {r['bound_vs_plain']['shift']:.3f}, with directions {r['bound_vs_plain']['k']:.3f}; "
              f"d8 / i8 bound {r['d8_vs_i8_bound']:.3f}", file=out)
        print("\n| Rule (odd prompts) | Arm | Bytes (median, p99, mean) | BF16 rows (median, p99) | Status |", file=out)
        print("|---|---|---|---|---|", file=out)
        for rn, tab in r["table"].items():
            for a, x in tab.items():
                print(f"| {rn} | {a} | {b3(x['bytes'])} | {x['R']['median']:.0f}, {x['R']['p99']:.0f} | {x['status']} |", file=out)


if __name__ == "__main__":
    for name, fn in (("t1", t1), ("r1", r1), ("g1", g1)):
        with open(os.path.join(L, f"{name}_summary.txt"), "w") as f:
            fn(f)
    print("wrote logs/t1_summary.txt, logs/r1_summary.txt, logs/g1_summary.txt")
