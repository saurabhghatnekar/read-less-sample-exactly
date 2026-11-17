"""Tables for M1 (mix_rule.json) from logs/mix_summary[_smoke].json. Plain Python.

  python3 -I mix_summarize.py [--smoke] > logs/mix_summary.txt
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("none", "k2", "int8g", "both", "k2_g26dir")
LANGS = ("bengali", "code", "english", "hindi", "tamil", "telugu", "all")


def main(smoke):
    r = json.load(open(os.path.join(HERE, "logs", f"mix_summary{'_smoke' if smoke else ''}.json")))
    p = r["prep"]
    print("# M1: one group in INT8 for every row, with and without the K2 correction, Sarvam-30B" + (" (SMOKE)" if smoke else ""))
    print("\nControls")
    for mode, c in r["controls"].items():
        print(f"- {mode}: {c['states']} states. M0_reproduce, values differing from the K2 addendum: {c['m0_diff']}. "
              f"Bound violations by arm: {c['violations']} (largest excess {c['largest_excess']:+.4f} logits). "
              f"Token mismatches: {c['mismatches']}.")
    print(f"- M0_group: on even-prompt Tamil and Telugu greedy states the most common largest group is "
          f"{p['tamil_telugu_even_most_common_largest_group']} (the run uses {p['group']}).")
    print("\nLargest group of the variance term, even-prompt greedy states: language, most common group (how often), "
          "group 26 is largest on, median largest share")
    for g, x in p["by_language"].items():
        print(f"  {g:8s} group {x['most_common_largest_group']:2d} ({100 * x['how_often']:3.0f}%), group 26 on "
              f"{100 * x['group26_is_largest']:3.0f}%, share {x['largest_share_median']:.3f}")
    print("\nDirection inside group 26, fitted without the language: share of the group's remaining energy along it, "
          "share of the direction inside the group")
    for g, x in p["direction"].items():
        print(f"  {g:8s} {x['share_of_group_energy']:.3f}, {x['energy_inside_group']:.3f}")
    cs = list(p["direction_cosines"].values())
    print(f"  cosines between the six left-out fits: {min(cs):.3f} to {max(cs):.3f}")
    for rn, tab in r["table"].items():
        print(f"\n## {rn}: bytes as a fraction of the BF16 head (median, p99, mean)\n")
        print("| Language | States | none | k2 | int8g | both | k2_g26dir |")
        print("|---|---|---|---|---|---|---|")
        for g in LANGS:
            if g not in tab:
                continue
            x = tab[g]
            print(f"| {g} | {x['states']} | " + " | ".join(
                f"{x[a]['bytes']['median']:.4f}, {x[a]['bytes']['p99']:.4f}, {x[a]['bytes']['mean']:.4f}" for a in ARMS) + " |")
        print("\n| Language | Median bound vs none: k2 | int8g | both | k2_g26dir | INT8-stage rows p99: none | k2 | int8g | both | k2_g26dir |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for g in LANGS:
            if g not in tab:
                continue
            x = tab[g]
            print(f"| {g} | " + " | ".join(f"{x[a]['bound_vs_none']:.3f}" for a in ARMS[1:]) + " | "
                  + " | ".join(f"{x[a]['S']['p99']:.0f}" for a in ARMS) + " |")
        print("\n| Language | both − k2, mean [95%] | states where both is lower | int8g − none, mean [95%] | k2_g26dir − k2, mean [95%] |")
        print("|---|---|---|---|---|")
        for g in LANGS:
            if g not in tab:
                continue
            x = tab[g]
            f = lambda k: "{:+.4f} [{:+.4f}, {:+.4f}]".format(*x[k]["mean_ci"])
            print(f"| {g} | {f('both_minus_k2')} | {100 * x['both_minus_k2']['share_of_states_lower']:.1f}% | "
                  f"{f('int8g_minus_none')} | {f('k2_g26dir_minus_k2')} |")
        if rn in ("greedy", "gumbel_0.6", "gumbel_1.0") and all(g in tab for g in LANGS):
            ok_tt, ok_rest = True, True
            for g in ("tamil", "telugu"):
                b = tab[g]["both"]["bytes"]
                for a in ("k2", "int8g"):
                    o = tab[g][a]["bytes"]
                    ok_tt &= b["p99"] < o["p99"] and b["mean"] < o["mean"]
            for g in ("bengali", "code", "english", "hindi"):
                b, o = tab[g]["both"]["bytes"], tab[g]["k2"]["bytes"]
                ok_rest &= b["median"] <= o["median"] + 0.001 and b["p99"] <= o["p99"] + 0.001
            print(f"\nCriterion on {rn}: 'both' beats k2 and int8g in Tamil and Telugu (p99 and mean): {ok_tt}; "
                  f"does not hurt the other four (median and p99 within 0.001 of k2): {ok_rest}; "
                  f"-> {'PASSES' if ok_tt and ok_rest else 'FAILS'}")


if __name__ == "__main__":
    main("--smoke" in sys.argv)
