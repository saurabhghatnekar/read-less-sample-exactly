"""Tables for M2 (stage_rule.json) from logs/stage_summary[_smoke].json. Plain Python.

  python3 -I stage_summarize.py [--smoke] > logs/stage_summary.txt
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ("bengali", "code", "english", "hindi", "tamil", "telugu", "all")


def main(smoke):
    r = json.load(open(os.path.join(HERE, "logs", f"stage_summary{'_smoke' if smoke else ''}.json")))
    est = {}
    p = os.path.join(HERE, "logs", "mix_staged_estimate.json")
    if os.path.exists(p) and not smoke:
        est = json.load(open(p))
    print("# M2: the heavy group read in INT8 only for the rows the rule still needs, Sarvam-30B" + (" (SMOKE)" if smoke else ""))
    print("\nControls")
    for mode, c in r["controls"].items():
        print(f"- {mode}: {c['states']} states. S0_reproduce, values differing from M1's stored k2: {c['s0_diff']}. "
              f"Reference logits outside an interval: {c['violations']}. Token mismatches: {c['mismatches']}.")
    verdict = {}
    for rn, tab in r["table"].items():
        print(f"\n## {rn}: bytes as a fraction of the BF16 head (median, p99, mean)\n")
        print("| Language | States | k2 | both (M1) | staged | staged, 138 bytes a row | staged − k2, mean [95%] | states lower / higher | largest increase |")
        print("|---|---|---|---|---|---|---|---|---|")
        for g in LANGS:
            if g not in tab:
                continue
            x = tab[g]
            f = lambda b: f"{b['median']:.4f}, {b['p99']:.4f}, {b['mean']:.4f}"
            d = x["staged_minus_k2"]
            print(f"| {g} | {x['states']} | {f(x['k2']['bytes'])} | {f(x['both']['bytes'])} | {f(x['staged']['bytes'])} | "
                  f"{f(x['staged']['bytes_kept_sum'])} | {d['mean_ci'][0]:+.4f} [{d['mean_ci'][1]:+.4f}, {d['mean_ci'][2]:+.4f}] | "
                  f"{100 * d['share_lower']:.0f}% / {100 * d['share_higher']:.0f}% | {d['largest_increase']:+.4f} |")
        print("\n| Language | Stage-G rows (median, p99, mean) | INT8 rows p99: k2 | staged | BF16 rows p99: k2 | staged | Status: k2 | staged |")
        print("|---|---|---|---|---|---|---|---|")
        for g in LANGS:
            if g not in tab:
                continue
            x = tab[g]
            sg = x["staged"]["Sg"]
            print(f"| {g} | {sg['median']:.0f}, {sg['p99']:.0f}, {sg['mean']:.0f} | {x['k2']['S']['p99']:.0f} | "
                  f"{x['staged']['S']['p99']:.0f} | {x['k2']['R']['p99']:.0f} | {x['staged']['R']['p99']:.0f} | "
                  f"{x['k2']['status']} | {x['staged']['status']} |")
        if est and all(g in tab for g in LANGS):
            print("\nAgainst M1's estimate (130 bytes a row), mean bytes: language estimate -> measured (204 bytes a row)")
            print("  " + "; ".join(f"{g} {est[f'{rn}/{g}']['staged'][2]:.4f} -> {tab[g]['staged']['bytes']['mean']:.4f}" for g in LANGS))
        if not all(g in tab for g in LANGS):
            continue
        if rn in ("greedy", "gumbel_0.6", "gumbel_1.0"):
            ok = all(tab[g]["staged"]["bytes"][k] <= tab[g]["k2"]["bytes"][k] + 0.0005
                     for g in LANGS[:-1] for k in ("median", "p99"))
            verdict[rn] = ok
            print(f"\nCriterion (a) on {rn}: median and p99 within 0.0005 of k2 in every language: {ok}")
        if rn == "top50":
            tt = all(tab[g]["staged_minus_k2"]["mean_ci"][0] <= -0.010 and tab[g]["staged_minus_k2"]["mean_ci"][2] < 0
                     for g in ("tamil", "telugu"))
            rest = all(tab[g]["staged_minus_k2"]["mean_ci"][0] <= 0.001 for g in ("bengali", "code", "english", "hindi"))
            verdict[rn] = tt and rest
            print(f"\nCriterion (b) on top50: Tamil and Telugu mean lower by at least 0.010, interval excluding zero: {tt}; "
                  f"other four not higher by more than 0.001: {rest}")
    if len(verdict) == 4:
        print(f"\nCriterion: (a) {all(verdict[k] for k in ('greedy', 'gumbel_0.6', 'gumbel_1.0'))}, (b) {verdict['top50']} "
              f"-> {'PASSES' if all(verdict.values()) else 'FAILS'}")


if __name__ == "__main__":
    main("--smoke" in sys.argv)
