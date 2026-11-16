"""Summarise logs/normprobe_<model>.json (N1, norm_rule.json). Plain Python.

  python3 -I norm_summarize.py [--smoke] > logs/normprobe_summary.txt
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
K3 = {"sarvam_30b": 0.744, "sarvam_1": 0.431}


def main(smoke):
    for name in ("sarvam_30b", "sarvam_1"):
        path = os.path.join(HERE, "logs", f"normprobe_{name}{'_smoke' if smoke else ''}.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        c, nm, v, m = r["control"], r["norm"], r["verdict"], r["meta"]
        print(f"== {name}: {r['n_states']} states from {r['prompts']} prompts ({r['n_even']} even, {r['n_odd']} odd), "
              f"d = {r['d']}, {m['gpu']}, torch {m['torch']}, transformers {m['transformers']} ==")
        print(f"blocks: {c['layers']} ({c['n_layers']}); final norm: {c['norm']} ({nm['cls']}, eps {nm['eps']:g}, "
              f"bias {nm['has_bias']})")
        print("controls:")
        print(f"  norm output differs from head input on {c['norm_out_differs_from_head_in']} of {c['steps']} steps; "
              f"last block output differs from norm input on {c['last_block_differs_from_norm_in']}")
        post = r["sites"]["post_norm"]["top1"]
        print(f"  post-norm top-1 share {post:.3f} against K3's {K3[name]:.3f} (difference {post - K3[name]:+.3f})"
              + (f"; cosine of post-norm p with K2's saved pca1 {r['cos_post_p_with_saved']:.4f}"
                 if "cos_post_p_with_saved" in r else ""))
        print(f"  gain: {nm['gain_source']}; cosine of gain-free state with pre-norm over rms, median "
              f"{nm['cos_gain_free_vs_normalised_pre_median']:.6f}, smallest {nm['cos_gain_free_vs_normalised_pre_min']:.6f}"
              f" (with the module weight: {nm['cos_with_module_weight_median']:.6f})")
        if nm.get("has_gain") or nm["gain_source"] != "module weight":
            print(f"gain vector: mean {nm['gain_mean']:.3f}, sd {nm['gain_std']:.3f}, min {nm['gain_min']:.3f}, max "
                  f"{nm['gain_max']:.3f}, negative entries {nm['gain_negative']}, five largest hold "
                  f"{100 * nm['gain_top5_share']:.1f}% of its squared norm; correlation of gain with |mean of gain-free state| "
                  f"{nm.get('corr_gain_with_abs_mean_of_gain_free', float('nan')):.3f}; cosine of post-norm mean with the gain "
                  f"{nm.get('cos_post_mean_with_gain', float('nan')):.3f}")
        print("\nsite                 top1  mean_share  unit_mean  rho_odd  cos_p_post  cos_mean_post  median_norm  p_top5  p_group")
        for k, s in r["sites"].items():
            print(f"{k:18s} {s['top1']:6.3f} {s['mean_share']:10.3f} {s['unit_mean_share']:10.3f} {s['rho_median_odd']:8.3f} "
                  f"{s['cos_p_with_post']:11.3f} {s['cos_mean_with_post']:14.3f} {s['norm_median']:12.2f} "
                  f"{s['p_top5_coordinates']:7.3f} {s.get('p_largest_group', float('nan')):8.3f}")
        print(f"\nverdict: {v['verdict']} (gain-free top-1 {v['gain_free_top1']:.3f} / post-norm {v['post_top1']:.3f} = "
              f"{v['ratio']:.2f}; mean share {v['mean_share_gain_free']:.3f} / {v['mean_share_post']:.3f})")
        print(f"  first block at half the pre-norm top-1 share: {v['first_block_at_half_of_pre_norm']}; "
              f"{v['block_at_two_thirds']} (two thirds up) has {v['top1_at_two_thirds']:.3f}; "
              f"the last block changes the share by {v['last_block_change']:+.3f}")
        if "groups_of_128" in r:
            print(f"\ngroups of 128 coordinates (odd-prompt post-norm states; uniform share would be {128 / r['d']:.3f}); "
                  f"largest over mean squared group scale {r.get('group_scale_sq_max_over_mean', float('nan')):.2f}")
            for key, per in r["groups_of_128"].items():
                print(f"  {key}: language: median largest-group share (most common largest group, how often) | top four groups")
                for g, x in per.items():
                    print(f"    {g:8s} {x['largest_group_share_median']:.3f} (group {x['most_common_largest_group']:2d}, "
                          f"{100 * x['how_often']:3.0f}%) | {x['top4_groups_share_median']:.3f}   [{x['states']} states]")
        print()


if __name__ == "__main__":
    main("--smoke" in sys.argv)
