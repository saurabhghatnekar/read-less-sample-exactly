"""CPU self-tests for A3. Run: python3 -I test_a3.py [--summarize <scratch dir>]"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib  # noqa: E402
import cert_core as cc  # noqa: E402

torch.manual_seed(0)


def fake_head(V=4096, d=512):
    W = torch.randn(V, d) * 0.03 + 0.03 * torch.randn(d)   # large shared mean row, like Sarvam
    W[:, :4] *= 8
    return W.bfloat16().float()


def fake_states(B, d, mu=None):
    h = torch.randn(B, d) * 2
    h[:, 7] *= 30
    if mu is not None:   # half the states lean along the mean row, so h.mu is large there (as on Sarvam)
        h[::2] += 400 * (mu / mu.norm()).float()
    return h.bfloat16().float()


RULES_G = {"greedy": dict(rule="greedy")}
RULES_T = {"gumbel_0.6": dict(rule="gumbel", T=0.6), "gumbel_1.0": dict(rule="gumbel", T=1.0),
           "top50": dict(rule="topk", k=50, T=1.0), "minp_0.05": dict(rule="minp", p=0.05, T=1.0)}


def exp01_reference(W, h, rows, rules):
    """The exp01 analyze_chunk computation for the certificates A3 compares at t = 0."""
    V, d = W.shape
    w_norm = W.norm(dim=1)
    z_ref = h @ W.T
    keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64)
    g = cc.gumbel(keys, V, "cpu")
    i8 = cc.prepare(cc.make_int8_row(W))
    b8all, _ = cc.bounds_for(i8, h, w_norm)
    z8, b8 = cc.approx_logits(i8, h), torch.minimum(b8all["wc"], b8all["res"])
    out = {}
    for v, bn in ((cc.prepare(cc.make_int4_dither(W)), "hoeff"), (cc.prepare(cc.make_int4_g(W)), "res")):
        zt = cc.approx_logits(v, h)
        b = cc.bounds_for(v, h, w_norm)[0][bn]
        for rn, rc in rules.items():
            for st in ([None, (z8, b8)] if bn == "hoeff" else [(z8, b8)]):
                S, R, _ = cc.certify(z_ref, zt, b, rc, g, st)
                key = f"{v.name}/{bn}/{'casc' if st else 'single'}/{rn}"
                Sc = None if S is None else S.sum(1).float()
                Rc = R.sum(1).float()
                out[f"{key}/S"] = Sc if Sc is not None else torch.zeros_like(Rc)
                out[f"{key}/R"] = Rc
                out[f"{key}/bytes"] = cc.bytes_fraction(V, d, v.bytes_per_weight, 0.0 if bn == "hoeff" else 2.0,
                                                        d // v.G, Sc, Rc, d + 4)
    return out


def main(scratch=None):
    V, d, B = 4096, 512, 48
    W = fake_head(V, d)
    mu64 = a3_lib.mean_row(W)
    print(f"fake head: ||mu|| / median row = {mu64.norm() / W.norm(dim=1).median():.2f}")
    saved = {}
    for cap in (None, 30.0):
        for mode, rules in (("greedy", RULES_G), ("t1", RULES_T)):
            rules = {k: dict(v, cap=cap) for k, v in rules.items()}
            h = fake_states(B, d, mu64)
            rows = [dict(pid=i // 8, group=["english", "hindi"][i % 2], step=i % 8, mode=mode) for i in range(B)]
            rec, st = a3_lib.run_chunk(W, mu64, h, rows, rules, controls=True)
            c2 = sum(st["c2_viol"].values())
            c3 = {k: v for k, v in st["c3_mismatch"].items() if v and "/hoeff/" not in k}
            hoeff_mm = sum(v for k, v in st["c3_mismatch"].items() if "/hoeff/" in k)
            assert c2 == 0, st["c2_viol"]
            assert not c3, c3
            assert st["c6_eligible"] > 0 and st["c6_eligible_missed"] == 0, st
            se = {t: st["dither_se"][t] / st["dither_n"][t] for t in st["dither_se"]}
            ref = exp01_reference(W, h, rows, rules)
            diff = sum(int((rec["t0/" + k] != x).sum()) for k, x in ref.items())
            assert diff == 0, f"t = 0 differs from the exp01 path on {diff} values"
            hb = {t: float(rec[f"t{t:g}/int4d_g128/hoeff/meanbound"].mean()) for t in a3_lib.GRID}
            print(f"cap={cap} {mode}: C2 0, C3 0 (dither mismatches {hoeff_mm}), C6 caught on "
                  f"{st['c6_shift_states']}/{B} (eligible {st['c6_eligible']}, missed 0), T0 identical, dither SE {min(se.values()):.3f}-{max(se.values()):.3f}, "
                  f"Hoeffding mean bound t=1/t=0 {hb[1.0] / hb[0.0]:.3f}, t=4/t=0 {hb[4.0] / hb[0.0]:.3f}")
            if cap is None:
                saved[mode] = (rec, ref, rows, st)
    if scratch:   # write fake result dirs so a3_summarize.py can be exercised end to end
        for sub, which in (("a3", 0), ("exp01", 1)):
            os.makedirs(f"{scratch}/{sub}", exist_ok=True)
            for mode, (rec, ref, rows, st) in saved.items():
                arrs = rec if which == 0 else ref
                np.savez(f"{scratch}/{sub}/{mode}_000000.npz", **{k: v.numpy() for k, v in arrs.items()})
                json.dump(dict(stats=st, rows=[(r["pid"], r["group"], r["step"]) for r in rows]),
                          open(f"{scratch}/{sub}/{mode}_000000.json", "w"))
    print("all A3 self-tests passed")


if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--summarize" else None)
