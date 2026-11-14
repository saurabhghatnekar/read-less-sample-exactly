"""CPU check for F1. Run: python3 -I test_flip.py"""
import math, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_lib, cert_core as cc, flip_lib  # noqa: E402
from test_a3 import fake_head, fake_states, RULES_G, RULES_T  # noqa: E402

torch.manual_seed(0)
V, d, B = 4096, 512, 64
W = fake_head(V, d); mu = a3_lib.mean_row(W)
w_norm = W.norm(dim=1)


def exact_copies(t=0.0):
    vs = a3_lib.build(W, mu, t)
    for v in vs.values():
        v.Wq = a3_lib.shifted(W, mu, t).clone()
    return vs


for cap in (None, 30.0):
    for mode, rules in (("greedy", RULES_G), ("t1", RULES_T)):
        rules = {k: dict(x, cap=cap) for k, x in rules.items()}
        h = fake_states(B, d, mu)
        rows = [dict(pid=i // 8, group="english", step=i % 8, mode=mode) for i in range(B)]
        keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64)
        g = cc.gumbel(keys, V, "cpu")
        z_ref = h @ W.T
        # 1. an exact copy never flips (t = 0: bitwise the reference)
        rec = flip_lib.run_chunk(W, mu, exact_copies(0.0), h, rows, rules)
        for k, x in rec.items():
            if k.endswith(("/flip", "/outside")):
                assert not x.any(), k
            if k.endswith(("/gap", "/setdiff", "/maxerr")):
                assert float(x.abs().max()) == 0, k
        for t in (0.0, 0.5, 1.5):
            vs = a3_lib.build(W, mu, t)
            rec = flip_lib.run_chunk(W, mu, vs, h, rows, rules)
            tag = f"t{t:g}"
            for vn in flip_lib.COPIES:
                # 2. logits and largest error are a3_lib's, bit for bit
                zt_a3, _, _ = a3_lib.intervals(vs[vn], h, w_norm, mu)
                zt = flip_lib.approx(vs[vn], h, mu)
                assert torch.equal(zt, zt_a3), (t, vn)
                assert torch.equal(rec[f"{tag}/{vn}/maxerr"], (z_ref - zt_a3).abs().amax(1))
                for rn, cfg in rules.items():
                    key = f"{tag}/{vn}/{rn}"
                    # 3. flips match a plain per-state loop
                    for i in range(6):                              # the loop is slow; six states per cell
                        def pick(zrow):
                            # scores come from torch (same float32 arithmetic); the selection is plain Python
                            ct = cc.softcap(zrow, cap)
                            c = ct.tolist()
                            if cfg["rule"] == "greedy":
                                s, ok = c, [True] * V
                            else:
                                s = (ct / cfg["T"] + g[i]).tolist()
                                if cfg["rule"] == "gumbel":
                                    ok = [True] * V
                                elif cfg["rule"] == "topk":
                                    kth = sorted(c, reverse=True)[cfg["k"] - 1]
                                    ok = [x >= kth for x in c]
                                else:
                                    thr = (ct.max() + cfg["T"] * math.log(cfg["p"])).item()
                                    ok = [x >= thr for x in c]
                            best = max(x for x, o in zip(s, ok) if o)
                            return next(j for j in range(V) if ok[j] and s[j] == best), ok
                        if True:
                            a, ok_ref = pick(z_ref[i])
                            b, ok_new = pick(zt[i])
                            assert int(rec[f"ref/{rn}"][i]) == a, (key, i)
                            assert bool(rec[f"{key}/flip"][i]) == (a != b), (key, i)
                            if cfg["rule"] in ("topk", "minp"):
                                assert int(rec[f"{key}/setdiff"][i]) == sum(x != y for x, y in zip(ok_ref, ok_new))
                                assert bool(rec[f"{key}/outside"][i]) == (not ok_ref[b]), (key, i)
                    # 4. gap is positive exactly on flipped states (greedy and Gumbel)
                    if f"{key}/gap" in rec:
                        fl, gap = rec[f"{key}/flip"], rec[f"{key}/gap"]
                        assert (gap[fl] > 0).all() and (gap[~fl] == 0).all(), key
        # 5. planted fault: rows rolled by one flip nearly every state
        vs = a3_lib.build(W, mu, 0.0)
        for v in vs.values():
            v.Wq = v.Wq.roll(1, 0)
        rec = flip_lib.run_chunk(W, mu, vs, h, rows, rules)
        # with the soft-cap, the synthetic states that lean on the mean row saturate every logit at the cap,
        # so nothing can flip there; the fault must show on the states whose reference maximum is unique
        top2 = cc.softcap(z_ref, cap).topk(2, dim=1).values
        um = top2[:, 0] > top2[:, 1]
        assert int(um.sum()) >= B // 4, int(um.sum())
        worst = min(float(rec[f"t0/{vn}/{rn}/flip"][um].float().mean()) for vn in flip_lib.COPIES for rn in rules)
        assert worst >= 0.9, worst
        rec = flip_lib.run_chunk(W, mu, a3_lib.build(W, mu, 0.0), h, rows, rules)
        rates = {f"{vn}/{rn}": round(float(rec[f"t0/{vn}/{rn}/flip"].float().mean()), 3)
                 for vn in flip_lib.COPIES for rn in rules}
        print(f"cap={cap} {mode}: exact copy 0 flips, rolled rows flip >= {worst:.2f}, synthetic flip rates {rates}")
print("F1 self-tests passed")

# ---- flip_addendum_bf16.json: the BF16 product as a copy
for cap in (None, 30.0):
    for mode, rules in (("greedy", RULES_G), ("t1", RULES_T)):
        rules = {k: dict(x, cap=cap) for k, x in rules.items()}
        rows = [dict(pid=i // 8, group="english", step=i % 8, mode=mode) for i in range(B)]
        keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows], dtype=torch.int64)
        g = cc.gumbel(keys, V, "cpu")
        # 1. small integers: the BF16 product is exact, so nothing flips
        Wi = torch.randint(-1, 2, (V, d)).float() * (torch.rand(V, d) < 0.02)
        hi = (torch.rand(B, d) < 0.05).float()
        rec = flip_lib.run_bf16_chunk(Wi, Wi.bfloat16(), hi, rows, rules)
        assert float(rec["bf16_prod/maxerr"].max()) == 0
        assert not any(x.any() for k, x in rec.items() if k.endswith(("/flip", "/outside")))
        # 2. real-valued head: flips equal a direct comparison, gaps are positive exactly on flips
        h = fake_states(B, d, mu)
        rec = flip_lib.run_bf16_chunk(W, W.bfloat16(), h, rows, rules)
        zb = (h.bfloat16() @ W.bfloat16().T).float()
        assert torch.equal(rec["bf16_prod/argmax"], zb.argmax(1))
        for rn, cfg in rules.items():
            direct = cc.decide(zb, cfg, g) != cc.decide(h @ W.T, cfg, g)
            assert torch.equal(rec[f"bf16_prod/{rn}/flip"], direct), rn
            if f"bf16_prod/{rn}/gap" in rec:
                gap = rec[f"bf16_prod/{rn}/gap"]
                assert (gap[direct] >= 0).all() and (gap[~direct] == 0).all(), rn
        print(f"cap={cap} {mode}: exact integer product 0 flips; synthetic BF16-product flip rates "
              f"{ {rn: round(float(rec[f'bf16_prod/{rn}/flip'].float().mean()), 3) for rn in rules} }")
print("F1 addendum self-tests passed")
