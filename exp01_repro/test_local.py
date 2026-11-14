"""CPU self-tests for cert_core (C9-style). Run: python3 -I test_local.py"""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cert_core as cc  # noqa: E402

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False


def fake_head(V=4096, d=512):
    W = torch.randn(V, d) * 0.03 + 0.01 * torch.randn(d)  # shared mean row
    W[:, :4] *= 8                                           # a few outlier dims
    return W.bfloat16().float()


def fake_states(B, d):
    h = torch.randn(B, d) * 2
    h[:, 7] *= 30                                           # concentrated h, like real final states
    return h.bfloat16().float()


def plain_topk_membership(L, U, k):
    V = len(L)
    cin, cout = [], []
    for i in range(V):
        cin.append(sum(1 for j in range(V) if j != i and U[j] > L[i]) < k)
        cout.append(sum(1 for j in range(V) if L[j] > U[i]) >= k)
    return torch.tensor(cin), torch.tensor(cout)


def main():
    V, d, B = 4096, 512, 64
    W = fake_head(V, d)
    h = fake_states(B, d)
    z_ref = h @ W.T
    w_norm = W.norm(dim=1)

    variants = [cc.prepare(f(W)) for f in (cc.make_int8_row, cc.make_int4_g, cc.make_int4_dither,
                                           cc.make_int4_rot, cc.make_nvfp4)]
    int8 = variants[0]
    z8 = cc.approx_logits(int8, h)
    b8all, _ = cc.bounds_for(int8, h, w_norm)
    b8 = torch.minimum(b8all["wc"], b8all["res"])

    keys = torch.arange(B, dtype=torch.int64) * 977
    g = cc.gumbel(keys, V, "cpu")
    assert torch.equal(g, cc.gumbel(keys, V, "cpu")), "gumbel not deterministic"

    rules = [dict(rule="greedy", cap=None), dict(rule="gumbel", T=1.0, cap=None),
             dict(rule="gumbel", T=0.6, cap=30.0), dict(rule="topk", k=50, T=1.0, cap=None),
             dict(rule="minp", p=0.05, T=1.0, cap=None), dict(rule="topk", k=50, T=1.0, cap=30.0)]

    fails = 0
    for v in variants:
        zt = cc.approx_logits(v, h)
        bounds, slack = cc.bounds_for(v, h, w_norm)
        err = (z_ref - zt).abs()
        bounds["oracle"] = err + slack
        for bn, b in bounds.items():
            viol = (err > b).sum().item()
            if bn in ("wc", "res") and viol:
                print(f"FAIL soundness {v.name}/{bn}: {viol} violations"); fails += 1
            for cfg in rules:
                for stage2 in (None, (z8, b8)):
                    S, R, tok = cc.certify(z_ref, zt, b, cfg, g, stage2)
                    ref_tok = cc.decide(z_ref, cfg, g)
                    bad = (tok != ref_tok).sum().item()
                    if bad and not (bn == "hoeff" and viol):
                        print(f"FAIL exactness {v.name}/{bn}/{cfg}/{'casc' if stage2 else 'single'}: {bad}")
                        fails += 1
            print(f"  {v.name:12s} {bn:6s} max err/bound {(err / b).max():.3f}  "
                  f"median |R| greedy {cc.certify(z_ref, zt, b, rules[0], g)[1].sum(1).median().item()}")

    # dither: normalized squared error ~ 1 (uniform error variance s^2/12 per weight)
    v = variants[2]
    e = (cc.approx_logits(v, h) - z_ref)
    hg = h.view(B, d // v.G, v.G)
    var = (hg.norm(dim=-1) ** 2) @ (v.scale ** 2).T / 12
    nmse = (e ** 2 / var).mean().item()
    print(f"dither normalized MSE {nmse:.3f} (expect ~1)")
    if not 0.9 < nmse < 1.1:
        print("FAIL dither nmse"); fails += 1

    # C9: vectorized top-k membership vs plain definition on a small case
    Vs = 200
    Ls = torch.randn(3, Vs)
    Us = Ls + torch.rand(3, Vs) * 0.5
    cin, cout = cc.topk_membership(Ls, Us, 10)
    for bi in range(3):
        pi, po = plain_topk_membership(Ls[bi].tolist(), Us[bi].tolist(), 10)
        if not (torch.equal(cin[bi], pi) and torch.equal(cout[bi], po)):
            print("FAIL topk membership vs plain"); fails += 1

    # byte fractions
    print("first-pass bytes: int4 g128", (0.5 + 2 / 128) / 2, " nvfp4", (0.5 + 1 / 16) / 2,
          " int8 row", (1 + 4 / d) / 2)
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    return fails


if __name__ == "__main__":
    sys.exit(main())
