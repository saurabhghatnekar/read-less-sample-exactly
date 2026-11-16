"""CPU check for N1 (norm_rule.json, control N0_cpu). Run: python3 -I test_norm_probe.py

1. A small random Llama model: the captured sites equal output_hidden_states where both exist, the norm's output
   equals the head's input, and the decode loop with the cache gives the tokens of a pass without it.
2. Synthetic states: a direction planted in the model gives "in_the_model"; a direction made by the gain vector
   gives "norm_makes_it"; a norm that applies 1 + weight is caught and its gain fitted.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import norm_lib  # noqa: E402

torch.manual_seed(0)
fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name)


# ---------------------------------------------------------------- 1. a small random Llama
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

cfg = LlamaConfig(vocab_size=300, hidden_size=256, intermediate_size=512, num_hidden_layers=6, num_attention_heads=4,
                  max_position_embeddings=128, eos_token_id=299, bos_token_id=1, pad_token_id=0)
m = LlamaForCausalLM(cfg).eval()
with torch.no_grad():
    m.model.norm.weight.copy_(torch.rand(256) + 0.5)
lname, layers, nname, norm, head = norm_lib.find_sites(m)
check(f"sites found: {lname} ({len(layers)}), {nname}", lname == "model.layers" and len(layers) == 6 and nname == "model.norm")
prompts = [dict(id=i, group="ab"[i % 2], text="") for i in range(8)]
ids_all = torch.randint(2, 298, (8, 7))


def encode(batch):
    ids = torch.stack([ids_all[p["id"]] for p in batch])
    return ids, torch.ones_like(ids)


sites, rows, ctl = norm_lib.greedy_capture(m, encode, prompts, 4, 5, {299})
check("norm output equals head input on every step", ctl["norm_out_differs_from_head_in"] == 0 and ctl["steps"] == 10)
check("last block output equals norm input", ctl["last_block_differs_from_norm_in"] == 0)
check("one row per state", len(rows) == sites["post_norm"].shape[0] == sites["block_1"].shape[0])
with torch.no_grad():
    full = m(input_ids=ids_all[:4], output_hidden_states=True)
step0 = [i for i, r in enumerate(rows) if r["step"] == 0 and r["pid"] < 4]
hs = full.hidden_states                                         # embeddings, blocks 1..5, then the normed output
ok = all(torch.allclose(sites[f"block_{i}"][step0], hs[i][:, -1], atol=1e-5) for i in range(1, 6))
check("blocks 1..5 at step 0 equal output_hidden_states", ok)
check("post-norm at step 0 equals the last hidden state", torch.allclose(sites["post_norm"][step0], hs[-1][:, -1], atol=1e-5))
# the cached loop's tokens against a pass with no cache
seq = ids_all[:4].clone()
want = []
with torch.no_grad():
    for t in range(5):
        nxt = m(input_ids=seq).logits[:, -1].argmax(-1)
        want.append(nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
got = {(r["pid"], r["step"]): r["tok"] for r in rows}
alive, ok = [True] * 4, True
for t in range(5):
    for j in range(4):
        if alive[j]:
            ok &= got.get((j, t)) == int(want[t][j])
            alive[j] = int(want[t][j]) != 299
check("cached greedy tokens equal the uncached ones", ok)
res = norm_lib.analyse(sites, rows, norm, W=head.weight.detach(), dev="cpu")
check("gain undone reproduces the normalised pre-norm state",
      res["norm"]["gain_source"] == "module weight" and res["norm"]["cos_gain_free_vs_normalised_pre_median"] > 0.9999)
check("every site has its measures", set(res["sites"]) == {f"block_{i}" for i in range(1, 7)}
      | {"pre_norm", "pre_norm_over_rms", "gain_free", "post_norm"} and "groups_of_128" in res)


# ---------------------------------------------------------------- 2. synthetic states
class RMS(torch.nn.Module):
    def __init__(self, g, plus_one=False):
        super().__init__()
        self.weight = torch.nn.Parameter(g.clone())
        self.eps, self.plus_one = 1e-6, plus_one

    def forward(self, x):
        y = x / x.pow(2).mean(-1, keepdim=True).sqrt()
        return y * ((1 + self.weight) if self.plus_one else self.weight)


d, n = 256, 4000
rows2 = [dict(pid=i % 40, group="ab"[i % 2], step=0, tok=0) for i in range(n)]


def synth(x, norm):
    s = {"block_1": 0.3 * x, "block_2": 0.6 * x, "block_3": x, "pre_norm": x, "post_norm": norm(x).detach()}
    return norm_lib.analyse(s, rows2, norm, dev="cpu")


v = torch.nn.functional.normalize(torch.randn(d), dim=0)
x_model = 3.0 * v[None, :] * (1 + 0.1 * torch.randn(n, 1)) + torch.randn(n, d) / d ** 0.5      # planted in the model
r = synth(x_model, RMS(torch.ones(d)))
check(f"planted in the model -> in_the_model (ratio {r['verdict']['ratio']:.2f})", r["verdict"]["verdict"] == "in_the_model")
check("its top-1 share is high before the norm too", r["sites"]["pre_norm"]["top1"] > 0.8)

x_iso = torch.randn(n, d) + 0.25                                 # a small common offset in every coordinate
g = torch.ones(d)
g[:8] = 40.0                                                     # the gain blows up eight coordinates
r = synth(x_iso, RMS(g))
check(f"made by the gain -> norm_makes_it (gain-free {r['verdict']['gain_free_top1']:.2f}, post "
      f"{r['verdict']['post_top1']:.2f})", r["verdict"]["verdict"] == "norm_makes_it")
check("the gain is reported as concentrated", r["norm"]["gain_top5_share"] > 0.5)

r = synth(x_model, RMS(torch.rand(d) - 0.2, plus_one=True))       # y = x^ (1 + weight): the stored weight is not the gain
check("a 1 + weight norm is caught and its gain fitted",
      r["norm"]["gain_source"] == "fitted per coordinate" and r["norm"]["cos_gain_free_vs_normalised_pre_median"] > 0.9999)
check("verdict functions", norm_lib.verdict(0.3, 0.74) == "norm_makes_it" and norm_lib.verdict(0.5, 0.74) == "both"
      and norm_lib.verdict(0.6, 0.74) == "in_the_model")

print("ALL PASS" if not fails else f"{fails} FAILURES")
sys.exit(fails)
