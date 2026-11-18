"""E4's integration pieces (e4_rule.json), free of the GPU kernel so that test_e4.py can run them on a CPU.

The certified variant is a second model object sharing every weight with the stock one. Its output layer is an
identity and its soft-cap is off, so its forward pass returns the final hidden state where generate expects logits.
A logits processor turns that state into scores whose argmax is the token `pick` returns.
"""
import copy

import torch
from transformers import LogitsProcessor

N_SHORT, N_LONG = 8, 72


def share_params_copy(m):
    """A deep copy of the module tree whose parameters and buffers are the same tensors as m's (no new weight memory)."""
    memo = {id(p): p for p in m.parameters()}
    memo.update({id(b): b for b in m.buffers()})
    return copy.deepcopy(m, memo)


def disable_softcap(m):
    """Switch off every final_logit_softcapping reachable from the model's configs. Returns {where: old value}."""
    found, seen = {}, set()
    cands = [("config", m.config)]
    if hasattr(m.config, "get_text_config"):
        cands.append(("config.get_text_config()", m.config.get_text_config()))
    cands += [(f"{n or 'model'}.config", mod.config) for n, mod in m.named_modules() if hasattr(mod, "config")]
    for where, c in cands:
        if id(c) in seen:
            continue
        seen.add(id(c))
        if getattr(c, "final_logit_softcapping", None) is not None:
            found[where] = float(c.final_logit_softcapping)
            c.final_logit_softcapping = None
    return found


def to_identity_head(m):
    """Replace the output layer by an identity. Returns the attribute path that was replaced."""
    head = m.get_output_embeddings()
    for name, mod in m.named_modules():
        if mod is head:
            parent = m.get_submodule(name.rpartition(".")[0]) if "." in name else m
            setattr(parent, name.rpartition(".")[2], torch.nn.Identity())
            return name
    raise RuntimeError("output layer not found among the model's modules")


def head_input(m, enc):
    """The tensor the model's output layer receives at the last position, for one forward pass over enc."""
    got = []
    hk = m.get_output_embeddings().register_forward_pre_hook(lambda mod, args: got.append(args[0][:, -1].detach().clone()))
    try:
        with torch.no_grad():
            m(**enc)
    finally:
        hk.remove()
    return got[-1]


class CertProcessor(LogitsProcessor):
    """Receives the final hidden state where generate expects logits; returns scores whose argmax is pick(state)."""

    def __init__(self, pick, V):
        self.pick, self.V = pick, V
        self.record, self.states, self.tokens = False, [], []

    def __call__(self, input_ids, scores):
        t = self.pick(scores)
        o = torch.full((scores.shape[0], self.V), float("-inf"), device=scores.device)
        o.scatter_(1, t[:, None], 0.0)
        if self.record:
            self.states.append(scores.clone())
            self.tokens.append(t.clone())
        return o


def set_fixed_length(m, pad_id):
    m.generation_config.eos_token_id = None                      # generate exactly the requested number of tokens
    m.generation_config.pad_token_id = pad_id


def encode(tok, texts, chat_kwargs, device):
    """Chat-templated, left-padded batch. The template adds its own special tokens."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok.apply_chat_template([[{"role": "user", "content": t}] for t in texts], add_generation_prompt=True, tokenize=True,
                                  padding=True, return_tensors="pt", return_dict=True, **chat_kwargs)
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}


def generate(m, enc, n, gen_kw, proc=None):
    from transformers import LogitsProcessorList
    kw = dict(gen_kw)
    if proc is not None:
        kw["logits_processor"] = LogitsProcessorList([proc])
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=n, do_sample=False, repetition_penalty=1.0, **kw)
    assert o.shape[1] == enc["input_ids"].shape[1] + n, (o.shape, n)
    return o[:, enc["input_ids"].shape[1]:]


def reference_tokens(H, W, chunk=16384):
    """argmax of the FP32 product of FP32 states H with the BF16 head W, lowest index on ties; rows in chunks."""
    best = torch.full((H.shape[0],), float("-inf"), device=H.device)
    arg = torch.zeros(H.shape[0], dtype=torch.long, device=H.device)
    for r0 in range(0, W.shape[0], chunk):
        z = H @ W[r0:r0 + chunk].float().T
        v, i = z.max(1)
        up = v > best
        best, arg = torch.where(up, v, best), torch.where(up, i + r0, arg)
    return arg
