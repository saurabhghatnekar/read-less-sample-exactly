"""CPU check of E4's integration pieces (e4_lib.py) on tiny random models. Run: python3 -I test_e4.py

Checks, for a model without a soft-cap (Qwen2 layout) and one with (Gemma 2 layout, soft-cap 30):
  the copy shares every weight and leaves the stock model as it was; the soft-cap is found and switched off;
  the certified model's output is bit for bit the tensor the stock output layer receives; with an FP32 argmax in
  place of the kernel, generate emits the stock model's tokens at batch 1 and at a left-padded batch of 4.
The GPU kernel and the tokenizer path (e4_lib.encode) are not exercised here; the Modal smoke covers them.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as L  # noqa: E402
from transformers import Gemma2Config, Gemma2ForCausalLM, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

V, D = 311, 64
COMMON = dict(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2)


def softcap(z, cap):
    return z if cap is None else cap * torch.tanh(z / cap)


def check(name, m_stock, cap):
    m_stock.eval()
    n_params = sum(p.numel() for p in m_stock.parameters())
    W = m_stock.get_output_embeddings().weight.detach()
    m_cert = L.share_params_copy(m_stock)
    found = L.disable_softcap(m_cert)
    where = L.to_identity_head(m_cert)
    assert (cap is None) == (not found) and all(v == cap for v in found.values()), found
    assert getattr(m_stock.config, "final_logit_softcapping", None) == cap, "the stock model's soft-cap was touched"
    assert isinstance(m_stock.get_output_embeddings(), torch.nn.Linear), "the stock model's head was touched"
    ps, pc = dict(m_stock.named_parameters()), dict(m_cert.named_parameters())
    assert all(pc[k] is ps[k] for k in pc), "a parameter was copied"
    assert set(ps) - set(pc) <= {"lm_head.weight"} and sum(p.numel() for p in m_cert.parameters()) <= n_params
    for m in (m_stock, m_cert):
        L.set_fixed_length(m, 0)

    g = torch.Generator().manual_seed(1)
    for B in (1, 4):
        lens = torch.randint(5, 12, (B,), generator=g)
        T = int(lens.max())
        ids = torch.randint(3, V, (B, T), generator=g)
        mask = (torch.arange(T)[None, :] >= (T - lens)[:, None]).long()               # left padding
        enc = dict(input_ids=ids * mask, attention_mask=mask)
        a = L.head_input(m_stock, enc)
        with torch.no_grad():
            b = m_cert(**enc).logits[:, -1]
        assert a.shape == b.shape == (B, D) and float((a - b).abs().max()) == 0.0, "state control"

        proc = L.CertProcessor(lambda h: softcap(h @ W.T, cap).argmax(1), V)
        proc.record = True
        seq_c = L.generate(m_cert, enc, 24, {}, proc)
        seq_s = L.generate(m_stock, enc, 24, {})
        assert torch.equal(torch.stack(proc.tokens, 1), seq_c), "processor tokens are not the emitted tokens"
        assert torch.equal(seq_c, seq_s), "certified and stock sequences differ"
        for h, t in zip(proc.states, proc.tokens):
            assert torch.equal(L.reference_tokens(h, W, chunk=100), (h @ W.float().T).argmax(1))
            assert torch.equal(t, L.reference_tokens(h, W, chunk=100))
        print(f"{name}: batch {B} ok ({seq_c.numel()} tokens; soft-cap switched off at {list(found) or 'none'}; head at {where})")


torch.manual_seed(0)
check("qwen2 layout, no soft-cap", Qwen2ForCausalLM(Qwen2Config(**COMMON, tie_word_embeddings=True)), None)
check("gemma2 layout, soft-cap 30", Gemma2ForCausalLM(Gemma2Config(**COMMON, head_dim=16, final_logit_softcapping=30.0,
                                                                   query_pre_attn_scalar=16, sliding_window=64)), 30.0)
print("ALL OK")
