"""E3 on Modal: the certified head inside the decode loop of Qwen3-0.6B (e3_rule.json).

  modal run e3_modal.py [--gpus A10G,A100-80GB] [--smoke]
Writes logs/e3_<gpu>.json (numbers only).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
KER = os.path.join(HERE, "..", "kernel")
RULE = os.path.join(HERE, "e3_rule.json")
app = modal.App("exp01-e3")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"}))
for _d, _f in ((HERE, "cert_core.py"), (KER, "cert_head.py"), (KER, "cert_head2.py"), (KER, "int4_pass1.py"), (KER, "fused.py"),
               (KER, "fused2.py")):
    image = image.add_local_file(os.path.join(_d, _f), f"/root/{_f}")

REPO, REV, SHIFT = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 0.5
N_SHORT, N_LONG = 8, 72


def _q(x):
    s = sorted(x)
    return dict(median=s[len(s) // 2], p99=s[int(0.99 * (len(s) - 1))], mean=sum(s) / len(s), n=len(s))


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", cpu=8, memory=32 * 1024, timeout=2 * 3600, retries=0)
def run(rule_sha: str, smoke: bool = False):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

    from cert_head2 import CertHead2
    from fused2 import FusedCert2, Graphed

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = dict(model="qwen3_0.6b", repo=REPO, rev=REV, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               transformers=transformers.__version__, rule_sha=rule_sha, smoke=smoke)
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)

    def load():
        m = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()
        m.generation_config.eos_token_id = None                  # generate exactly the requested number of tokens
        m.generation_config.pad_token_id = tok.eos_token_id
        return m

    m_stock, m_cert = load(), load()
    W = m_cert.get_output_embeddings().weight.detach().to(torch.bfloat16).clone()
    V, d = W.shape
    ch = CertHead2(W, t=SHIFT)
    fc = FusedCert2(ch)
    cfg = dict(rule="greedy", cap=None)
    g0 = torch.zeros(1, V, device="cuda")
    h0 = torch.zeros(1, d, device="cuda")
    fc(h0, g0, cfg)
    gf = Graphed(lambda h, g: fc(h, g, cfg), h0, g0)
    m_cert.lm_head = torch.nn.Identity()                           # the forward pass now returns the final hidden state

    class CertHead(LogitsProcessor):
        """Receives the final hidden state where generate expects logits; returns scores whose argmax is certified."""

        def __init__(self):
            self.record, self.states, self.tokens = False, [], []

        def __call__(self, input_ids, scores):
            t = gf(scores, g0)[0]
            o = torch.full((scores.shape[0], V), float("-inf"), device=scores.device)
            o.scatter_(1, t[:, None], 0.0)
            if self.record:
                self.states.append(scores.clone())
                self.tokens.append(t.clone())
            return o

    proc = CertHead()
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    P = [p for p in P if p["id"] % 40 < (1 if smoke else 4)]
    encs = []
    for p in P:
        text = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        encs.append(tok(text, return_tensors="pt", add_special_tokens=False).to("cuda"))

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def gen(m, enc, n, cert):
        kw = dict(logits_processor=LogitsProcessorList([proc])) if cert else {}
        torch.cuda.synchronize()
        s = ev()
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=n, do_sample=False, repetition_penalty=1.0, cache_implementation="static", **kw)
        e = ev(); e.synchronize()
        assert o.shape[1] == enc["input_ids"].shape[1] + n, (o.shape, n)
        return s.elapsed_time(e), o[0, enc["input_ids"].shape[1]:].tolist()

    def per_token(m, cert):
        vals = []
        for enc in encs:
            gen(m, enc, N_SHORT, cert); gen(m, enc, N_LONG, cert)             # warm up this prompt's shapes
            a = min(gen(m, enc, N_SHORT, cert)[0] for _ in range(2))
            b = min(gen(m, enc, N_LONG, cert)[0] for _ in range(2))
            vals.append((b - a) / (N_LONG - N_SHORT))
        return vals

    step = {}
    step["stock"] = per_token(m_stock, False)
    print("stock", _q(step["stock"]), flush=True)
    step["certified"] = per_token(m_cert, True)
    print("certified", _q(step["certified"]), flush=True)
    step["stock_again"] = per_token(m_stock, False)
    print("stock again", _q(step["stock_again"]), flush=True)
    out["step_ms"] = {k: _q(v) for k, v in step.items()}
    out["ratio_per_prompt"] = _q([c / s for c, s in zip(step["certified"], step["stock"])])
    out["ratio_of_medians"] = out["step_ms"]["certified"]["median"] / out["step_ms"]["stock"]["median"]
    out["stock_drift"] = out["step_ms"]["stock_again"]["median"] / out["step_ms"]["stock"]["median"]

    # untimed check pass: every certified token against the FP32 reference, and the sequences against the stock model's
    proc.record = True
    same, n_tok, mism = 0, 0, 0
    for enc in encs:
        proc.states, proc.tokens = [], []
        _, seq_c = gen(m_cert, enc, N_LONG, True)
        _, seq_s = gen(m_stock, enc, N_LONG, False)
        H = torch.cat(proc.states)
        T = torch.cat(proc.tokens)
        ref = (H @ W.float().T).argmax(1)
        mism += int((T != ref).sum())
        n_tok += int(T.numel())
        assert T.tolist() == seq_c, "the processor's tokens are not the tokens generate emitted"
        same += int(seq_c == seq_s)
    proc.record = False
    out["check"] = dict(tokens=n_tok, mismatches_vs_reference=mism, prompts=len(encs), sequences_identical_to_stock=same)
    return out


@app.local_entrypoint()
def main(gpus: str = "A10G,A100-80GB", smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"e3_rule.json sha256 {sha[:12]} is not logged")
    calls = [(g, run.with_options(gpu=g).spawn(sha, smoke)) for g in gpus.split(",")]
    for g, c in calls:
        try:
            r = c.get()
        except Exception as e:
            print(g, "FAILED", f"{type(e).__name__}: {e}"[:1500], flush=True)
            continue
        json.dump(r, open(os.path.join(HERE, "logs", f"e3_{g}{'_smoke' if smoke else ''}.json"), "w"), indent=1)
        print(g, r["gpu"], "| step ms", {k: round(v["median"], 3) for k, v in r["step_ms"].items()}, "| certified / stock",
              round(r["ratio_of_medians"], 4), "| drift", round(r["stock_drift"], 4), "| check", r["check"], flush=True)
