"""E4 on Modal: the head's share of a decode step, and the certified head inside the loop, on Qwen3-1.7B and
Gemma 4 E4B at batch 1 and 8 (e4_rule.json).

  modal run e4_modal.py [--models qwen3_1.7b,gemma4_e4b] [--gpu A10G] [--smoke]
Writes logs/e4_<model>_<gpu>.json (numbers only).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
KER = os.path.join(HERE, "..", "kernel")
RULE = os.path.join(HERE, "e4_rule.json")
app = modal.App("exp01-e4")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"}))
for _d, _f in ((HERE, "cert_core.py"), (KER, "cert_head.py"), (KER, "cert_head2.py"), (KER, "int4_pass1.py"), (KER, "fused.py"),
               (KER, "fused2.py"), (KER, "common.py"), (HERE, "e4_lib.py")):
    image = image.add_local_file(os.path.join(_d, _f), f"/root/{_f}")

MODELS = {
    "qwen3_1.7b": dict(repo="Qwen/Qwen3-1.7B", rev="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", head="lm_head.weight", cap=None,
                       t=0.5, chat={"enable_thinking": False}),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0, t=1.5, chat={}),
}
PROMPTS = {1: (4, 1), 8: (16, 3)}        # batch size -> prompts with id mod 40 below (full, smoke)
HEAD_STATES = 256                        # recorded state batches used for the head-alone timing


def _q(x):
    s = sorted(x)
    return dict(median=s[len(s) // 2], p99=s[int(0.99 * (len(s) - 1))], mean=sum(s) / len(s), n=len(s))


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", cpu=8, memory=48 * 1024, timeout=2 * 3600, retries=0)
def run(model: str, rule_sha: str, smoke: bool = False):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import e4_lib as L
    from cert_head import softcap
    from cert_head2 import CertHead2
    from common import load_head
    from fused2 import FusedCert2, Graphed

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = MODELS[model]
    out = dict(model=model, repo=c["repo"], rev=c["rev"], gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               transformers=transformers.__version__, rule_sha=rule_sha, smoke=smoke, errors={}, batches={})

    # the certified head first, from the head tensor alone: its construction needs a few GB of scratch
    ch = CertHead2(load_head(c).cuda(), t=c["t"])
    fc = FusedCert2(ch)
    V, d = ch.V, ch.d
    cfg = dict(rule="greedy", cap=c["cap"])
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(c["repo"], revision=c["rev"])
    m_stock = AutoModelForCausalLM.from_pretrained(c["repo"], revision=c["rev"], dtype=torch.bfloat16, device_map="cuda").eval()
    out["control_head_weights_equal"] = bool(torch.equal(m_stock.get_output_embeddings().weight.detach(), ch.W))
    m_cert = L.share_params_copy(m_stock)
    out["softcap_switched_off"] = L.disable_softcap(m_cert)
    out["head_replaced"] = L.to_identity_head(m_cert)
    assert (c["cap"] is None) == (not out["softcap_switched_off"]), ("soft-cap", c["cap"], out["softcap_switched_off"])
    assert all(v == c["cap"] for v in out["softcap_switched_off"].values()), out["softcap_switched_off"]
    for m in (m_stock, m_cert):
        L.set_fixed_length(m, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    out["gpu_mem_gb_after_load"] = torch.cuda.memory_allocated() / 1e9

    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]

    def batches(B):
        ps = [p["text"] for p in P if p["id"] % 40 < PROMPTS[B][1 if smoke else 0]]
        return [L.encode(tok, ps[i:i + B], c["chat"], "cuda") for i in range(0, len(ps) - B + 1, B)]

    # control: the certified model's output is the tensor the stock output layer receives
    diffs = []
    for enc in batches(1)[:2]:
        a = L.head_input(m_stock, enc)
        with torch.no_grad():
            b = m_cert(**enc).logits[:, -1]
        diffs.append(float((a.float() - b.float()).abs().max()))
        assert a.shape == b.shape == (1, d), (a.shape, b.shape)
    out["control_state_max_abs_diff"] = diffs
    if max(diffs) != 0.0:
        raise RuntimeError(f"state control failed: {diffs}")

    # the loop configuration: static cache if it works on this model, else the default cache (recorded)
    gen_kw = dict(cache_implementation="static")
    try:
        L.generate(m_stock, batches(1)[0], 4, gen_kw)
    except Exception as e:
        out["errors"]["static_cache"] = f"{type(e).__name__}: {e}"[:800]
        print("static cache FAILED, using the default cache:", out["errors"]["static_cache"][:300], flush=True)
        gen_kw = {}
    out["cache"] = "static" if gen_kw else "default"

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def timed(m, enc, n, proc):
        torch.cuda.synchronize()
        s = ev()
        L.generate(m, enc, n, gen_kw, proc)
        e = ev(); e.synchronize()
        return s.elapsed_time(e)

    def per_step(m, encs, proc):
        vals = []
        for enc in encs:
            timed(m, enc, L.N_SHORT, proc); timed(m, enc, L.N_LONG, proc)           # warm up this batch's shapes
            a = min(timed(m, enc, L.N_SHORT, proc) for _ in range(2))
            b = min(timed(m, enc, L.N_LONG, proc) for _ in range(2))
            vals.append((b - a) / (L.N_LONG - L.N_SHORT))
        return vals

    def base(h, g):
        return softcap((h.bfloat16() @ ch.W.T).float(), c["cap"]).argmax(1)

    for B in (1, 8):
        encs = batches(B)
        h0, g0 = torch.zeros(B, d, device="cuda"), torch.zeros(1, 1, device="cuda")     # greedy: the noise input is not read
        fc(h0, g0, cfg); base(h0, g0)
        gf = Graphed(lambda h, g: fc(h, g, cfg), h0, g0)
        gb = Graphed(base, h0, g0)
        proc = L.CertProcessor(lambda h: gf(h, g0)[0], V)
        r = dict(n_batches=len(encs), prompt_tokens=_q([int(e["input_ids"].shape[1]) for e in encs]))

        step = {}
        step["stock"] = per_step(m_stock, encs, None)
        print(model, B, "stock", _q(step["stock"]), flush=True)
        step["certified"] = per_step(m_cert, encs, proc)
        print(model, B, "certified", _q(step["certified"]), flush=True)
        step["stock_again"] = per_step(m_stock, encs, None)
        r["step_ms"] = {k: _q(v) for k, v in step.items()}
        r["ratio_per_batch"] = _q([x / y for x, y in zip(step["certified"], step["stock"])])
        r["ratio_per_batch_all"] = [x / y for x, y in zip(step["certified"], step["stock"])]
        r["in_loop_ratio"] = r["step_ms"]["certified"]["median"] / r["step_ms"]["stock"]["median"]
        r["stock_drift"] = r["step_ms"]["stock_again"]["median"] / r["step_ms"]["stock"]["median"]

        # untimed check pass: every certified token against the FP32 reference; sequences against the stock model's
        proc.record = True
        same = n_tok = mism = rows = 0
        states = []
        for enc in encs:
            proc.states, proc.tokens = [], []
            seq_c = L.generate(m_cert, enc, L.N_LONG, gen_kw, proc)
            seq_s = L.generate(m_stock, enc, L.N_LONG, gen_kw)
            T = torch.stack(proc.tokens, 1)                                           # B x steps
            assert torch.equal(T, seq_c), "the processor's tokens are not the tokens generate emitted"
            for h, t in zip(proc.states, proc.tokens):
                mism += int((t != L.reference_tokens(h, ch.W)).sum())
                n_tok += int(t.numel())
            states += proc.states
            same += int((seq_c == seq_s).all(1).sum())
            rows += B
        proc.record = False
        r["check"] = dict(tokens=n_tok, mismatches_vs_reference=mism, rows=rows, rows_identical_to_stock=same)

        # the two heads alone, on recorded states at this batch size
        pick = torch.randperm(len(states), generator=torch.Generator().manual_seed(0))[:HEAD_STATES].tolist()
        t_stock, t_cert, hm, hs, rows8 = [], [], 0, 0, []
        for j, i in enumerate([pick[0]] * 8 + pick):
            h = states[i]
            a, b = [], []
            for _ in range(3):
                s = ev(); tb = gb(h, g0).clone(); e = ev(); e.synchronize(); a.append(s.elapsed_time(e))
                s = ev(); o = gf(h, g0); e = ev(); e.synchronize(); b.append(s.elapsed_time(e))
                tc, n8 = o[0].clone(), int(o[1].sum())
            if j < 8:
                continue
            ref = L.reference_tokens(h, ch.W)
            t_stock.append(sorted(a)[1]); t_cert.append(sorted(b)[1])
            hm += int((tc != ref).sum()); hs += int((tb != ref).sum()); rows8.append(n8)
        r["head_ms"] = dict(stock=_q(t_stock), certified=_q(t_cert))
        r["head_check"] = dict(states=len(t_stock) * B, cert_mismatches=hm, stock_mismatches=hs, int8_rows_per_step=_q(rows8))
        st, hsm, hcm = r["step_ms"]["stock"]["median"], r["head_ms"]["stock"]["median"], r["head_ms"]["certified"]["median"]
        r["head_share"] = hsm / st
        r["head_ratio"] = hcm / hsm
        r["estimate_by_substitution"] = (st - hsm + hcm) / st
        out["batches"][str(B)] = r
        print(model, B, "| step ms", {k: round(v["median"], 3) for k, v in r["step_ms"].items()}, "| head ms", round(hsm, 3),
              round(hcm, 3), "| share", round(r["head_share"], 4), "| estimate", round(r["estimate_by_substitution"], 4),
              "| in loop", round(r["in_loop_ratio"], 4), "| drift", round(r["stock_drift"], 4), "| check", r["check"], flush=True)
        del gf, gb, proc, states
        torch.cuda.empty_cache()
    out["gpu_mem_gb_peak"] = torch.cuda.max_memory_allocated() / 1e9
    return out


@app.local_entrypoint()
def main(models: str = "qwen3_1.7b,gemma4_e4b", gpu: str = "A10G", smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"e4_rule.json sha256 {sha[:12]} is not logged")
    calls = [(m, run.with_options(gpu=gpu).spawn(m, sha, smoke)) for m in models.split(",")]
    for m, c in calls:
        try:
            r = c.get()
        except Exception as e:
            print(m, "FAILED", f"{type(e).__name__}: {e}"[:1500], flush=True)
            continue
        json.dump(r, open(os.path.join(HERE, "logs", f"e4_{m}_{gpu}{'_smoke' if smoke else ''}.json"), "w"), indent=1)
        print(m, r["gpu"], "| cache", r["cache"], "| controls", r["control_head_weights_equal"], r["control_state_max_abs_diff"],
              "| errors", r["errors"], flush=True)
        for B, b in r["batches"].items():
            print(f"  batch {B}: share {b['head_share']:.4f}, head ratio {b['head_ratio']:.3f}, estimate "
                  f"{b['estimate_by_substitution']:.4f}, in loop {b['in_loop_ratio']:.4f}, drift {b['stock_drift']:.4f}, "
                  f"check {b['check']}", flush=True)
