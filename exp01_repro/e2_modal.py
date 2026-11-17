"""E2 on Modal: the head's share of a decode step on Qwen3-0.6B in the fastest Hugging Face configuration (e2_rule.json).

  modal run e2_modal.py                 # A100-80GB and A10G
Writes logs/e2_<gpu>.json (numbers only).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
KER = os.path.join(HERE, "..", "kernel")
RULE = os.path.join(HERE, "e2_rule.json")
app = modal.App("exp01-e2")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"}))
for _d, _f in ((HERE, "cert_core.py"), (KER, "cert_head.py"), (KER, "cert_head2.py"), (KER, "int4_pass1.py"), (KER, "fused.py"),
               (KER, "fused2.py")):
    image = image.add_local_file(os.path.join(_d, _f), f"/root/{_f}")

REPO, REV, NAME, SHIFT = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "qwen3_0.6b", 0.5
N_SHORT, N_LONG = 8, 72


def _q(x):
    s = sorted(x)
    return dict(median=s[len(s) // 2], p99=s[int(0.99 * (len(s) - 1))], mean=sum(s) / len(s), n=len(s))


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", cpu=8, memory=32 * 1024, timeout=2 * 3600, retries=0)
def run(rule_sha: str):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import cert_core as cc
    from cert_head2 import CertHead2
    from fused2 import FusedCert2, Graphed

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = dict(model=NAME, repo=REPO, rev=REV, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               transformers=transformers.__version__, rule_sha=rule_sha, errors={})

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    # ---- heads alone, on saved states
    sdir = f"{VOL}/states/{NAME}"
    W = torch.load(f"{sdir}/W.pt").cuda()
    H = torch.load(f"{sdir}/h_greedy.pt")
    pick = torch.randperm(H.shape[0], generator=torch.Generator().manual_seed(0))[:512]
    Hs = H[pick].cuda().float()
    ch = CertHead2(W, t=SHIFT)
    fc = FusedCert2(ch)
    cfg = dict(rule="greedy", cap=None)
    g0 = torch.zeros(1, ch.V, device="cuda")

    def base(h, g):
        return (h.bfloat16() @ ch.W.T).float().argmax(1)

    fc(Hs[:1], g0, cfg); base(Hs[:1], g0)
    gf = Graphed(lambda h, g: fc(h, g, cfg), Hs[:1], g0)
    gb = Graphed(base, Hs[:1], g0)
    t_stock, t_cert, mism, mism_stock, rows8 = [], [], 0, 0, []
    for i in range(-8, Hs.shape[0]):
        h = Hs[max(i, 0):max(i, 0) + 1]
        ref = (h @ ch.W.float().T).argmax(1)
        a, b, c = [], [], []
        for _ in range(3):
            s = ev(); tb = gb(h, g0).clone(); e = ev(); e.synchronize(); a.append(s.elapsed_time(e))
            s = ev(); o = gf(h, g0); e = ev(); e.synchronize(); b.append(s.elapsed_time(e))
            tok, n8 = o[0].clone(), int(o[1][0])
        if i < 0:
            continue
        t_stock.append(sorted(a)[1]); t_cert.append(sorted(b)[1])
        mism += int((tok != ref).sum()); mism_stock += int((tb != ref).sum()); rows8.append(n8)
    out["head"] = dict(stock_ms=_q(t_stock), cert_ms=_q(t_cert), states=len(t_stock), cert_mismatches=mism,
                       stock_mismatches=mism_stock, int8_rows=_q(rows8), V=ch.V, d=ch.d,
                       head_bytes_bf16=2 * ch.V * ch.d, first_pass_bytes=ch.bytes_first())
    print("head alone:", json.dumps(out["head"]), flush=True)
    del fc, gf, gb, ch, W, H, Hs
    torch.cuda.empty_cache()

    # ---- the decode step, by timing generate
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    m = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    P = [p for p in P if p["id"] % 40 < 4]
    encs = []
    for p in P:
        text = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        encs.append(tok(text, return_tensors="pt", add_special_tokens=False).to("cuda"))
    out["prompt_tokens"] = _q([int(e["input_ids"].shape[1]) for e in encs])

    def gen(enc, n, **kw):
        torch.cuda.synchronize()
        s = ev()
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id, **kw)
        e = ev(); e.synchronize()
        assert o.shape[1] == enc["input_ids"].shape[1] + n, (o.shape, n)
        return s.elapsed_time(e)

    def per_token(**kw):
        vals = []
        for enc in encs:
            gen(enc, N_SHORT, **kw); gen(enc, N_LONG, **kw)                 # warm up this prompt's shapes
            a = min(gen(enc, N_SHORT, **kw) for _ in range(2))
            b = min(gen(enc, N_LONG, **kw) for _ in range(2))
            vals.append((b - a) / (N_LONG - N_SHORT))
        return _q(vals)

    out["step"] = {}
    out["step"]["eager"] = per_token()
    print("eager step:", out["step"]["eager"], flush=True)
    for name, setup in (("static_cache", None), ("static_cache_compiled", "compile")):
        try:
            if setup == "compile":
                m.forward = torch.compile(m.forward, mode="reduce-overhead", fullgraph=True)
            out["step"][name] = per_token(cache_implementation="static")
            print(name, "step:", out["step"][name], flush=True)
        except Exception as e:                                   # recorded, not hidden
            out["errors"][name] = f"{type(e).__name__}: {e}"[:600]
            print(name, "FAILED", out["errors"][name][:300], flush=True)
    best = min(out["step"], key=lambda k: out["step"][k]["median"])
    hs, hc = out["head"]["stock_ms"]["median"], out["head"]["cert_ms"]["median"]
    out["derived"] = {k: dict(step_ms=v["median"], head_share=hs / v["median"],
                              estimated_certified_step_ratio=(v["median"] - hs + hc) / v["median"]) for k, v in out["step"].items()}
    out["fastest"] = best
    out["head_ratio"] = hc / hs
    return out


@app.local_entrypoint()
def main(gpus: str = "A100-80GB,A10G"):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"e2_rule.json sha256 {sha[:12]} is not logged")
    calls = [(g, run.with_options(gpu=g).spawn(sha)) for g in gpus.split(",")]
    for g, c in calls:
        try:
            r = c.get()
        except Exception as e:
            print(g, "FAILED", f"{type(e).__name__}: {e}"[:800], flush=True)
            continue
        json.dump(r, open(os.path.join(HERE, "logs", f"e2_{g}.json"), "w"), indent=1)
        print(g, r["gpu"], "| head stock/cert ms", round(r["head"]["stock_ms"]["median"], 3), round(r["head"]["cert_ms"]["median"], 3),
              "ratio", round(r["head_ratio"], 2), "mismatches", r["head"]["cert_mismatches"], "| derived",
              {k: {a: round(b, 3) for a, b in v.items()} for k, v in r["derived"].items()}, "| errors", r["errors"], flush=True)
