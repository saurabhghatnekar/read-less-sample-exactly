"""E1 on Modal: the head's share of a greedy decode step and the step time with the certified head (e2e_rule.json).

  modal run e2e_modal.py --models qwen3_8b --smoke
  modal run e2e_modal.py                                   # qwen3_8b, gemma4_e4b, sarvam_30b
Writes logs/e2e_<model>.json (Sarvam: e2e_sarvam_30b_stock.json and e2e_sarvam_30b_head.json).
Summarize with: python3 -I e2e_summarize.py
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "vizuara_certified_sampling")
RULE = os.path.join(HERE, "e2e_rule.json")
app = modal.App("exp01-e2e")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
main_img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
                 "numpy", "pandas", "pyarrow", "requests")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
)
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "l4_kernels.py", "l4_cert.py", "l4_graph.py",
           "l4_corr.py"):
    main_img = main_img.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")
sarvam_img = (modal.Image.debian_slim(python_version="3.11")
              .pip_install_from_requirements(os.path.join(LAB, "sarvam.lock.txt"))
              .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"}))

T_STAR = {"qwen3_8b": 0.5, "gemma4_e4b": 1.5, "sarvam_30b": 0.5}
CHAT = {"qwen3_8b": {"enable_thinking": False}, "gemma4_e4b": {}}
GEN_TOKENS, WARM_STEPS = 64, 4
SARVAM_PATH = f"{VOL}/models/sarvam_30b"
SARVAM_PREFILL = "<think></think>\n"


def _stats(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return dict(median=float(np.median(x)), p99=float(np.percentile(x, 99)), mean=float(x.mean()))


def _prompt_sets(smoke):
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    one = [p for p in P if p["id"] % 40 < (1 if smoke else 8)]        # 8 per group = 48 prompts
    many = P[:16] if smoke else P
    return one, many


def _loop(m, head, encode, prompts, B, ntok, cert=None, Wf=None):
    """Minimal greedy decode with the cache. One record per timed step. encode(batch) -> (ids, mask) on the GPU.
    cert: an object with .step(hb) -> (tokens, rows8, rowsR, fell_back); it is timed on the same state."""
    import torch

    marks = {}

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def pre(mod, args):
        marks["e1"] = ev()
        marks["h"] = args[0][:, -1, :].detach()

    def post(mod, args, out):
        marks["e2"] = ev()

    hooks = [head.register_forward_pre_hook(pre), head.register_forward_hook(post)]
    recs, toks = [], []
    with torch.no_grad():
        for bi in range(0, len(prompts) - B + 1, B):
            ids, mask = encode(prompts[bi:bi + B])
            out = m(input_ids=ids, attention_mask=mask, use_cache=True)          # the prompt pass is not timed
            pkv, tok = out.past_key_values, out.logits[:, -1, :].argmax(-1)
            seq = [tok.cpu()]
            for t in range(ntok - 1):
                mask = torch.cat([mask, mask.new_ones(B, 1)], 1)
                torch.cuda.synchronize()
                e0 = ev()
                out = m(input_ids=tok[:, None], attention_mask=mask, past_key_values=pkv, use_cache=True)
                nxt = out.logits[:, -1, :].argmax(-1)
                e3 = ev()
                torch.cuda.synchronize()
                r = dict(batch=bi // B, t=t, ctx=int(mask.shape[1]), body=e0.elapsed_time(marks["e1"]),
                         head=marks["e1"].elapsed_time(marks["e2"]), tail=marks["e2"].elapsed_time(e3))
                if cert is not None:
                    hb = marks["h"].to(torch.bfloat16)
                    e4 = ev()
                    ctok, _, _, fb = cert.step(hb)
                    e5 = ev()
                    torch.cuda.synchronize()
                    r["cert"] = e4.elapsed_time(e5)
                    r["fallback"] = int(fb)
                    r["mismatch"] = int((ctok.to(hb.device) != (hb.float() @ Wf.T).argmax(1)).sum())
                    r["differs_from_stock"] = int((ctok.to(hb.device) != nxt).sum())
                pkv, tok = out.past_key_values, nxt
                seq.append(tok.cpu())
                recs.append(r)
            toks.append(torch.stack(seq, 1))
    for h in hooks:
        h.remove()
    return recs, toks


def _summ(recs):
    import numpy as np
    R = [r for r in recs if r["t"] >= WARM_STEPS]
    a = {k: np.array([r[k] for r in R], dtype=float) for k in ("body", "head", "tail", "ctx")}
    stock = a["body"] + a["head"] + a["tail"]
    out = dict(steps=len(R), ctx=dict(min=float(a["ctx"].min()), max=float(a["ctx"].max())),
               body_ms=_stats(a["body"]), head_ms=_stats(a["head"]), tail_ms=_stats(a["tail"]), stock_ms=_stats(stock),
               head_share=dict(_stats((a["head"] + a["tail"]) / stock),
                               of_totals=float((a["head"] + a["tail"]).sum() / stock.sum())))
    if "cert" in R[0]:
        c = np.array([r["cert"] for r in R], dtype=float)
        out.update(cert_ms=_stats(c), cert_over_head=_stats(c / (a["head"] + a["tail"])),
                   end_to_end=dict(_stats((a["body"] + c) / stock), of_totals=float((a["body"] + c).sum() / stock.sum())),
                   fallbacks=int(sum(r["fallback"] for r in R)), mismatches=int(sum(r["mismatch"] for r in R)),
                   tokens_differing_from_stock=int(sum(r["differs_from_stock"] for r in R)))
    return out


@app.function(image=main_img, volumes={VOL: vol}, gpu="A100-80GB", cpu=8, memory=96 * 1024, timeout=3 * 3600)
def dense(model: str, smoke: bool):
    """Qwen3-8B and Gemma 4 E4B: stock step and certified head in one process."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import l4_cert
    import l4_graph
    from a3_modal import MODELS

    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = MODELS[model]
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["rev"])
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=cfg["rev"], dtype=torch.bfloat16,
                                             device_map="cuda").eval()
    head = m.get_output_embeddings()
    Wb = head.weight.detach()
    chead = l4_cert.CertHead(Wb, T_STAR[model])
    Wf = Wb.float()

    def encode(batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                         add_generation_prompt=True, **CHAT[model]) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        return enc["input_ids"], enc["attention_mask"]

    one, many = _prompt_sets(smoke)
    ntok = 12 if smoke else GEN_TOKENS
    res = dict(model=model, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               transformers=transformers.__version__, V=Wb.shape[0], D=Wb.shape[1])
    for B, prompts in ((1, one), (16, many)):
        gs = l4_graph.GraphStep(chead, B)
        gs.capture()
        for i in range(10):
            gs.step(Wb[i:i + B].contiguous())                 # warm the graph on any BF16 rows of the right shape
        recs, toks = _loop(m, head, encode, prompts, B, ntok, cert=gs, Wf=Wf)
        res[f"B{B}"] = _summ(recs)
        if B == 1:                                            # control T3: the loop is the model's greedy decode
            agree = total = 0
            for i in range(min(4, len(prompts))):
                ids, mask = encode(prompts[i:i + 1])
                with torch.no_grad():
                    g = m.generate(input_ids=ids, attention_mask=mask, max_new_tokens=ntok, do_sample=False,
                                   pad_token_id=tok.pad_token_id)[0, ids.shape[1]:].cpu()
                mine = toks[i][0, :len(g)]
                agree += int((mine == g).sum())
                total += len(g)
            res["T3_loop_equals_generate"] = dict(agree=agree, total=total)
        del gs
    return res


@app.function(image=sarvam_img, volumes={VOL: vol}, gpu="A100-80GB", cpu=8, memory=96 * 1024, timeout=3 * 3600)
def sarvam_stock(smoke: bool):
    """Sarvam-30B, container (a): the stock loop in Sarvam's pinned environment."""
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    torch.backends.cuda.matmul.allow_tf32 = False
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{SARVAM_PATH}/tokenizer.json", pad_token="<pad>", unk_token="<unk>",
                                  clean_up_tokenization_spaces=False)
    tok.chat_template = open(f"{SARVAM_PATH}/chat_template.jinja").read()
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(SARVAM_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
                                             trust_remote_code=True, attn_implementation="sdpa").eval()
    head = m.get_output_embeddings()

    def encode(batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + SARVAM_PREFILL
                 for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        return enc["input_ids"], enc["attention_mask"]

    one, many = _prompt_sets(smoke)
    ntok = 12 if smoke else GEN_TOKENS
    res = dict(model="sarvam_30b", part="stock", gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               transformers=transformers.__version__)
    for B, prompts in ((1, one), (16, many)):
        _loop(m, head, encode, prompts[:B], B, 6)             # warm-up
        recs, _ = _loop(m, head, encode, prompts, B, ntok)
        res[f"B{B}"] = _summ(recs)
    return res


@app.function(image=main_img, volumes={VOL: vol}, gpu="A100-80GB", cpu=8, memory=96 * 1024, timeout=3 * 3600)
def sarvam_head(smoke: bool):
    """Sarvam-30B, container (b): the BF16 head and the certified head (K2 pca1) on the saved greedy states."""
    import numpy as np
    import torch

    import l4_corr
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    Wb = _load_head(MODELS["sarvam_30b"]).cuda()
    Pall = torch.load(f"{VOL}/hdir/sarvam_30b/P.pt")
    head = l4_corr.CorrHead(Wb, T_STAR["sarvam_30b"], {"pca1": Pall[:, 1]})
    head.use("pca1")
    Wf = Wb.float()
    n = 64 if smoke else 512
    H = torch.load(f"{VOL}/states/sarvam_30b/h_greedy.pt")
    H = H[torch.randperm(len(H), generator=torch.Generator().manual_seed(0))[:n + 64]].cuda()

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    res = dict(model="sarvam_30b", part="head", gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    for B in (1, 16):
        gs = l4_corr.GraphStepCorr(head, B)
        gs.capture()
        steps = [H[i:i + B] for i in range(0, n, B)]
        for hb in steps[:10]:
            torch.argmax(hb @ Wb.T, dim=1)
            gs.step(hb)
        bf, ce, mism, fbs = [], [], 0, 0
        for hb in steps:
            torch.cuda.synchronize()
            a = ev()
            torch.argmax(hb @ Wb.T, dim=1)
            b = ev()
            torch.cuda.synchronize()
            c = ev()
            ctok, _, _, fb = gs.step(hb)
            d = ev()
            torch.cuda.synchronize()
            bf.append(a.elapsed_time(b))
            ce.append(c.elapsed_time(d))
            fbs += int(fb)
            mism += int((ctok.cuda() != (hb.float() @ Wf.T).argmax(1)).sum())
        res[f"B{B}"] = dict(steps=len(steps), bf16_head_ms=_stats(bf), cert_ms=_stats(ce),
                            cert_over_bf16=_stats(np.array(ce) / np.array(bf)), mismatches=mism, fallbacks=fbs)
        del gs
    return res


@app.local_entrypoint()
def main(models: str = "qwen3_8b,gemma4_e4b,sarvam_30b", smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"e2e_rule.json sha256 {sha[:12]} is not logged")
    calls = []
    for name in models.split(","):
        if name == "sarvam_30b":
            calls += [("sarvam_30b_stock", sarvam_stock.spawn(smoke)), ("sarvam_30b_head", sarvam_head.spawn(smoke))]
        else:
            calls.append((name, dense.spawn(name, smoke)))
    for name, call in calls:
        try:
            r = call.get()
        except Exception as e:
            print(name, "FAILED", f"{type(e).__name__}: {e}"[:800], flush=True)
            continue
        r["rule_sha"] = sha
        json.dump(r, open(os.path.join(HERE, "logs", f"e2e_{name}{'_smoke' if smoke else ''}.json"), "w"), indent=1)
        for B in ("B1", "B16"):
            x = r[B]
            if "body_ms" in x:
                print(name, B, "steps", x["steps"], "body %.2f head %.2f tail %.2f ms" % (
                    x["body_ms"]["median"], x["head_ms"]["median"], x["tail_ms"]["median"]),
                    "head share %.3f" % x["head_share"]["of_totals"],
                    ("cert %.2f ms e2e %.3f mism %d fb %d" % (x["cert_ms"]["median"], x["end_to_end"]["of_totals"],
                                                             x["mismatches"], x["fallbacks"])) if "cert_ms" in x else "",
                    flush=True)
            else:
                print(name, B, "bf16 head %.2f ms cert %.2f ms ratio %.3f mism %d fb %d" % (
                    x["bf16_head_ms"]["median"], x["cert_ms"]["median"], x["cert_over_bf16"]["median"], x["mismatches"],
                    x["fallbacks"]), flush=True)
        if "T3_loop_equals_generate" in r:
            print(name, "T3", r["T3_loop_equals_generate"], flush=True)
