"""A second prompt set, and state dumps for it and for three more models.

  modal run gen2_modal.py --step prompts                 # prompts2.jsonl on the volume (kept private, like prompts.jsonl)
  modal run gen2_modal.py --step sarvam [--smoke]        # Sarvam-30B on the second prompt set -> states/sarvam_30b_p2
  modal run gen2_modal.py --step small  [--smoke]        # Sarvam-1, SmolLM3-3B, Phi-4-mini on the first set -> states/<model>

The second set shares no prompt with the first and changes the task for the Indian languages:
  english  MT-Bench questions 6 to 10 of each category (the first set took 1 to 5)
  code     HumanEval tasks 40 to 79 (the first set took 0 to 39)
  hindi, bengali, tamil, telugu   XQuAD-IN dev: a passage and a question in the language, answer in the language
                                  (the first set was translation from English, from Flores-IN)
Dumps follow exp01_modal.dump_states and sarvam_modal.dump_states: greedy and T = 1 sampling, 128 new tokens, the
head's input at the last position. The small-model dump also saves the head (W.pt), since tied heads have no
tensor of their own in the checkpoint.
"""
import json
import os
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "vizuara_certified_sampling")
app = modal.App("exp01-gen2")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
main_img = (modal.Image.debian_slim(python_version="3.12")
            .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy", "pandas",
                         "pyarrow", "requests")
            .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"}))
sarvam_img = (modal.Image.debian_slim(python_version="3.11")
              .pip_install_from_requirements(os.path.join(LAB, "sarvam.lock.txt"))
              .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"}))

LANGS = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu"}
SMALL = {
    "sarvam_1": dict(repo="sarvamai/sarvam-1", rev="e9607337286ddf496d4a2562b194e489dcf3feea"),
    "smollm3_3b": dict(repo="HuggingFaceTB/SmolLM3-3B", rev=None),
    "phi4_mini": dict(repo="microsoft/Phi-4-mini-instruct", rev=None),
    "qwen3_0.6b": dict(repo="Qwen/Qwen3-0.6B", rev=None),          # added for E2: a small model with a large head
}
SARVAM_PATH = f"{VOL}/models/sarvam_30b"
PREFILL = "<think></think>\n"
GEN_TOKENS, BATCH = 128, 16


@app.function(image=main_img, volumes={VOL: vol}, timeout=900)
def prompts():
    import hashlib
    import io

    import pandas as pd
    import requests

    out = []
    q = [json.loads(l) for l in requests.get(
        "https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/resolve/main/raw/question.jsonl").text.splitlines() if l]
    bycat = {}
    for x in q:
        bycat.setdefault(x["category"], []).append(x)
    for cat in sorted(bycat):
        for x in sorted(bycat[cat], key=lambda r: r["question_id"])[5:10]:
            out.append(dict(group="english", src=f"mtbench:{x['question_id']}", text=(x.get("turns") or x["prompt"])[0]))
    he = pd.read_parquet(io.BytesIO(requests.get(
        "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet").content))
    for _, r in he.iloc[40:80].iterrows():
        out.append(dict(group="code", src=f"humaneval:{r['task_id']}",
                        text="Complete the following Python function. Reply with code only.\n\n```python\n" + r["prompt"] + "```"))
    for lc, lname in LANGS.items():
        ex = requests.get(f"https://raw.githubusercontent.com/google-research-datasets/indic-gen-bench/main/"
                          f"xquad_in/xquad_{lc}_dev.json").json()["examples"]
        for e in ex[:40]:
            out.append(dict(group=lname.lower(), src=f"xquad_{lc}_dev:{e['id']}",
                            text=f"Read the passage and answer the question in {lname}.\n\nPassage: {e['context']}\n\n"
                                 f"Question: {e['question']}"))
    for i, p in enumerate(out):
        p["id"] = 1000 + i
    counts = {g: sum(1 for p in out if p["group"] == g) for g in sorted({p["group"] for p in out})}
    assert len(out) == 240 and set(counts.values()) == {40}, counts
    old = {json.loads(l)["text"] for l in open(f"{VOL}/data/prompts.jsonl")}
    assert not old & {p["text"] for p in out}, "a prompt of the second set is also in the first"
    with open(f"{VOL}/data/prompts2.jsonl", "w") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    vol.commit()
    sha = hashlib.sha256(open(f"{VOL}/data/prompts2.jsonl", "rb").read()).hexdigest()
    return dict(counts=counts, sha256=sha, median_chars={g: sorted(len(p["text"]) for p in out if p["group"] == g)[20]
                                                         for g in counts})


def _generate_and_save(m, tok, head, encode, P, outdir, eos, meta, gen_extra):
    import torch

    captured = []
    hook = head.register_forward_pre_hook(lambda mod, args: captured.append(args[0][:, -1, :].detach().clone()))
    os.makedirs(outdir, exist_ok=True)
    t0, n_states = time.time(), 0
    for mode in ("greedy", "t1"):
        H, rows = [], []
        for bi in range(0, len(P), BATCH):
            batch = P[bi:bi + BATCH]
            enc = encode(batch)
            captured.clear()
            torch.manual_seed(1000 + bi)
            kw = dict(do_sample=False) if mode == "greedy" else dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
            with torch.no_grad():
                out = m.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=GEN_TOKENS, pad_token_id=tok.pad_token_id, repetition_penalty=1.0,
                                 use_cache=True, **gen_extra, **kw)
            new = out[:, enc["input_ids"].shape[1]:].cpu()
            hs = torch.stack(captured[:new.shape[1]], 1).cpu()
            for j, p in enumerate(batch):
                ids = new[j].tolist()
                n = min(next((t + 1 for t, x in enumerate(ids) if x in eos), len(ids)), hs.shape[1])
                H.append(hs[j, :n])
                rows += [dict(pid=p["id"], group=p["group"], step=t, tok=ids[t], mode=mode) for t in range(n)]
            print(f"{mode} batch {bi // BATCH}: {sum(len(x) for x in H)} states, {time.time() - t0:.0f}s", flush=True)
        torch.save(torch.cat(H).to(torch.bfloat16), f"{outdir}/h_{mode}.pt")
        with open(f"{outdir}/rows_{mode}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        meta[f"n_{mode}"] = len(rows)
        n_states += len(rows)
    hook.remove()
    meta.update(n_states=n_states, seconds=round(time.time() - t0, 1), prompts=len(P))
    json.dump(meta, open(f"{outdir}/meta.json", "w"), indent=1)
    vol.commit()
    return meta


@app.function(image=sarvam_img, gpu="A100-80GB", cpu=8, memory=96 * 1024, volumes={VOL: vol}, timeout=3 * 3600)
def sarvam(smoke: bool = False):
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
    e = m.generation_config.eos_token_id
    eos = sorted(set(e if isinstance(e, list) else [26 if e is None else e]))
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts2.jsonl")]
    if smoke:
        P = [p for p in P if p["id"] % 40 < 2]

    def encode(batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + PREFILL for p in batch]
        return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

    meta = dict(model="sarvam_30b", prompt_set="prompts2", torch=torch.__version__, transformers=transformers.__version__,
                eos=eos, prefill=PREFILL)
    return _generate_and_save(m, tok, m.get_output_embeddings(), encode, P,
                              f"{VOL}/states/sarvam_30b_p2{'_smoke' if smoke else ''}", set(eos), meta, dict(eos_token_id=eos))


@app.function(image=main_img, gpu=["A10G", "L4", "A100-40GB"], cpu=8, memory=48 * 1024, volumes={VOL: vol}, timeout=3 * 3600)
def small(name: str, smoke: bool = False):
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = SMALL[name]
    rev = cfg["rev"] or HfApi().model_info(cfg["repo"]).sha
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
    head = m.get_output_embeddings()
    templ = bool(getattr(tok, "chat_template", None))
    g = getattr(m.generation_config, "eos_token_id", None)
    eos = {tok.eos_token_id} | ({g} if isinstance(g, int) else set(g or []))
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    if smoke:
        P = [p for p in P if p["id"] % 40 < 2]

    def encode(batch):                                             # the text K3 fed these models
        if templ:
            msgs = [[{"role": "user", "content": p["text"]}] for p in batch]
            try:
                texts = [tok.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                         for x in msgs]
            except Exception:
                texts = [tok.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in msgs]
        else:
            texts = [p["text"] for p in batch]
        return tok(texts, return_tensors="pt", padding=True, add_special_tokens=not templ).to("cuda")

    outdir = f"{VOL}/states/{name}{'_smoke' if smoke else ''}"
    os.makedirs(outdir, exist_ok=True)
    W = head.weight.detach().to(torch.bfloat16).cpu()
    torch.save(W, f"{outdir}/W.pt")
    cap = None
    for c in (m.config, getattr(m.config, "text_config", None)):
        v = getattr(c, "final_logit_softcapping", None) if c is not None else None
        cap = float(v) if v else cap
    meta = dict(model=name, repo=cfg["repo"], rev=rev, prompt_set="prompts", torch=torch.__version__,
                transformers=transformers.__version__, eos=sorted(eos), chat_template=templ, cap=cap, V=W.shape[0], d=W.shape[1],
                head_bias=getattr(head, "bias", None) is not None,
                tied=bool(head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()),
                gpu=torch.cuda.get_device_name(0))
    return _generate_and_save(m, tok, head, encode, P, outdir, eos, meta, {})


@app.local_entrypoint()
def main(step: str = "prompts", smoke: bool = False, models: str = "sarvam_1,smollm3_3b,phi4_mini"):
    if step == "prompts":
        r = prompts.remote()
        print(json.dumps(r))
        json.dump(r, open(os.path.join(HERE, "logs", "prompts2_meta.json"), "w"), indent=1)
    elif step == "sarvam":
        print(json.dumps(sarvam.remote(smoke)))
    elif step == "small":
        for r in small.starmap([(n, smoke) for n in models.split(",")], order_outputs=False, return_exceptions=True):
            print(json.dumps(r) if isinstance(r, dict) else f"FAILED {r!r}"[:500], flush=True)
