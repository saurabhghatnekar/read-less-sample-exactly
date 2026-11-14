"""Sarvam-30B states for A3, in the exp01 reproduction's format, on the exp01-repro volume.

  modal run sarvam_modal.py --step fetch       # download the pinned checkpoint to the volume (CPU, ~64 GB)
  modal run sarvam_modal.py --smoke            # 2 prompts per group -> states/sarvam_30b_smoke
  modal run sarvam_modal.py                    # all 240 prompts      -> states/sarvam_30b

Environment and model handling follow Vizuara's run B (../vizuara_certified_sampling/exp01_modal.py, MIT):
their Sarvam lockfile (torch 2.5.1, transformers 4.57.2), the pinned revision, hash checks on Sarvam's own model code,
the PreTrainedTokenizerFast workaround, and the "<think></think>\\n" prefill that closes the reasoning block.
Prompts and generation settings are this reproduction's (prompts.jsonl on the volume, batch 16, 128 new tokens,
greedy and T = 1 with seeds 1000 + batch index), so the states line up with the Qwen and Gemma runs.
"""
import hashlib
import json
import os
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "vizuara_certified_sampling")

app = modal.App("exp01-sarvam")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
ENV = {"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"}
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install_from_requirements(os.path.join(LAB, "sarvam.lock.txt")).env(ENV))

REPO, REV = "sarvamai/sarvam-30b", "071ae95e933605ca1104a6b4524a36a98488efa4"
V, D = 262144, 4096
SARVAM_CODE = {"configuration_sarvam_moe.py": "105ba18705cd28a075231f2b4b8764c3171256616187470f9da4b55c315a1360",
               "modeling_sarvam_moe.py": "87b0983d7094de03df116219a3d75ae083dc4aac5b9b70455c2ac60706b7e4e3"}
PREFILL = "<think></think>\n"
GEN_TOKENS, BATCH = 128, 16
SCRIPTS = {"hindi": (0x0900, 0x097F), "bengali": (0x0980, 0x09FF), "tamil": (0x0B80, 0x0BFF), "telugu": (0x0C00, 0x0C7F)}
MPATH = f"{VOL}/models/sarvam_30b"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def script_share(text, group):
    lo, hi = SCRIPTS[group]
    letters = [c for c in text if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in letters) / max(1, len(letters))


@app.function(image=img, cpu=4, memory=16 * 1024, volumes={VOL: vol}, timeout=3 * 3600)
def fetch_model():
    from huggingface_hub import snapshot_download
    ok = f"{MPATH}/.complete"
    if not os.path.exists(ok):
        snapshot_download(REPO, revision=REV, local_dir=MPATH, max_workers=8,
                          ignore_patterns=["*.gguf", "*.bin", "*.pth", "*.msgpack", "*.h5", "*.onnx", "*.tflite"])
        for f, sha in SARVAM_CODE.items():
            assert sha_file(f"{MPATH}/{f}") == sha, f"{f} differs from the file Vizuara read"
        open(ok, "w").write(REV)
        vol.commit()
    return sorted(os.listdir(MPATH))


@app.function(image=img, gpu="A100-80GB", cpu=8, memory=96 * 1024, volumes={VOL: vol}, timeout=3 * 3600)
def dump_states(smoke: bool = False):
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    torch.backends.cuda.matmul.allow_tf32 = False
    assert open(f"{MPATH}/.complete").read() == REV
    for f, sha in SARVAM_CODE.items():
        assert sha_file(f"{MPATH}/{f}") == sha, f"{f} differs from the file Vizuara read"
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{MPATH}/tokenizer.json", pad_token="<pad>", unk_token="<unk>",
                                  clean_up_tokenization_spaces=False)
    tok.chat_template = open(f"{MPATH}/chat_template.jinja").read()
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MPATH, torch_dtype=torch.bfloat16, device_map="cuda",
                                             trust_remote_code=True, attn_implementation="sdpa").eval()
    head = m.get_output_embeddings()
    assert tuple(head.weight.shape) == (V, D)
    e = m.generation_config.eos_token_id
    eos = sorted(set(e if isinstance(e, list) else [26 if e is None else e]))

    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    if smoke:
        P = [p for p in P if p["id"] % 40 < 2]
    captured = []
    hook = head.register_forward_pre_hook(lambda mod, args: captured.append(args[0][:, -1, :].detach().clone()))
    outdir = f"{VOL}/states/sarvam_30b{'_smoke' if smoke else ''}"
    os.makedirs(outdir, exist_ok=True)
    meta = dict(torch=torch.__version__, transformers=transformers.__version__, repo=REPO, rev=REV, prefill=PREFILL,
                eos=eos, head_dtype=str(head.weight.dtype),
                tied=bool(head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()))
    t0, n_states, c7 = time.time(), 0, {}
    for mode in ("greedy", "t1"):
        H, rows, outs = [], [], []
        for bi in range(0, len(P), BATCH):
            batch = P[bi:bi + BATCH]
            texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) + PREFILL for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            captured.clear()
            torch.manual_seed(1000 + bi)
            kw = dict(do_sample=False) if mode == "greedy" else dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
            with torch.no_grad():
                out = m.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=GEN_TOKENS, pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                 repetition_penalty=1.0, use_cache=True, **kw)
            new = out[:, enc["input_ids"].shape[1]:].cpu()
            assert len(captured) == new.shape[1], f"{len(captured)} head calls for {new.shape[1]} tokens"
            hs = torch.stack(captured, 1).cpu()
            for j, p in enumerate(batch):
                ids = new[j].tolist()
                n = next((t + 1 for t, x in enumerate(ids) if x in eos), len(ids))
                H.append(hs[j, :n])
                rows += [dict(pid=p["id"], group=p["group"], step=t, tok=ids[t], mode=mode) for t in range(n)]
                text = tok.decode(ids[:n], skip_special_tokens=True)
                outs.append(dict(pid=p["id"], group=p["group"], n=n, text=text))
                if p["group"] in SCRIPTS:
                    c7.setdefault(f"{mode}/{p['group']}", []).append(script_share(text, p["group"]))
            print(f"{mode} batch {bi // BATCH}: {sum(len(x) for x in H)} states, {time.time() - t0:.0f}s", flush=True)
        torch.save(torch.cat(H).to(torch.bfloat16), f"{outdir}/h_{mode}.pt")
        with open(f"{outdir}/rows_{mode}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(f"{outdir}/outputs_{mode}.jsonl", "w", encoding="utf-8") as f:   # private: Flores-derived prompts
            for o in outs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        n_states += len(rows)
    hook.remove()
    import statistics
    meta.update(n_states=n_states, seconds=round(time.time() - t0, 1),
                c7_median_target_script_share={k: round(statistics.median(v), 3) for k, v in sorted(c7.items())},
                mean_tokens_per_prompt=round(n_states / (2 * len(P)), 1))
    json.dump(meta, open(f"{outdir}/meta.json", "w"), indent=1)
    vol.commit()
    return meta


@app.local_entrypoint()
def main(step: str = "dump", smoke: bool = False):
    if step == "fetch":
        print(fetch_model.remote())
    else:
        print(json.dumps(dump_states.remote(smoke), indent=1))
