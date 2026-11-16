"""N1 on Modal: where the dominant state direction appears, before or after the final norm and by layer (norm_rule.json).

  modal run norm_probe_modal.py --smoke          # 12 prompts, 16 new tokens, nothing saved
  modal run norm_probe_modal.py                  # sarvam_30b and sarvam_1
Writes logs/normprobe_<model>.json (numbers and module names only). The states of the last eight blocks, the
pre-norm and the post-norm sites are saved to the volume (normprobe/<model>/), which is private.
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "vizuara_certified_sampling")
RULE = os.path.join(HERE, "norm_rule.json")
app = modal.App("exp01-normprobe")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
main_img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(os.path.join(HERE, "norm_lib.py"), "/root/norm_lib.py")
)
sarvam_img = (modal.Image.debian_slim(python_version="3.11")
              .pip_install_from_requirements(os.path.join(LAB, "sarvam.lock.txt"))
              .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
              .add_local_file(os.path.join(HERE, "norm_lib.py"), "/root/norm_lib.py"))

SARVAM_PATH = f"{VOL}/models/sarvam_30b"
SARVAM_PREFILL = "<think></think>\n"
SARVAM1 = dict(repo="sarvamai/sarvam-1", rev="e9607337286ddf496d4a2562b194e489dcf3feea")
GEN_TOKENS, BATCH, KEEP_BLOCKS = 64, 16, 8


def _prompts(smoke):
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    return [p for p in P if p["id"] % 40 < 2] if smoke else P


def _eos(m, tok):
    ids = set()
    for x in (getattr(tok, "eos_token_id", None), getattr(m.config, "eos_token_id", None),
              getattr(m.generation_config, "eos_token_id", None)):
        if isinstance(x, int):
            ids.add(x)
        elif x:
            ids.update(int(i) for i in x)
    return ids


def _run(name, m, tok, encode, smoke, ref_p=None, extra=None):
    import gc

    import torch
    import transformers

    import norm_lib

    torch.backends.cuda.matmul.allow_tf32 = False
    prompts = _prompts(smoke)
    sites, rows, ctl = norm_lib.greedy_capture(m, encode, prompts, BATCH, 16 if smoke else GEN_TOKENS, _eos(m, tok))
    _, _, _, norm, head = norm_lib.find_sites(m)
    W = head.weight.detach().clone()
    meta = dict(model=name, smoke=smoke, torch=torch.__version__, transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0), model_type=getattr(m.config, "model_type", None),
                eos=sorted(_eos(m, tok)), **(extra or {}))
    norm = norm.float()
    del m
    gc.collect()
    torch.cuda.empty_cache()
    out = norm_lib.analyse(sites, rows, norm, W=W, ref_p=ref_p, dev="cuda")
    out.update(meta=meta, control=ctl)
    if not smoke:
        os.makedirs(f"{VOL}/normprobe/{name}", exist_ok=True)
        blocks = [k for k in sites if k.startswith("block_")][-KEEP_BLOCKS:]
        torch.save({k: sites[k] for k in blocks + ["pre_norm", "post_norm"]}, f"{VOL}/normprobe/{name}/sites.pt")
        with open(f"{VOL}/normprobe/{name}/rows.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        vol.commit()
    return out


@app.function(image=sarvam_img, volumes={VOL: vol}, gpu="A100-80GB", cpu=8, memory=96 * 1024, timeout=5400, retries=0)
def sarvam_30b(smoke: bool):
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=f"{SARVAM_PATH}/tokenizer.json", pad_token="<pad>", unk_token="<unk>",
                                  clean_up_tokenization_spaces=False)
    tok.chat_template = open(f"{SARVAM_PATH}/chat_template.jinja").read()
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(SARVAM_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
                                             trust_remote_code=True, attn_implementation="sdpa").eval()

    def encode(batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + SARVAM_PREFILL
                 for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        return enc["input_ids"], enc["attention_mask"]

    try:
        ref = torch.load(f"{VOL}/hdir/sarvam_30b/P.pt", map_location="cpu")[:, 1]
    except Exception as e:                                         # the comparison is then missing, and reported so
        ref, note = None, f"{type(e).__name__}: {e}"[:200]
    else:
        note = "K2 pca1 from hdir/sarvam_30b/P.pt"
    return _run("sarvam_30b", m, tok, encode, smoke, ref_p=ref, extra=dict(saved_direction=note))


@app.function(image=main_img, volumes={VOL: vol}, gpu=["L4", "A10G", "A100-40GB"], cpu=8, memory=48 * 1024, timeout=5400,
              retries=0)                                    # the first of these that is free: an L4 queue stalled the smoke
def sarvam_1(smoke: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SARVAM1["repo"], revision=SARVAM1["rev"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(SARVAM1["repo"], revision=SARVAM1["rev"], dtype=torch.bfloat16,
                                             device_map="cuda").eval()
    templ = bool(getattr(tok, "chat_template", None))

    def encode(batch):                                             # the same text K3 fed this model
        if templ:
            msgs = [[{"role": "user", "content": p["text"]}] for p in batch]
            try:
                texts = [tok.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                         for x in msgs]
            except Exception:
                texts = [tok.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in msgs]
        else:
            texts = [p["text"] for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=not templ).to("cuda")
        return enc["input_ids"], enc["attention_mask"]

    return _run("sarvam_1", m, tok, encode, smoke, extra=dict(chat_template=templ, **SARVAM1))


FUNCS = {"sarvam_30b": sarvam_30b, "sarvam_1": sarvam_1}


@app.local_entrypoint()
def main(models: str = "sarvam_30b,sarvam_1", smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"norm_rule.json sha256 {sha[:12]} is not logged")
    calls = [(n, FUNCS[n].spawn(smoke)) for n in models.split(",")]
    for name, c in calls:
        r = c.get()
        r["rule_sha"] = sha
        json.dump(r, open(os.path.join(HERE, "logs", f"normprobe_{name}{'_smoke' if smoke else ''}.json"), "w"), indent=1)
        v, ctl, nm = r["verdict"], r["control"], r["norm"]
        print(name, "states", r["n_states"], "| layers", ctl["n_layers"], ctl["layers"], "| norm", ctl["norm"], nm["cls"],
              "gain:", nm["gain_source"], "| hook diffs", ctl["norm_out_differs_from_head_in"],
              ctl["last_block_differs_from_norm_in"], "of", ctl["steps"],
              "| top1 pre %.3f gain-free %.3f post %.3f -> %s" % (
                  r["sites"]["pre_norm"]["top1"], v["gain_free_top1"], v["post_top1"], v["verdict"]),
              "| first block at half:", v["first_block_at_half_of_pre_norm"], flush=True)
