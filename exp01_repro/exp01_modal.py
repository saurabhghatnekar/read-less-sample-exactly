"""Exp01 reproduction on Modal.

  modal run exp01_modal.py::prompts                 # build prompts.jsonl on the volume
  modal run exp01_modal.py::dump --model qwen3_8b [--smoke]
  modal run exp01_modal.py::analyze --model qwen3_8b [--smoke]
  modal volume get exp01-repro results/qwen3_8b ./results/

Prompts include Flores-IN text (cleared for evaluation with attribution, not for sharing): keep
prompts.jsonl and model outputs private.
"""
import hashlib
import json
import os
import time

import modal

app = modal.App("exp01-repro")
vol = modal.Volume.from_name("exp01-repro", create_if_missing=True)
VOL = "/vol"
HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
                 "numpy", "pandas", "pyarrow", "requests")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(os.path.join(HERE, "cert_core.py"), "/root/cert_core.py")
)

MODELS = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218",
                     head="lm_head.weight", cap=None, chat_kwargs={"enable_thinking": False}),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0, chat_kwargs={}),
    "sarvam_30b": dict(repo="sarvamai/sarvam-30b", rev="071ae95e933605ca1104a6b4524a36a98488efa4",
                       head="lm_head.weight", cap=None, chat_kwargs=None),  # states come from sarvam_modal.py
}
LANGS = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu"}
GEN_TOKENS = 128
BATCH = 16


# ------------------------------------------------------------------------------------------------
@app.function(image=image, volumes={VOL: vol}, timeout=600)
def prompts():
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
        for x in sorted(bycat[cat], key=lambda r: r["question_id"])[:5]:
            out.append(dict(group="english", src=f"mtbench:{x['question_id']}", text=(x.get("turns") or x["prompt"])[0]))
    he = pd.read_parquet(io.BytesIO(requests.get(
        "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet").content))
    for _, r in he.iloc[:40].iterrows():
        out.append(dict(group="code", src=f"humaneval:{r['task_id']}",
                        text="Complete the following Python function. Reply with code only.\n\n```python\n" + r["prompt"] + "```"))
    for lc, lname in LANGS.items():
        ex = requests.get(f"https://raw.githubusercontent.com/google-research-datasets/indic-gen-bench/main/"
                          f"flores_in/flores_en_{lc}_dev.json").json()["examples"]
        for i in range(40):
            src = " ".join(e["source"] for e in ex[5 * i:5 * i + 5])
            out.append(dict(group=lname.lower(), src=f"flores_en_{lc}_dev:{5*i}-{5*i+4}",
                            text=f"Translate the following English text into {lname}. Reply with the translation only.\n\n{src}"))
    for i, p in enumerate(out):
        p["id"] = i
    assert len(out) == 240, len(out)
    os.makedirs(f"{VOL}/data", exist_ok=True)
    with open(f"{VOL}/data/prompts.jsonl", "w") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    vol.commit()
    blob = open(f"{VOL}/data/prompts.jsonl", "rb").read()
    print(len(out), "prompts, sha256", hashlib.sha256(blob).hexdigest()[:16])
    return {g: sum(1 for p in out if p["group"] == g) for g in sorted({p["group"] for p in out})}


# ------------------------------------------------------------------------------------------------
@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def dump_states(model: str, smoke: bool = False):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = MODELS[model]
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    if smoke:
        P = [p for p in P if p["id"] % 40 < 2]       # 2 per group = 12 prompts
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["rev"])
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=cfg["rev"], dtype=torch.bfloat16,
                                             device_map="cuda").eval()
    head = m.get_output_embeddings()
    captured = []
    hook = head.register_forward_pre_hook(lambda mod, args: captured.append(args[0][:, -1, :].detach().clone()))

    outdir = f"{VOL}/states/{model}{'_smoke' if smoke else ''}"
    os.makedirs(outdir, exist_ok=True)
    meta = dict(torch=str(torch.__version__), transformers=str(transformers.__version__), repo=cfg["repo"], rev=cfg["rev"])
    t0 = time.time()
    n_states = 0
    for mode in ("greedy", "t1"):
        H, rows = [], []
        for bi in range(0, len(P), BATCH):
            batch = P[bi:bi + BATCH]
            texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                             add_generation_prompt=True, **cfg["chat_kwargs"]) for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            captured.clear()
            torch.manual_seed(1000 + bi)
            gen_kw = dict(do_sample=False) if mode == "greedy" else dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=None)
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=GEN_TOKENS, pad_token_id=tok.pad_token_id, **gen_kw)
            new = out[:, enc["input_ids"].shape[1]:].cpu()
            hs = torch.stack(captured[:new.shape[1]], 1).cpu()     # [b, steps, d]
            eos = set([tok.eos_token_id] + list(getattr(m.generation_config, "eos_token_id", []) or []))
            for j, p in enumerate(batch):
                ids = new[j].tolist()
                n = len(ids)
                for t, x in enumerate(ids):
                    if x in eos:
                        n = t + 1
                        break
                if n > hs.shape[1]:
                    n = hs.shape[1]
                H.append(hs[j, :n])
                rows += [dict(pid=p["id"], group=p["group"], step=t, tok=ids[t], mode=mode) for t in range(n)]
            print(f"{mode} batch {bi // BATCH}: {sum(len(x) for x in H)} states, {time.time() - t0:.0f}s", flush=True)
        Hc = torch.cat(H).to(torch.bfloat16)
        torch.save(Hc, f"{outdir}/h_{mode}.pt")
        with open(f"{outdir}/rows_{mode}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_states += len(rows)
    hook.remove()
    meta["n_states"] = n_states
    meta["seconds"] = time.time() - t0
    json.dump(meta, open(f"{outdir}/meta.json", "w"), indent=1)
    vol.commit()
    return meta


# ------------------------------------------------------------------------------------------------
RULESETS = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_0.6": ("t1", dict(rule="gumbel", T=0.6)),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
}


def _load_head(cfg):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx_path = None
    try:
        idx_path = hf_hub_download(cfg["repo"], "model.safetensors.index.json", revision=cfg["rev"])
        shard = json.load(open(idx_path))["weight_map"][cfg["head"]]
    except Exception:
        shard = "model.safetensors"
    f = hf_hub_download(cfg["repo"], shard, revision=cfg["rev"])
    with safe_open(f, "pt") as s:
        return s.get_tensor(cfg["head"]).to(torch.bfloat16)


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def analyze_chunk(model: str, mode: str, start: int, stop: int, smoke: bool = False):
    import numpy as np
    import torch

    import cert_core as cc

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}{'_smoke' if smoke else ''}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    Wb = _load_head(cfg).cuda()
    W = Wb.float()
    V, d = W.shape
    w_norm = W.norm(dim=1)
    variants = {v.name: cc.prepare(v) for v in (cc.make_int8_row(W), cc.make_int4_g(W), cc.make_int4_dither(W),
                                                cc.make_int4_rot(W), cc.make_nvfp4(W))}
    cap = cfg["cap"]
    rules = {k: dict(c, cap=cap) for k, (md, c) in RULESETS.items() if md == mode}
    int8 = variants["int8_row"]
    int8_row_bytes = d + 2 + 2
    out = {}
    stats = dict(c2_viol={}, c3_mismatch={}, dither_se=0.0, dither_n=0, c1_match=0, c1_n=0)
    CH = 128
    for c0 in range(0, len(rows), CH):
        h = H[c0:c0 + CH].cuda().float()
        Bc = h.shape[0]
        z_ref = h @ W.T
        keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows[c0:c0 + CH]],
                            dtype=torch.int64, device="cuda")
        g = cc.gumbel(keys, V, "cuda")
        if mode == "greedy":  # C1 alignment: BF16 product from stored h vs emitted token
            zb = (h.bfloat16() @ Wb.T).float()
            em = torch.tensor([r["tok"] for r in rows[c0:c0 + CH]], device="cuda")
            stats["c1_match"] += (zb.argmax(1) == em).sum().item()
            stats["c1_n"] += Bc
        ref_toks = {rn: cc.decide(z_ref, rcfg, g) for rn, rcfg in rules.items()}
        z8 = cc.approx_logits(int8, h)
        b8all, _ = cc.bounds_for(int8, h, w_norm)
        b8 = torch.minimum(b8all["wc"], b8all["res"])
        rec = {}
        for vn, v in variants.items():
            zt = cc.approx_logits(v, h)
            bounds, slack = cc.bounds_for(v, h, w_norm)
            err = (z_ref - zt).abs()
            bounds["oracle"] = err + slack
            rec[f"{vn}/maxerr"] = err.amax(1)
            for bn, b in bounds.items():
                rec[f"{vn}/{bn}/maxbound"] = b.amax(1)
                rec[f"{vn}/{bn}/meanbound"] = b.mean(1)
                stats["c2_viol"][f"{vn}/{bn}"] = stats["c2_viol"].get(f"{vn}/{bn}", 0) + (err > b).sum().item()
            if vn == "int4d_g128":
                hg = h.view(Bc, d // v.G, v.G)
                var = (hg.norm(dim=-1) ** 2) @ (v.scale ** 2).T / 12
                stats["dither_se"] += ((z_ref - zt) ** 2 / var).sum().item()
                stats["dither_n"] += var.numel()
            for bn, b in bounds.items():
                meta_pg = 0.0 if bn in ("hoeff", "oracle") else 2.0
                for rn, rcfg in rules.items():
                    stages = [None] if vn == "int8_row" or bn == "oracle" else [None, (z8, b8)]
                    for st in stages:
                        S, R, tok = cc.certify(z_ref, zt, b, rcfg, g, st)
                        ref_tok = ref_toks[rn]
                        key = f"{vn}/{bn}/{'casc' if st else 'single'}/{rn}"
                        Sc = None if S is None else S.sum(1).float()
                        Rc = R.sum(1).float()
                        by = cc.bytes_fraction(V, d, v.bytes_per_weight, meta_pg, d // v.G, Sc, Rc, int8_row_bytes)
                        rec[f"{key}/S"] = Sc if Sc is not None else torch.zeros_like(Rc)
                        rec[f"{key}/R"] = Rc
                        rec[f"{key}/bytes"] = by
                        stats["c3_mismatch"][key] = stats["c3_mismatch"].get(key, 0) + (tok != ref_tok).sum().item()
        top2 = cc.softcap(z_ref, cap).topk(2, dim=1).values
        rec["margin"] = top2[:, 0] - top2[:, 1]
        rec["toplogit"] = z_ref.amax(1)
        rec["h_conc"] = h.abs().sum(1) / (d ** 0.5 * h.norm(dim=1))
        for k, t in rec.items():
            out.setdefault(k, []).append(t.float().cpu())
        del z_ref, g
    arrs = {k: torch.cat(v).numpy() for k, v in out.items()}
    rdir = f"{VOL}/results/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **arrs)
    json.dump(dict(stats=stats, rows=[(r["pid"], r["group"], r["step"]) for r in rows]),
              open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    return dict(mode=mode, start=start, n=len(rows),
                c2=sum(stats["c2_viol"][k] for k in stats["c2_viol"] if k.split("/")[1] in ("wc", "res")),
                c3=sum(v for k, v in stats["c3_mismatch"].items() if "/hoeff/" not in k))


# ------------------------------------------------------------------------------------------------
@app.local_entrypoint()
def main(step: str = "prompts", model: str = "qwen3_8b", smoke: bool = False, chunk: int = 4096,
         max_parallel: int = 6):
    if step == "prompts":
        print(prompts.remote())
    elif step == "dump":
        print(dump_states.remote(model, smoke))
    elif step == "analyze":
        sdir = f"states/{model}{'_smoke' if smoke else ''}"
        jobs = []
        for mode in ("greedy", "t1"):
            data = b"".join(vol.read_file(f"{sdir}/rows_{mode}.jsonl"))
            n = data.count(b"\n")
            jobs += [(model, mode, s, min(s + chunk, n), smoke) for s in range(0, n, chunk)]
        print(f"{len(jobs)} analysis jobs")
        tot = dict(n=0, c2=0, c3=0)
        for r in analyze_chunk.starmap(jobs, order_outputs=False):
            for k in tot:
                tot[k] += r[k] if k != "n" else r["n"]
            print(r, flush=True)
        print("TOTAL", tot)
