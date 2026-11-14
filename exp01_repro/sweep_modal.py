"""K3 on Modal: the one-direction share of the final hidden states across models (sweep_rule.json).

  modal run sweep_modal.py --models qwen3_4b --smoke     # 2 prompts per group, 16 new tokens
  modal run sweep_modal.py                               # every model of the rule
Writes logs/sweep_<model>.json (one summary per model) and per-state arrays to the volume (sweep/<model>.npz).
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "sweep_rule.json")
app = modal.App("exp01-sweep")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
                 "numpy", "pandas", "pyarrow", "requests")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
)
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "hdir_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

NEW = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_30b_a3b": "Qwen/Qwen3-30B-A3B",
    "gemma4_e2b": "google/gemma-4-E2B-it",
    "phi4_mini": "microsoft/Phi-4-mini-instruct",
    "sarvam_m": "sarvamai/sarvam-m",
    "sarvam_1": "sarvamai/sarvam-1",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "granite33_8b": "ibm-granite/granite-3.3-8b-instruct",
    "smollm3_3b": "HuggingFaceTB/SmolLM3-3B",
    "gpt_oss_20b": "openai/gpt-oss-20b",
}
EXISTING = ("qwen3_8b", "gemma4_e4b", "sarvam_30b")
GEN_TOKENS, BATCH = 64, 16


def _q(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return dict(median=float(np.median(x)), p99=float(np.percentile(x, 99)), mean=float(x.mean()))


def _dump(repo, smoke):
    """Greedy states of a new model: (H bf16 [n, d], rows, W bf16 [V, d], cap, meta)."""
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = HfApi().model_info(repo).sha
    P = [json.loads(l) for l in open(f"{VOL}/data/prompts.jsonl")]
    if smoke:
        P = [p for p in P if p["id"] % 40 < 2]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
    head = m.get_output_embeddings()
    cap = None
    for c in (m.config, getattr(m.config, "text_config", None)):
        v = getattr(c, "final_logit_softcapping", None) if c is not None else None
        cap = float(v) if v else cap
    captured = []
    hook = head.register_forward_pre_hook(lambda mod, args: captured.append(args[0][:, -1, :].detach().clone()))
    templ = bool(getattr(tok, "chat_template", None))
    eos = set([tok.eos_token_id] + list(getattr(m.generation_config, "eos_token_id", None) or [])
              if not isinstance(getattr(m.generation_config, "eos_token_id", None), int)
              else [tok.eos_token_id, m.generation_config.eos_token_id])
    H, rows = [], []
    ntok = 16 if smoke else GEN_TOKENS
    for bi in range(0, len(P), BATCH):
        batch = P[bi:bi + BATCH]
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
        captured.clear()
        with torch.no_grad():
            out = m.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=ntok,
                             do_sample=False, pad_token_id=tok.pad_token_id, repetition_penalty=1.0, use_cache=True)
        new = out[:, enc["input_ids"].shape[1]:].cpu()
        hs = torch.stack(captured[:new.shape[1]], 1).cpu()
        for j, p in enumerate(batch):
            ids = new[j].tolist()
            n = next((t + 1 for t, x in enumerate(ids) if x in eos), len(ids))
            n = min(n, hs.shape[1])
            H.append(hs[j, :n])
            rows += [dict(pid=p["id"], group=p["group"], step=t, tok=ids[t], mode="greedy") for t in range(n)]
    hook.remove()
    W = head.weight.detach().clone().to(torch.bfloat16)
    meta = dict(repo=repo, rev=rev, torch=torch.__version__, transformers=transformers.__version__,
                chat_template=templ, model_type=getattr(m.config, "model_type", None),
                head_bias=getattr(head, "bias", None) is not None,
                tied=bool(head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()))
    del m
    torch.cuda.empty_cache()
    return torch.cat(H).to(torch.bfloat16), rows, W, cap, meta


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", cpu=8, memory=96 * 1024, timeout=3 * 3600)
def survey(name: str, smoke: bool, rule_sha: str):
    import gc

    import numpy as np
    import torch

    import a3_lib
    import hdir_lib
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if name in EXISTING:
            cfg = MODELS[name]
            Hb = torch.load(f"{VOL}/states/{name}/h_greedy.pt")
            rows = [json.loads(l) for l in open(f"{VOL}/states/{name}/rows_greedy.jsonl")]
            keep = [i for i, r in enumerate(rows) if r["step"] < GEN_TOKENS and (not smoke or r["pid"] % 40 < 2)]
            Hb, rows = Hb[keep], [rows[i] for i in keep]
            Wb, cap = _load_head(cfg), cfg["cap"]
            meta = dict(repo=cfg["repo"], rev=cfg["rev"], source="saved exp01 states, steps 0 to 63")
        else:
            Hb, rows, Wb, cap, meta = _dump(NEW[name], smoke)
        gc.collect()
        torch.cuda.empty_cache()
        W = Wb.cuda().float()
        V, d = W.shape
        H = Hb.cuda().float()
        pid = torch.tensor([r["pid"] for r in rows])
        cal, test = (pid % 2 == 0), (pid % 2 == 1)
        mu64 = a3_lib.mean_row(W)
        Pall, info = hdir_lib.fit_directions(mu64, H[cal.cuda()], kmax=16)
        P = {"mu": Pall[:, :1], "pca1": Pall[:, 1:2]}
        ti = test.nonzero().squeeze(1).tolist()
        trows = [rows[i] for i in ti]
        Ht = H[test.cuda()]
        emitted = torch.tensor([r["tok"] for r in trows], device="cuda")
        Wb_g = Wb.cuda()
        st = {k: [] for k in ("top_raw", "top_minus_median", "share_within_32", "offset", "offset_share",
                              "rho_mu", "rho_pca1", "align")}
        for c0 in range(0, len(trows), 128):
            h = Ht[c0:c0 + 128]
            z = h @ W.T
            top = z.amax(1)
            off = (h.double() @ mu64).float()
            hd = h.double()
            n2 = (hd ** 2).sum(1)
            st["top_raw"].append(top)
            st["top_minus_median"].append(top - z.median(1).values)
            st["share_within_32"].append((z >= top[:, None] - 32).float().mean(1))
            st["offset"].append(off)
            st["offset_share"].append(off / top)
            st["rho_mu"].append((((hd @ P["mu"]) ** 2).sum(1) / n2).float())
            st["rho_pca1"].append((((hd @ P["pca1"]) ** 2).sum(1) / n2).float())
            zb = (h.bfloat16() @ Wb_g.T).float()
            st["align"].append((zb.argmax(1) == emitted[c0:c0 + 128]).float())
        st = {k: torch.cat(v).cpu().numpy() for k, v in st.items()}
        del Wb_g
        out = dict(name=name, meta=meta, V=V, d=d, cap=cap, n_states=len(rows), n_cal=int(cal.sum()), n_test=len(trows),
                   prompts=len(set(pid.tolist())), spectrum=dict(top1=info["share"][0], top4=info["share"][1],
                                                                  top16=info["share"][2], cos_mu_pc1=info["cos_mu_pc1"]),
                   mu_norm=info["mu_norm"], alignment=float(st["align"].mean()),
                   **{k: _q(st[k]) for k in st if k != "align"})
        arrays = dict(st, pid=np.array([r["pid"] for r in trows]), step=np.array([r["step"] for r in trows]))
        if d % 128 == 0:
            w_norm = W.norm(dim=1)
            vs = a3_lib.build(W, mu64, 0.0)
            del vs["int4_g128"]
            rall = hdir_lib.err_proj(vs["int4d_g128"].Wq, W, Pall[:, :2])
            R = {"mu": rall[:, :1], "pca1": rall[:, 1:2]}
            rules = {"greedy": dict(rule="greedy", cap=cap)}
            rec, stats = {}, None
            for c0 in range(0, len(trows), 128):
                r_, s_ = hdir_lib.run_chunk(W, mu64, w_norm, vs, P, R, Ht[c0:c0 + 128], trows[c0:c0 + 128], rules)
                for k, x in r_.items():
                    rec.setdefault(k, []).append(x)
                if stats is None:
                    stats = s_
                else:
                    for f in ("viol", "mismatch"):
                        for k, v in s_[f].items():
                            stats[f][k] = stats[f].get(k, 0) + v
            rec = {k: torch.cat(v).numpy() for k, v in rec.items()}
            cert = dict(violations=sum(stats["viol"].values()), mismatches=sum(stats["mismatch"].values()))
            for s in ("none", "mu", "pca1"):
                cert[s] = dict(bytes=_q(rec[f"t0/{s}/greedy/bytes"]), S=_q(rec[f"t0/{s}/greedy/S"]),
                               bound_vs_none=float(np.median(rec[f"t0/{s}/b_med"] / rec["t0/none/b_med"])))
            out["certificate"] = cert
            arrays.update({k.replace("/", "__"): v for k, v in rec.items()})
        else:
            out["certificate"] = None
        if not smoke:
            os.makedirs(f"{VOL}/sweep", exist_ok=True)
            np.savez_compressed(f"{VOL}/sweep/{name}.npz", **arrays)
            json.dump(dict(out, rule_sha=rule_sha), open(f"{VOL}/sweep/{name}.json", "w"))
            vol.commit()
        return out
    except Exception as e:                                    # a model that fails is reported, not replaced
        import traceback
        return dict(name=name, failed=f"{type(e).__name__}: {e}"[:600], trace=traceback.format_exc()[-1500:])


@app.local_entrypoint()
def main(models: str = "all", smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"sweep_rule.json sha256 {sha[:12]} is not logged")
    names = list(EXISTING) + list(NEW) if models == "all" else models.split(",")
    for r in survey.starmap([(n, smoke, sha) for n in names], order_outputs=False):
        r["rule_sha"] = sha
        json.dump(r, open(os.path.join(HERE, "logs", f"sweep_{r['name']}{'_smoke' if smoke else ''}.json"), "w"), indent=1)
        if "failed" in r:
            print(r["name"], "FAILED", r["failed"][:300], flush=True)
            continue
        c = r["certificate"]
        print(r["name"], "V", r["V"], "d", r["d"], "states", r["n_states"], "align %.3f" % r["alignment"],
              "top1 %.3f top4 %.3f" % (r["spectrum"]["top1"], r["spectrum"]["top4"]),
              "cos %.2f" % r["spectrum"]["cos_mu_pc1"], "rho_mu %.3f rho_pca1 %.3f" % (r["rho_mu"]["median"], r["rho_pca1"]["median"]),
              "offset share %.2f" % r["offset_share"]["median"],
              ("| bytes p99 none %.3f mu %.3f pca1 %.3f | viol %d mism %d" % (
                  c["none"]["bytes"]["p99"], c["mu"]["bytes"]["p99"], c["pca1"]["bytes"]["p99"], c["violations"],
                  c["mismatches"])) if c else "| no certificate (d not a multiple of 128)", flush=True)
