"""Run crosscheck_lib on our saved T=1 decode states (Modal volume exp01-repro), with Vizuara's cert_core.

  modal run crosscheck_modal.py --model gemma4_e4b [--limit 512]
  modal run crosscheck_modal.py --model qwen3_8b

Writes results/crosscheck_<model>.json (per-cell medians, p99, means, mismatches, by language group).
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
VZ = os.path.join(HERE, "..", "vizuara_certified_sampling", "cert_core.py")
app = modal.App("exp01-crosscheck")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
                 "numpy", "pandas", "pyarrow", "requests")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(os.path.join(HERE, "cert_core.py"), "/root/cert_core.py")
    .add_local_file(os.path.join(HERE, "crosscheck_lib.py"), "/root/crosscheck_lib.py")
    .add_local_file(VZ, "/root/vz_core.py")
)
MODELS = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218",
                     head="lm_head.weight", cap=None),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0),
}


def _load_head(cfg):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    try:
        idx = hf_hub_download(cfg["repo"], "model.safetensors.index.json", revision=cfg["rev"])
        shard = json.load(open(idx))["weight_map"][cfg["head"]]
    except Exception:
        shard = "model.safetensors"
    f = hf_hub_download(cfg["repo"], shard, revision=cfg["rev"])
    with safe_open(f, "pt") as s:
        return s.get_tensor(cfg["head"]).to(torch.bfloat16)


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=2 * 3600)
def run_chunk(model: str, start: int, stop: int):
    import torch

    import cert_core as cc
    import crosscheck_lib as xl
    import vz_core as vz

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_t1.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_t1.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    hd = xl.Heads(cc, vz, W)
    out = {}
    CH = 64
    for c0 in range(0, len(rows), CH):
        h = H[c0:c0 + CH].cuda().float()
        keys = torch.tensor([cc.state_key(r["pid"], r["step"], r["mode"]) for r in rows[c0:c0 + CH]],
                            dtype=torch.int64, device="cuda")
        st = xl.chunk_stats(cc, vz, hd, W, h, keys, cfg["cap"])
        for k, v in st.items():
            out.setdefault(k, []).extend(v.float().cpu().tolist())
        torch.cuda.empty_cache()
    out["group"] = [r["group"] for r in rows]
    return out


def _pct(x, q):
    import numpy as np
    return float(np.percentile(np.asarray(x), q, method="inverted_cdf"))


@app.local_entrypoint()
def main(model: str = "gemma4_e4b", chunk: int = 4096, limit: int = 0):
    import numpy as np
    data = b"".join(vol.read_file(f"states/{model}/rows_t1.jsonl"))
    n = data.count(b"\n")
    if limit:
        n = min(n, limit)
    jobs = [(model, s, min(s + chunk, n)) for s in range(0, n, chunk)]
    print(f"{model}: {n} T=1 states, {len(jobs)} jobs", flush=True)
    allv = {}
    for r in run_chunk.starmap(jobs):           # ordered, so groups line up
        for k, v in r.items():
            allv.setdefault(k, []).extend(v)
    groups = np.array(allv.pop("group"))
    res = dict(model=model, n_states=int(len(groups)), cells={})
    for k in sorted(allv):
        a = np.asarray(allv[k])
        if k.endswith("/mismatch") or k.startswith("refagree"):
            res["cells"][k] = dict(sum=float(a.sum()), mean=float(a.mean()))
            continue
        e = dict(median=_pct(a, 50), p99=_pct(a, 99), mean=float(a.mean()))
        if k.endswith("/bytes") or k.endswith("/S"):
            e["by_group"] = {g: _pct(a[groups == g], 50) for g in sorted(set(groups))}
        res["cells"][k] = e
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tag = f"_limit{limit}" if limit else ""
    path = os.path.join(HERE, "results", f"crosscheck_{model}{tag}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)
    for k, e in res["cells"].items():
        if k.endswith("/bytes") or k.endswith("/S") or k.endswith("/mismatch") or k.startswith("refagree"):
            if "median" in e:
                print(f"{k:50s} median {e['median']:>10.4f}  p99 {e['p99']:>10.4f}  mean {e['mean']:.4f}")
            else:
                print(f"{k:50s} sum {e['sum']:.0f}  mean {e['mean']:.5f}")
