"""Shared model table and head loader for the kernel runners (same pins as exp01_repro)."""
import json

MODELS = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218",
                     head="lm_head.weight", cap=None),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0),
}
RULES = {"greedy": ("greedy", dict(rule="greedy")), "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0))}


def load_head(cfg):
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


def med(x):
    x = sorted(x)
    return x[len(x) // 2]
