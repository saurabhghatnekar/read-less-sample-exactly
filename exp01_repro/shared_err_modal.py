"""Exploratory shared-error probe on the three real heads (bound widths only; no sets, bytes or verdicts).

  modal run shared_err_modal.py            # all three models -> logs/shared_err_probe.json
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("exp01-shared-err")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"})
         .add_local_file(os.path.join(HERE, "cert_core.py"), "/root/cert_core.py")
         .add_local_file(os.path.join(HERE, "a3_lib.py"), "/root/a3_lib.py")
         .add_local_file(os.path.join(HERE, "a3_modal.py"), "/root/a3_modal.py")
         .add_local_file(os.path.join(HERE, "shared_err_lib.py"), "/root/shared_err_lib.py"))


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", memory=96 * 1024, timeout=2 * 3600)
def run(model: str, n_states: int = 1024):
    import torch

    import a3_lib
    import cert_core as cc
    import shared_err_lib as se
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    W = _load_head(MODELS[model]).cuda().float()
    mu = a3_lib.mean_row(W)
    g = torch.Generator().manual_seed(0)
    hs = []
    for mode in ("greedy", "t1"):
        H = torch.load(f"{VOL}/states/{model}/h_{mode}.pt")
        hs.append(H[torch.randperm(len(H), generator=g)[:n_states // 2]])
    h = torch.cat(hs).cuda().float()
    out = {}
    for t in (0.0, 1.0):
        Wt = a3_lib.shifted(W, mu, t)
        for f in (cc.make_int8_row, cc.make_int4_g, cc.make_int4_dither):
            v = cc.prepare(f(Wt))
            out.update({f"t{t:g}/{k}": x for k, x in se.probe(h, {v.name: (Wt, v)}).items()})
            del v
            torch.cuda.empty_cache()
        del Wt
    return model, out


@app.local_entrypoint()
def main(models: str = "qwen3_8b,gemma4_e4b,sarvam_30b"):
    res = dict(run.starmap([(m,) for m in models.split(",")]))
    path = os.path.join(HERE, "logs", "shared_err_probe.json")
    json.dump(res, open(path, "w"), indent=1)
    for m, out in res.items():
        print(f"\n== {m}")
        for k, r in out.items():
            extra = f"  hoeff/res {r['hoeff_over_res_mean']:.3f}" if "hoeff_over_res_mean" in r else ""
            print(f"{k:22s}" + "".join(f"  {x}: E{r[x]['energy_share']:.3f} mean{r[x]['mean_ratio']:.3f} max{r[x]['max_ratio']:.3f}"
                                         for x in ("rmean", "r1", "r4", "r16")) + extra)
