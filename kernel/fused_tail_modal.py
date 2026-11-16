"""Tail times of the fused certified head (fused.py) on a Modal L4. Protocol: tail_protocol.json.

  modal run fused_tail_modal.py [--models qwen3_8b,gemma4_e4b] [--gpu L4]

Same kernels, states, noise, baseline and CUDA-event timing as fused_modal.py. What changes is coverage and what
is kept: batch 1 runs over all 512 states, batch 8 over 64 batches, batch 32 over 16; each batch is timed 3 times
per variant; every time, every per-state row count and every mismatch count is returned. No summary is computed
here: tail_summarize.py reads results/fused_tail_<model>_<gpu>.json. Only numbers leave Modal.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("kernel-fused-tail")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "huggingface_hub", "safetensors", "numpy")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_file(os.path.join(HERE, "..", "exp01_repro", "cert_core.py"), "/root/cert_core.py")
    .add_local_file(os.path.join(HERE, "cert_head.py"), "/root/cert_head.py")
    .add_local_file(os.path.join(HERE, "int4_pass1.py"), "/root/int4_pass1.py")
    .add_local_file(os.path.join(HERE, "fused.py"), "/root/fused.py")
    .add_local_file(os.path.join(HERE, "common.py"), "/root/common.py")
)
BATCHES = {1: 512, 8: 64, 32: 16}
REPS = 3
VARIANTS = ("base", "base_graph", "fused", "fused_graph")


def _timed(fn, *a):
    import torch
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn(*a)
    e.record()
    e.synchronize()
    return out, s.elapsed_time(e)


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=0)
def run(model: str, n_states: int = 512):
    import torch
    import triton

    import cert_core as cc
    from cert_head import CertHead, softcap
    from common import MODELS, RULES, load_head
    from fused import FusedCert, Graphed

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = dict(model=model, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
               triton=str(triton.__version__), reps=REPS)
    cfg_m = MODELS[model]
    W = load_head(cfg_m).cuda()
    ch = CertHead(W)
    del W
    torch.cuda.empty_cache()
    fc = FusedCert(ch)
    V = ch.V
    out.update(V=V, d=ch.d)
    gen = torch.Generator().manual_seed(0)              # same generator and rule order as fused_modal.py
    out["rules"] = {}

    def base(h, g, cfg):
        c = softcap((h.bfloat16() @ ch.W.T).float(), cfg["cap"])
        return (c if cfg["rule"] == "greedy" else c / cfg["T"] + g).argmax(1)

    for rn, (mode, rcfg) in RULES.items():
        cfg = dict(rcfg, cap=cfg_m["cap"])
        sdir = f"{VOL}/states/{model}"
        H = torch.load(f"{sdir}/h_{mode}.pt")
        rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")]
        pick = torch.randperm(len(rows), generator=gen)[:n_states].tolist()
        Hs = H[pick].cuda().float()
        keys = torch.tensor([cc.state_key(rows[i]["pid"], rows[i]["step"], rows[i]["mode"]) for i in pick],
                            dtype=torch.int64, device="cuda")
        del H
        res = {}
        for B, nb in BATCHES.items():
            h0, g0 = Hs[:B], cc.gumbel(keys[:B], V, "cuda")
            fc(h0, g0, cfg); base(h0, g0, cfg)            # autotune and warm up before capture
            gf = Graphed(lambda h, g: fc(h, g, cfg), h0, g0)
            gb = Graphed(lambda h, g: base(h, g, cfg), h0, g0)
            rec = dict(ms={k: [] for k in VARIANTS}, n8=[], nr=[], mism=0, mism_graph=0, mism_base=0,
                       rows_differ_across_reps=0)
            for it in range(-3, nb):
                i0 = max(it, 0) * B
                h, g = Hs[i0:i0 + B], cc.gumbel(keys[i0:i0 + B], V, "cuda")
                ref = ch.reference(h, g, cfg, cc)
                t = {k: [] for k in VARIANTS}
                first = None
                for rep in range(REPS):
                    tb, x = _timed(base, h, g, cfg); t["base"].append(x)
                    _, x = _timed(gb, h, g); t["base_graph"].append(x)
                    (tok, n8, nr), x = _timed(fc, h, g, cfg); t["fused"].append(x)
                    tok, n8, nr = tok.clone(), n8.clone(), nr.clone()
                    tg, x = _timed(gf, h, g); t["fused_graph"].append(x)
                    tg, n8g, nrg = (y.clone() for y in tg)
                    if it < 0:
                        continue
                    rec["mism"] += int((tok != ref).sum())
                    rec["mism_graph"] += int((tg != ref).sum())
                    rec["mism_base"] += int((tb != ref).sum())
                    cur = (n8.tolist(), nr.tolist(), n8g.tolist(), nrg.tolist())
                    if first is None:
                        first = cur
                    rec["rows_differ_across_reps"] += int(cur != first or cur[0] != cur[2] or cur[1] != cur[3])
                if it < 0:
                    continue
                for k in VARIANTS:
                    rec["ms"][k].append(t[k])
                rec["n8"].append(first[0]); rec["nr"].append(first[1])
            res[str(B)] = dict(batches=nb, states=nb * B, ms=rec["ms"], int8_rows=rec["n8"], exact_rows=rec["nr"],
                               mismatches=rec["mism"], mismatches_graph=rec["mism_graph"],
                               mismatches_bf16_baseline=rec["mism_base"],
                               rows_differ_across_reps=rec["rows_differ_across_reps"])
            fg = sorted(sorted(x)[REPS // 2] for x in rec["ms"]["fused_graph"])
            print(model, rn, B, "fused graphed ms: median", round(fg[len(fg) // 2], 3), "max", round(fg[-1], 3),
                  "| mismatches", rec["mism"], rec["mism_graph"], "| rows differ", rec["rows_differ_across_reps"],
                  flush=True)
            del gf, gb
        out["rules"][rn] = res
    return out


@app.local_entrypoint()
def main(models: str = "qwen3_8b,gemma4_e4b", gpu: str = "L4"):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for res in run.with_options(gpu=gpu).map(models.split(",")):
        path = os.path.join(HERE, "results", f"fused_tail_{res['model']}_{gpu}.json")
        json.dump(res, open(path, "w"))
        print("wrote", path, flush=True)
