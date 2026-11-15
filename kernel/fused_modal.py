"""Check and time the fused certified head (fused.py) on a Modal GPU.

  modal run fused_modal.py --model gemma4_e4b [--gpu L4]

1. Synthetic head: tokens against cert_core's FP32 reference (soft-cap on and off; greedy, Gumbel T = 1 and 0.6;
   batch 1, 3, 17, 64), eager and CUDA-graphed.
2. Real head, 512 saved states per rule (same selection as baseline_modal / pass1_modal): every token against the
   FP32 reference, eager and graphed; per-state INT8 and exact row counts.
3. Timing with CUDA events per batch (median): BF16 head eager and graphed, fused head eager and graphed.
   Gumbel noise is generated outside the timed region for both.
Writes results/fused_<model>_<gpu>.json. Only numbers leave Modal.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("kernel-fused")
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
BATCHES = {1: 128, 8: 32, 32: 16}


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
    from common import MODELS, RULES, load_head, med
    from fused import FusedCert, Graphed

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = dict(model=model, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
               triton=str(triton.__version__))

    # 1. synthetic
    torch.manual_seed(0)
    Ws = (torch.randn(4133, 512, device="cuda") * 0.03).bfloat16()
    chs = CertHead(Ws, chunk=1000)
    fs = FusedCert(chs)
    hs = (torch.randn(64, 512, device="cuda") * 2).bfloat16().float()
    ks = torch.arange(64, dtype=torch.int64, device="cuda") * 977
    gs = cc.gumbel(ks, chs.V, "cuda")
    syn = {}
    for cap in (None, 30.0):
        for cfg in (dict(rule="greedy", cap=cap), dict(rule="gumbel", T=1.0, cap=cap), dict(rule="gumbel", T=0.6, cap=cap)):
            for B in (1, 3, 17, 64):
                ref = chs.reference(hs[:B], gs[:B], cfg, cc)
                tok = fs(hs[:B], gs[:B], cfg)[0]
                gr = Graphed(lambda h, g, c=cfg: fs(h, g, c), hs[:B], gs[:B])
                tg = gr(hs[:B], gs[:B])[0]
                syn[f"{cfg['rule']}_T{cfg.get('T')}_cap{cap}_B{B}"] = [int((tok != ref).sum()), int((tg != ref).sum())]
    out["synthetic_mismatches_eager_graph"] = syn
    print("synthetic", syn, flush=True)
    del chs, fs, Ws

    cfg_m = MODELS[model]
    W = load_head(cfg_m).cuda()
    ch = CertHead(W)
    del W
    torch.cuda.empty_cache()
    fc = FusedCert(ch)
    V = ch.V
    out.update(V=V, d=ch.d)
    gen = torch.Generator().manual_seed(0)
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
            rec = dict(base=[], base_graph=[], fused=[], fused_graph=[], n8=[], nr=[], mism=0, mism_graph=0,
                       mism_base=0)
            for it in range(-3, nb):
                i0 = max(it, 0) * B
                h, g = Hs[i0:i0 + B], cc.gumbel(keys[i0:i0 + B], V, "cuda")
                tb, t_b = _timed(base, h, g, cfg)
                _, t_bg = _timed(gb, h, g)
                (tok, n8, nr), t_f = _timed(fc, h, g, cfg)
                tok, n8, nr = tok.clone(), n8.clone(), nr.clone()
                tg, t_fg = _timed(gf, h, g)
                tg = tg[0].clone()
                if it < 0:
                    continue
                ref = ch.reference(h, g, cfg, cc)
                rec["mism"] += int((tok != ref).sum())
                rec["mism_graph"] += int((tg != ref).sum())
                rec["mism_base"] += int((tb != ref).sum())
                rec["base"].append(t_b); rec["base_graph"].append(t_bg)
                rec["fused"].append(t_f); rec["fused_graph"].append(t_fg)
                rec["n8"] += n8.tolist(); rec["nr"] += nr.tolist()
            s8 = sorted(rec["n8"])
            res[str(B)] = dict(
                states=nb * B,
                ms={k: med(rec[k]) for k in ("base", "base_graph", "fused", "fused_graph")},
                int8_rows_per_state=dict(median=med(s8), p99=s8[int(0.99 * (len(s8) - 1))], max=s8[-1]),
                exact_rows_per_state=dict(median=med(rec["nr"]), max=max(rec["nr"])),
                mismatches=rec["mism"], mismatches_graph=rec["mism_graph"], mismatches_bf16_baseline=rec["mism_base"])
            print(model, rn, B, json.dumps(res[str(B)]), flush=True)
            del gf, gb
        out["rules"][rn] = res
    return out


@app.local_entrypoint()
def main(model: str = "gemma4_e4b", gpu: str = "L4"):
    res = run.with_options(gpu=gpu).remote(model)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"fused_{model}_{gpu}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)
    bad = {k: v for k, v in res["synthetic_mismatches_eager_graph"].items() if any(v)}
    print(f"{res['model']} on {res['gpu']}; synthetic mismatches: {bad or 'none'}")
    print("rule        B | BF16 eager graph | fused eager graph | INT8 rows/state med p99 | exact med max | mism eager/graph/bf16")
    for rn, r in res["rules"].items():
        for B in ("1", "8", "32"):
            s = r[B]; m = s["ms"]; i8 = s["int8_rows_per_state"]; ex = s["exact_rows_per_state"]
            print(f"{rn:10s} {B:>2s} | {m['base']:6.2f} {m['base_graph']:6.2f} | {m['fused']:6.2f} {m['fused_graph']:6.2f} | "
                  f"{i8['median']:6d} {i8['p99']:6d} | {ex['median']:4d} {ex['max']:4d} | "
                  f"{s['mismatches']}/{s['mismatches_graph']}/{s['mismatches_bf16_baseline']} of {s['states']}")
