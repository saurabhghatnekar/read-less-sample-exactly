"""Check and time the Triton pass-1 kernel (int4_pass1.py) on a Modal GPU.

  modal run pass1_modal.py --model gemma4_e4b [--gpu L4]

1. Synthetic: kernel against the PyTorch pass 1 (cert_head.CertHead.pass1/bound1), odd V, batch 1 to 32.
2. Real head: the same comparison on saved states; the kernel's z~ must differ from PyTorch's by far less than the
   FP32 slack (both are FP32 sums of the same products, in different orders).
3. Exactness: the full certified head with the kernel as pass 1, 512 states per rule, against the FP32 reference.
4. Timing: kernel and cuBLAS BF16 head (triton do_bench, median), and the whole certified head per batch.
Writes results/pass1_<model>_<gpu>.json. Only numbers leave Modal.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("kernel-pass1")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "huggingface_hub", "safetensors", "numpy")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_file(os.path.join(HERE, "..", "exp01_repro", "cert_core.py"), "/root/cert_core.py")
    .add_local_file(os.path.join(HERE, "cert_head.py"), "/root/cert_head.py")
    .add_local_file(os.path.join(HERE, "int4_pass1.py"), "/root/int4_pass1.py")
    .add_local_file(os.path.join(HERE, "common.py"), "/root/common.py")
)
BATCHES = {1: 128, 8: 32, 32: 16}


def _compare(ch, p1, h):
    """z~ difference against PyTorch's FP32 pass 1, in units of the PyTorch (cert_core) slack; bound difference."""
    import torch
    zt, bt = ch.pass1(h), ch.bound1(h)
    zk, bk = p1(h)
    slack = ch.gamma * h.norm(dim=1, keepdim=True) * (ch.w_norm + ch.wq4_norm)
    return dict(max_dz_over_slack=float(((zk - zt).abs() / slack).max()),
                max_rel_db=float(((bk - bt).abs() / bt).max()), min_db_over_slack=float(((bk - bt) / slack).min()),
                finite=bool(torch.isfinite(zk).all() and torch.isfinite(bk).all()))


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=0)
def run(model: str, n_states: int = 512):
    import time

    import torch
    import triton

    import cert_core as cc
    from cert_head import CertHead
    from common import MODELS, RULES, load_head, med
    from int4_pass1 import Pass1

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sync = torch.cuda.synchronize
    out = dict(model=model, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
               triton=str(triton.__version__))

    # 1. synthetic
    torch.manual_seed(0)
    Ws = (torch.randn(4133, 512, device="cuda") * 0.03).bfloat16()
    chs = CertHead(Ws, chunk=1000)
    p1s, p1sf = Pass1(chs), Pass1(chs, "fp32")
    hs = (torch.randn(64, 512, device="cuda") * 2).bfloat16().float()
    out["synthetic"] = {str(B): _compare(chs, p1s, hs[:B]) for B in (1, 2, 3, 8, 17, 32, 64)}
    out["synthetic"]["fp32_1"] = _compare(chs, p1sf, hs[:1])
    print("synthetic", out["synthetic"], flush=True)
    del Ws, chs, p1s

    cfg_m = MODELS[model]
    W = load_head(cfg_m).cuda()
    ch = CertHead(W)
    del W
    torch.cuda.empty_cache()
    p1, p1f = Pass1(ch), Pass1(ch, "fp32")
    V, d = ch.V, ch.d
    out.update(V=V, d=d, bytes=dict(bf16_head=2 * V * d, pass1=ch.bytes_pass1()))
    gen = torch.Generator().manual_seed(0)
    out["rules"] = {}
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
        res = {"compare": {str(B): _compare(ch, p1, Hs[:B]) for B in (1, 8, 32)}}
        res["compare"]["fp32_1"] = _compare(ch, p1f, Hs[:1])
        res["fp32_kernel_b1_ms"] = triton.testing.do_bench(lambda: p1f(Hs[:1]), warmup=50, rep=300,
                                                           return_mode="median")
        print(model, rn, "compare", res["compare"], flush=True)
        for B, nb in BATCHES.items():
            h = Hs[:B]
            t_k = triton.testing.do_bench(lambda: p1(h), warmup=50, rep=300, return_mode="median")
            t_b = triton.testing.do_bench(lambda: h.bfloat16() @ ch.W.T, warmup=50, rep=300, return_mode="median")
            rec = dict(cert=[], mism=0, **{k: [] for k in ("pass1_read", "pass1_bound", "int8", "bf16")})
            for it in range(-3, nb):
                i0 = max(it, 0) * B
                hb, kb = Hs[i0:i0 + B], keys[i0:i0 + B]
                g = cc.gumbel(kb, V, "cuda")
                sync(); t0 = time.perf_counter()
                tok, S, R, st = ch.certify(hb, g, cfg, cc, sync, p1=p1)
                sync(); t1 = time.perf_counter()
                if it < 0:
                    continue
                rec["mism"] += int((tok != ch.reference(hb, g, cfg, cc)).sum())
                rec["cert"].append(t1 - t0)
                for k, v in st.items():
                    rec[k].append(v)
            ms = {k: 1e3 * med(rec[k]) for k in ("cert", "pass1_read", "pass1_bound", "int8", "bf16")}
            res[str(B)] = dict(kernel_ms=t_k, bf16_head_ms=t_b,
                               kernel_GBps=out["bytes"]["pass1"] / t_k / 1e6,
                               bf16_GBps=out["bytes"]["bf16_head"] / t_b / 1e6,
                               certified_ms=ms, states=nb * B, mismatches=rec["mism"])
            print(model, rn, B, json.dumps(res[str(B)]), flush=True)
        out["rules"][rn] = res
    return out


@app.local_entrypoint()
def main(model: str = "gemma4_e4b", gpu: str = "L4"):
    res = run.with_options(gpu=gpu).remote(model)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"pass1_{model}_{gpu}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)
    print(f"{res['model']} on {res['gpu']}, torch {res['torch']}, triton {res['triton']}")
    for rn, r in res["rules"].items():
        print(rn, "FP32 batch-1 kernel ms", round(r["fp32_kernel_b1_ms"], 3), "compare", r["compare"])
    print("rule        B  kernel ms  GB/s | BF16 head ms  GB/s | certified ms (p1 / int8 / bf16) | mismatches")
    for rn, r in res["rules"].items():
        for B in ("1", "8", "32"):
            s = r[B]; c = s["certified_ms"]
            print(f"{rn:10s} {B:>2s} {s['kernel_ms']:9.3f} {s['kernel_GBps']:5.0f} | {s['bf16_head_ms']:12.3f} "
                  f"{s['bf16_GBps']:5.0f} | {c['cert']:6.2f} ({c['pass1_read']:.2f} / {c['int8']:.2f} / {c['bf16']:.2f})"
                  f" | {s['mismatches']}/{s['states']}")
