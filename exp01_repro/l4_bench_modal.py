"""Time the certified greedy head on an L4 (l4_protocol.json).

  modal run l4_bench_modal.py --model qwen3_8b --n 64       # smoke
  modal run l4_bench_modal.py                                # all three models, 512 states each
Writes logs/l4_timing_<model>.json.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("exp01-l4")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "l4_kernels.py", "l4_cert.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

T_STAR = {"qwen3_8b": 0.5, "gemma4_e4b": 1.5, "sarvam_30b": 0.5}


def _stats(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return dict(median=float(np.median(x)), p99=float(np.percentile(x, 99)), mean=float(x.mean()))


@app.function(image=image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=64 * 1024, timeout=2 * 3600)
def bench(model: str, n: int = 512, warm: int = 20):
    import numpy as np
    import torch

    import l4_cert
    import l4_kernels as K
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    Wb = _load_head(MODELS[model]).to(dev)
    head = l4_cert.CertHead(Wb, T_STAR[model])
    V, D = Wb.shape
    H = torch.load(f"{VOL}/states/{model}/h_greedy.pt")
    g = torch.Generator().manual_seed(0)
    H = H[torch.randperm(len(H), generator=g)[:n + warm + 16 * 4]].to(dev)
    Wf = Wb.float()                                         # reference only, outside the timed region

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def timed(fn):
        a, b = ev(), ev()
        a.record()
        out = fn()
        b.record()
        torch.cuda.synchronize()
        return out, a.elapsed_time(b)

    res = dict(model=model, V=V, D=D, t=T_STAR[model], gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               first_pass_bytes_over_bf16=head.first_pass_bytes() / (2 * V * D))
    for B in (1, 16):
        steps = [H[i:i + B] for i in range(warm * B if B == 1 else 0, (warm + n) if B == 1 else n, B)][: (n if B == 1 else n // B)]
        for i in range(warm):                               # warm-up, autotune
            hb = H[i * B % len(H): i * B % len(H) + B]
            torch.argmax(hb @ Wb.T, dim=1)
            head.step(hb)
        base, cert, kern, n8s, nRs, mism = [], [], [], [], [], []
        for hb in steps:
            _, tb = timed(lambda: torch.argmax(hb @ Wb.T, dim=1))
            (tok, n8, nR), tc = timed(lambda: head.step(hb))
            H16 = head._pad(hb)
            z = torch.zeros(16, head.NG, device=dev)
            _, tk = timed(lambda: K.int4_head(H16, z, z, head.Q, head.S, torch.zeros(16, device=dev)))
            ref = (hb.float() @ Wf.T)
            rt = ref.argmax(1)
            for j in range(B):
                if tok[j] != rt[j]:
                    top2 = ref[j].topk(2).values
                    mism.append(dict(margin=float(top2[0] - top2[1])))
            base.append(tb); cert.append(tc); kern.append(tk); n8s.append(n8); nRs.append(nR)
        bytes_ = [(head.first_pass_bytes() + a * (D + 4) + r * 2 * D) / (2 * V * D) for a, r in zip(n8s, nRs)]
        ratio = np.array(cert) / np.array(base)
        res[f"B{B}"] = dict(steps=len(steps), bf16_ms=_stats(base), cert_ms=_stats(cert), kernel_ms=_stats(kern),
                            cert_over_bf16=_stats(ratio), kernel_over_bf16=float(np.median(kern) / np.median(base)),
                            bytes_over_bf16=_stats(bytes_), rows_int8=_stats(n8s), rows_bf16=_stats(nRs),
                            bf16_GBps=float(2 * V * D / (np.median(base) * 1e6)),
                            kernel_GBps=float(head.first_pass_bytes() / (np.median(kern) * 1e6)),
                            mismatches=len(mism), mismatch_detail=mism[:10])
    return res


@app.local_entrypoint()
def main(model: str = "qwen3_8b,gemma4_e4b,sarvam_30b", n: int = 512):
    for r in bench.starmap([(m, n) for m in model.split(",")]):
        path = os.path.join(HERE, "logs", f"l4_timing_{r['model']}{'' if n >= 512 else '_smoke'}.json")
        json.dump(r, open(path, "w"), indent=1)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in
              ("bf16_ms", "cert_ms", "kernel_ms", "cert_over_bf16", "kernel_over_bf16", "bytes_over_bf16",
               "bf16_GBps", "kernel_GBps", "mismatches") if kk in v}) for k, v in r.items()}, indent=1))
