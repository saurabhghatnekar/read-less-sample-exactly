"""Time the certified greedy step with the K2 correction on an L4 (l4_corr_protocol.json).

  modal run l4_corr_bench_modal.py --model qwen3_8b --n 64      # smoke
  modal run l4_corr_bench_modal.py                               # all three models, 512 states each
Writes logs/l4_corr_timing_<model>.json. States, baseline and timing are l4_bench_modal.py's.
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = os.path.join(HERE, "l4_corr_protocol.json")
app = modal.App("exp01-l4-corr")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "l4_kernels.py", "l4_cert.py", "l4_graph.py",
           "l4_corr.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

T_STAR = {"qwen3_8b": 0.5, "gemma4_e4b": 1.5, "sarvam_30b": 0.5}
CONFIGS = (None, "mu", "pca1")


def _stats(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return dict(median=float(np.median(x)), p99=float(np.percentile(x, 99)), mean=float(x.mean()),
                max=float(x.max()))


@app.function(image=image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=64 * 1024, timeout=2 * 3600)
def bench(model: str, n: int = 512, warm: int = 20):
    import numpy as np
    import torch

    import l4_corr
    import l4_graph
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    Wb = _load_head(MODELS[model]).to(dev)
    Pall = torch.load(f"{VOL}/hdir/{model}/P.pt")              # column 0: unit mean row, column 1: K2's pca1
    head = l4_corr.CorrHead(Wb, T_STAR[model], {"mu": Pall[:, 0], "pca1": Pall[:, 1]})
    V, D = Wb.shape
    H = torch.load(f"{VOL}/states/{model}/h_greedy.pt")
    g = torch.Generator().manual_seed(0)
    H = H[torch.randperm(len(H), generator=g)[:n + warm + 16 * 4]].to(dev)   # the states of l4_bench_modal.py
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

    res = dict(model=model, V=V, D=D, t=T_STAR[model], gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    for B in (1, 16):
        steps = [H[i:i + B] for i in range(warm * B if B == 1 else 0, (warm + n) if B == 1 else n, B)][: (n if B == 1 else n // B)]
        gs = {}
        for cfg in CONFIGS:
            head.use(cfg)
            for i in range(warm):                           # warm-up, autotune
                hb = H[(i * B) % (len(H) - B):][:B]
                torch.argmax(hb @ Wb.T, dim=1)
                head.step(hb)
            gs[cfg] = l4_graph.GraphStep(head, B) if cfg is None else l4_corr.GraphStepCorr(head, B)
            gs[cfg].capture()
            for i in range(warm):
                gs[cfg].step(H[(i * B) % (len(H) - B):][:B])
        base = []
        rec = {cfg: dict(graph=[], loop=[], g8=[], gR=[], l8=[], lR=[], fb=0, mism=0, gmism=0, outside=0, rowdiff=0,
                         loop_outside=0) for cfg in CONFIGS}
        for hb in steps:
            _, tb = timed(lambda: torch.argmax(hb @ Wb.T, dim=1))
            base.append(tb)
            ref = hb.float() @ Wf.T
            rt = ref.argmax(1)
            for cfg in CONFIGS:
                head.use(cfg)
                r = rec[cfg]
                (gtok, g8, gR, fb), tg = timed(lambda: gs[cfg].step(hb))
                (tok, l8, lR), tl_ = timed(lambda: head.step(hb))
                r["graph"].append(tg); r["loop"].append(tl_)
                r["g8"].append(g8); r["gR"].append(gR); r["l8"].append(l8); r["lR"].append(lR)
                r["fb"] += int(fb)
                r["gmism"] += int((gtok.to(dev) != rt).sum())
                r["mism"] += int((tok != rt).sum())
                if not fb:
                    Lb, Ub = gs[cfg].L[:B], gs[cfg].U[:B]
                    r["outside"] += int(((ref < Lb) | (ref > Ub)).sum())
                    r["rowdiff"] += int(g8 != l8) + int(gR != lR)
                if cfg is not None:                          # the eager corrected first pass, every row
                    Z, b, *_ = head.first_pass(hb)
                    r["loop_outside"] += int(((ref - Z).abs() > b).sum())
        base = np.array(base)
        out = dict(steps=len(steps), bf16_ms=_stats(base))
        for cfg in CONFIGS:
            head.use(cfg)
            r = rec[cfg]
            fpb = head.first_pass_bytes()
            bytes_ = [(fpb + a * (D + 4) + x * 2 * D) / (2 * V * D) for a, x in zip(r["g8"], r["gR"])]
            out["none" if cfg is None else cfg] = dict(
                graph_ms=_stats(r["graph"]), graph_over_bf16=_stats(np.array(r["graph"]) / base),
                loop_ms=_stats(r["loop"]), loop_over_bf16=_stats(np.array(r["loop"]) / base),
                rows_int8=_stats(r["g8"]), rows_bf16=_stats(r["gR"]), bytes_over_bf16=_stats(bytes_),
                first_pass_bytes_over_bf16=fpb / (2 * V * D),
                fallbacks=r["fb"], mismatches=r["mism"], graph_mismatches=r["gmism"],
                rows_outside_interval=r["outside"], eager_rows_outside_bound=r["loop_outside"],
                row_count_differences=r["rowdiff"])
        res[f"B{B}"] = out
    return res


@app.local_entrypoint()
def main(model: str = "qwen3_8b,gemma4_e4b,sarvam_30b", n: int = 512):
    sha = hashlib.sha256(open(PROTOCOL, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"l4_corr_protocol.json sha256 {sha[:12]} is not logged")
    for r in bench.starmap([(m, n) for m in model.split(",")]):
        r["protocol_sha"] = sha
        path = os.path.join(HERE, "logs", f"l4_corr_timing_{r['model']}{'' if n >= 512 else '_smoke'}.json")
        json.dump(r, open(path, "w"), indent=1)
        for B in ("B1", "B16"):
            x = r[B]
            for cfg in ("none", "mu", "pca1"):
                c = x[cfg]
                print(r["model"], B, cfg, "bf16 %.2f" % x["bf16_ms"]["median"],
                      "graph ms %.2f/%.2f/%.2f" % (c["graph_ms"]["median"], c["graph_ms"]["p99"], c["graph_ms"]["mean"]),
                      "ratio %.3f/%.3f/%.3f" % (c["graph_over_bf16"]["median"], c["graph_over_bf16"]["p99"], c["graph_over_bf16"]["mean"]),
                      "rows8 %.0f/%.0f" % (c["rows_int8"]["median"], c["rows_int8"]["p99"]),
                      "fb", c["fallbacks"], "mism", c["mismatches"], c["graph_mismatches"],
                      "outside", c["rows_outside_interval"], c["eager_rows_outside_bound"], "rowdiff", c["row_count_differences"])
