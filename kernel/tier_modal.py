"""Time the memory-tier layouts of the unified head (tiered.py). Protocol: tier_protocol.json.

  modal run tier_modal.py [--models qwen3_8b,gemma4_e4b,sarvam_30b,qwen3_0.6b] [--gpu A100-80GB] [--smoke]
Writes results/tier_<model>_<gpu>.json. Only numbers leave Modal.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "exp01_repro")
app = modal.App("kernel-tier")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "huggingface_hub", "safetensors", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _d, _f in ((EXP, "cert_core.py"), (HERE, "cert_head.py"), (HERE, "cert_head2.py"), (HERE, "int4_pass1.py"),
               (HERE, "fused.py"), (HERE, "fused2.py"), (HERE, "tiered.py"), (HERE, "common.py")):
    image = image.add_local_file(os.path.join(_d, _f), f"/root/{_f}")

CFG = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218", head="lm_head.weight", cap=None,
                     t=0.5, k2=False),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0, t=1.5, k2=False),
    "sarvam_30b": dict(repo="sarvamai/sarvam-30b", rev="071ae95e933605ca1104a6b4524a36a98488efa4", head="lm_head.weight",
                       cap=None, t=0.5, k2=True),
    "qwen3_0.6b": dict(saved_head=True, cap=None, t=0.5, k2=False),
}
RULES = {"greedy": ("greedy", dict(rule="greedy")), "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0))}
BATCHES = {1: 512, 8: 64}
REPS = 3
VARIANTS = ("stock", "stock_graph", "all", "B", "A")


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", cpu=8, memory=48 * 1024, timeout=2 * 3600, retries=0)
def run(model: str, smoke: bool = False):
    import time

    import torch

    import cert_core as cc
    from cert_head import softcap
    from cert_head2 import CertHead2
    from common import load_head
    from fused2 import FusedCert2, Graphed
    from tiered import Tiered

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = CFG[model]
    W = (torch.load(f"{VOL}/states/{model}/W.pt") if c.get("saved_head") else load_head(c)).cuda()
    p = torch.load(f"{VOL}/hdir/{model}/P.pt")[:, 1].cuda() if c["k2"] else None
    ch = CertHead2(W, t=c["t"], p=p)
    del W
    V, d = ch.V, ch.d
    fc, tb, ta = FusedCert2(ch), Tiered(ch, "B"), Tiered(ch, "A")
    head = 2 * V * d
    full = tb.gpu_bytes() + head
    out = dict(model=model, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), V=V, d=d, reps=REPS,
               bytes=dict(bf16_head=head, layout_all=full, layout_B=tb.gpu_bytes(), layout_A=ta.gpu_bytes(),
                          work_buffers_per_state=20 * V), rules={})

    def base(h, g, cfg):
        cz = softcap((h.bfloat16() @ ch.W.T).float(), cfg["cap"])
        return (cz if cfg["rule"] == "greedy" else cz / cfg["T"] + g).argmax(1)

    def wall(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tok = fn().cpu()
        return tok, (time.perf_counter() - t0) * 1e3

    gen = torch.Generator().manual_seed(0)
    for rn, (mode, rcfg) in RULES.items():
        cfg = dict(rcfg, cap=c["cap"])
        sdir = f"{VOL}/states/{model}"
        H = torch.load(f"{sdir}/h_{mode}.pt")
        rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")]
        pick = torch.randperm(len(rows), generator=gen)[:(64 if smoke else 512)].tolist()
        Hs = H[pick].cuda().float()
        keys = torch.tensor([cc.state_key(rows[i]["pid"], rows[i]["step"], rows[i]["mode"]) for i in pick],
                            dtype=torch.int64, device="cuda")
        del H
        res = {}
        for B, nb in BATCHES.items():
            nb = min(nb, len(pick) // B)
            h0, g0 = Hs[:B], cc.gumbel(keys[:B], V, "cuda")
            fc(h0, g0, cfg); base(h0, g0, cfg); tb(h0, g0, cfg); ta(h0, g0, cfg)
            gf = Graphed(lambda h, g: fc(h, g, cfg), h0, g0)
            gb = Graphed(lambda h, g: base(h, g, cfg), h0, g0)
            rec = dict(ms={k: [] for k in VARIANTS}, mism={k: 0 for k in VARIANTS}, fetch_exact=[], fetch_int8=[],
                       rows_differ=0)
            for it in range(-4, nb):
                i0 = max(it, 0) * B
                h, g = Hs[i0:i0 + B], cc.gumbel(keys[i0:i0 + B], V, "cuda")
                z = torch.empty(B, V, device="cuda")
                for r0 in range(0, V, 16384):
                    z[:, r0:r0 + 16384] = h @ ch.W[r0:r0 + 16384].float().T
                ref = cc.decide(z, cfg, g).cpu()
                t = {k: [] for k in VARIANTS}
                tk = {}
                for rep in range(REPS):
                    tk["stock"], x = wall(lambda: base(h, g, cfg)); t["stock"].append(x)
                    tk["stock_graph"], x = wall(lambda: gb(h, g)); t["stock_graph"].append(x)
                    tk["all"], x = wall(lambda: gf(h, g)[0]); t["all"].append(x)
                    n8_all, nr_all = gf.out[1].tolist(), gf.out[2].tolist()
                    tk["B"], x = wall(lambda: tb(h, g, cfg)); t["B"].append(x)
                    tk["A"], x = wall(lambda: ta(h, g, cfg)); t["A"].append(x)
                    if it >= 0:
                        for k in VARIANTS:
                            rec["mism"][k] += int((tk[k] != ref).sum())
                        rec["rows_differ"] += int(tb.last["exact_rows"] != nr_all or ta.last["exact_rows"] != nr_all
                                                  or ta.last["int8_rows"] != n8_all)
                if it < 0:
                    continue
                for k in VARIANTS:
                    rec["ms"][k].append(sorted(t[k])[REPS // 2])
                rec["fetch_exact"].append(sum(tb.last["exact_rows"]))
                rec["fetch_int8"].append(sum(ta.last["int8_rows"]))
            res[str(B)] = dict(batches=nb, ms=rec["ms"], mismatches=rec["mism"], rows_differ=rec["rows_differ"],
                               slow_steps=dict(B=tb.slow_steps, A=ta.slow_steps),
                               exact_rows_fetched=rec["fetch_exact"], int8_rows_fetched=rec["fetch_int8"])
            med = {k: sorted(v)[len(v) // 2] for k, v in rec["ms"].items()}
            print(model, rn, B, "ms median", {k: round(v, 3) for k, v in med.items()}, "| mismatches", rec["mism"],
                  "| rows differ", rec["rows_differ"], "| fetched per step, median: exact",
                  sorted(rec["fetch_exact"])[nb // 2], "int8", sorted(rec["fetch_int8"])[nb // 2], flush=True)
            del gf, gb
        out["rules"][rn] = res
    return out


@app.local_entrypoint()
def main(models: str = "qwen3_8b,gemma4_e4b,sarvam_30b,qwen3_0.6b", gpu: str = "A100-80GB", smoke: bool = False):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for res in run.with_options(gpu=gpu).starmap([(m, smoke) for m in models.split(",")], order_outputs=False,
                                                 return_exceptions=True):
        if not isinstance(res, dict):
            print(f"FAILED {res!r}"[:1500], flush=True)
            continue
        path = os.path.join(HERE, "results", f"tier_{res['model']}_{gpu}{'_smoke' if smoke else ''}.json")
        json.dump(res, open(path, "w"))
        b = res["bytes"]
        print("wrote", path, "| GPU bytes / BF16 head: all %.3f, B %.3f, A %.3f" % (
            b["layout_all"] / b["bf16_head"], b["layout_B"] / b["bf16_head"], b["layout_A"] / b["bf16_head"]), flush=True)
