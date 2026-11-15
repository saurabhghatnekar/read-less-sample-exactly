"""Time the PyTorch certified head (cert_head.py) against the plain BF16 head on saved decode states.

  modal run baseline_modal.py --model gemma4_e4b [--gpu L4]
  modal run baseline_modal.py --model qwen3_8b

States come from the Exp01 Modal volume (greedy states for the greedy rule, T = 1 states for Gumbel T = 1),
512 per rule, shuffled with seed 0, so batches mix language groups. Every certified token is checked against the
FP32 reference. Writes results/baseline_<model>_<gpu>.json and prints a table. Only numbers leave Modal.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("kernel-baseline")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "huggingface_hub", "safetensors", "numpy")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_file(os.path.join(HERE, "..", "exp01_repro", "cert_core.py"), "/root/cert_core.py")
    .add_local_file(os.path.join(HERE, "cert_head.py"), "/root/cert_head.py")
)
MODELS = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218",
                     head="lm_head.weight", cap=None),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0),
}
RULES = {"greedy": ("greedy", dict(rule="greedy")), "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0))}
BATCHES = {1: 128, 8: 32, 32: 16}   # batch size -> number of batches timed


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


def _med(x):
    x = sorted(x)
    return x[len(x) // 2]


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=0)
def run(model: str, n_states: int = 512):
    import time

    import torch

    import cert_core as cc
    from cert_head import CertHead

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sync = torch.cuda.synchronize
    cfg_m = MODELS[model]
    W = _load_head(cfg_m).cuda()
    ch = CertHead(W)
    del W
    torch.cuda.empty_cache()
    V, d = ch.V, ch.d
    out = dict(model=model, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), V=V, d=d,
               bytes=dict(bf16_head=2 * V * d, pass1=ch.bytes_pass1(), int8_row=ch.bytes_int8_row(),
                          bf16_row=ch.bytes_bf16_row()),
               mem_gb=torch.cuda.max_memory_allocated() / 1e9, rules={})
    gen = torch.Generator().manual_seed(0)
    for rn, (mode, rcfg) in RULES.items():
        cfg = dict(rcfg, cap=cfg_m["cap"])
        sdir = f"{VOL}/states/{model}"
        H = torch.load(f"{sdir}/h_{mode}.pt")
        rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")]
        pick = torch.randperm(len(rows), generator=gen)[:n_states].tolist()
        Hs = H[pick].cuda().float()
        keys = torch.tensor([cc.state_key(rows[i]["pid"], rows[i]["step"], rows[i]["mode"]) for i in pick],
                            dtype=torch.int64, device="cuda")
        groups = [rows[i]["group"] for i in pick]
        del H
        res_r = {}
        for B, nb in BATCHES.items():
            nb = min(nb, n_states // B)
            rec = dict(base=[], matmul=[], noise=[], cert=[], S=[], R=[], mism_cert=0, mism_base=0, groups=[],
                       **{k: [] for k in ("pass1_read", "pass1_bound", "int8", "bf16")})
            for it in range(-3, nb):                               # 3 warm-up batches, not recorded
                i0 = (max(it, 0) * B) % n_states
                h, k = Hs[i0:i0 + B], keys[i0:i0 + B]
                sync(); t0 = time.perf_counter()
                g = cc.gumbel(k, V, "cuda")
                sync(); t1 = time.perf_counter()
                tb = ch.baseline(h, g, cfg)
                sync(); t2 = time.perf_counter()
                zb = h.bfloat16() @ ch.W.T
                sync(); t3 = time.perf_counter()
                tok, S, R, st = ch.certify(h, g, cfg, cc, sync)
                sync(); t4 = time.perf_counter()
                if it < 0:
                    continue
                ref = ch.reference(h, g, cfg, cc)
                rec["mism_cert"] += int((tok != ref).sum())
                rec["mism_base"] += int((tb != ref).sum())
                rec["noise"].append(t1 - t0); rec["base"].append(t2 - t1); rec["matmul"].append(t3 - t2)
                rec["cert"].append(t4 - t3); rec["S"].append(S); rec["R"].append(R)
                for kk, vv in st.items():
                    rec[kk].append(vv)
                if B == 1:
                    rec["groups"].append(groups[i0])
                del zb
            ms = lambda key: 1e3 * _med(rec[key])
            bw = out["bytes"]["bf16_head"] / _med(rec["matmul"])
            proj_p1 = out["bytes"]["pass1"] / bw
            rest = _med(rec["pass1_bound"]) + _med(rec["int8"]) + _med(rec["bf16"])
            summ = dict(
                batches=nb, states=nb * B,
                ms={k: ms(k) for k in ("noise", "base", "matmul", "cert", "pass1_read", "pass1_bound", "int8", "bf16")},
                bf16_matmul_GBps=bw / 1e9,
                projected_pass1_ms=1e3 * proj_p1,
                projected_cert_ms=1e3 * (proj_p1 + rest),
                pass1_budget_ms=1e3 * (_med(rec["base"]) - rest),
                union_rows_int8=dict(median=_med(rec["S"]), max=max(rec["S"])),
                union_rows_bf16=dict(median=_med(rec["R"]), max=max(rec["R"])),
                mismatches_certified=rec["mism_cert"], mismatches_bf16_baseline=rec["mism_base"],
            )
            if B == 1:
                byg = {}
                for gname, s_, r_, c_ in zip(rec["groups"], rec["S"], rec["R"], rec["cert"]):
                    byg.setdefault(gname, []).append((s_, r_, c_))
                summ["by_group_b1"] = {gname: dict(n=len(v), S_median=_med([x[0] for x in v]),
                                                   R_median=_med([x[1] for x in v]),
                                                   cert_ms_median=1e3 * _med([x[2] for x in v]))
                                       for gname, v in sorted(byg.items())}
            res_r[str(B)] = summ
            print(model, rn, B, json.dumps(summ["ms"]), flush=True)
        out["rules"][rn] = res_r
    return out


@app.local_entrypoint()
def main(model: str = "gemma4_e4b", gpu: str = "L4"):
    res = run.with_options(gpu=gpu).remote(model)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"baseline_{model}_{gpu}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)
    print(f"{res['model']} on {res['gpu']}, torch {res['torch']}, V={res['V']}, d={res['d']}")
    hdr = "rule        B  base   matmul  cert   p1read p1bound int8  bf16  | S med  R med | proj p1  proj cert  p1 budget | mism cert/base"
    print(hdr)
    for rn, byb in res["rules"].items():
        for B, s in byb.items():
            m = s["ms"]
            print(f"{rn:10s} {B:>2s} {m['base']:6.2f} {m['matmul']:6.2f} {m['cert']:6.2f} {m['pass1_read']:6.2f} "
                  f"{m['pass1_bound']:6.2f} {m['int8']:5.2f} {m['bf16']:5.2f} | {s['union_rows_int8']['median']:5d} "
                  f"{s['union_rows_bf16']['median']:5d} | {s['projected_pass1_ms']:7.2f} {s['projected_cert_ms']:9.2f} "
                  f"{s['pass1_budget_ms']:10.2f} | {s['mismatches_certified']}/{s['mismatches_bf16_baseline']}")
