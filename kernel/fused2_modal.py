"""Check and time the unified certified head (fused2.py). Protocol: unified_protocol.json.

  modal run fused2_modal.py [--models qwen3_8b,gemma4_e4b,sarvam_30b] [--gpu A100-80GB] [--smoke]
Writes results/fused2_<model>_<gpu>.json in the layout tail_summarize.py reads, plus the checks and the per-kernel
breakdown. Only numbers leave Modal.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "exp01_repro")
app = modal.App("kernel-fused2")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "huggingface_hub", "safetensors", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _d, _f in ((EXP, "cert_core.py"), (HERE, "cert_head.py"), (HERE, "cert_head2.py"), (HERE, "int4_pass1.py"),
               (HERE, "fused.py"), (HERE, "fused2.py"), (HERE, "common.py")):
    image = image.add_local_file(os.path.join(_d, _f), f"/root/{_f}")

CFG = {
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218", head="lm_head.weight", cap=None,
                     t=0.5, k2=False),
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                       head="model.language_model.embed_tokens.weight", cap=30.0, t=1.5, k2=False),
    "sarvam_30b": dict(repo="sarvamai/sarvam-30b", rev="071ae95e933605ca1104a6b4524a36a98488efa4", head="lm_head.weight",
                       cap=None, t=0.5, k2=True),
}
RULES = {"greedy": ("greedy", dict(rule="greedy")), "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0))}
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


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=2 * 3600, retries=0)
def run(model: str, smoke: bool = False):
    import torch
    import triton

    import cert_core as cc
    from cert_head import softcap
    from cert_head2 import U32, CertHead2
    from common import load_head
    from fused2 import FusedCert2, Graphed

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = dict(model=model, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), triton=str(triton.__version__),
               reps=REPS, checks={})

    def reference(W16, h, g, cfg):
        z = torch.empty(h.shape[0], W16.shape[0], device=h.device)
        for r0 in range(0, W16.shape[0], 16384):
            z[:, r0:r0 + 16384] = h @ W16[r0:r0 + 16384].float().T
        return z, cc.decide(z, cfg, g)

    # 1. synthetic head with a shift and a direction
    torch.manual_seed(0)
    Ws = (torch.randn(4133, 512, device="cuda") * 0.03 + 0.01 * torch.randn(512, device="cuda")).bfloat16()
    chs = CertHead2(Ws, t=0.5, p=torch.randn(512, device="cuda"), chunk=1000)
    fs = FusedCert2(chs)
    hs = (torch.randn(64, 512, device="cuda") * 2 + 6 * chs.p.float()[None, :]).bfloat16().float()
    gs = cc.gumbel(torch.arange(64, dtype=torch.int64, device="cuda") * 977, chs.V, "cuda")
    syn = {}
    for cap in (None, 30.0):
        for cfg in (dict(rule="greedy", cap=cap), dict(rule="gumbel", T=1.0, cap=cap), dict(rule="gumbel", T=0.6, cap=cap)):
            for B in (1, 3, 17, 64):
                _, ref = reference(Ws, hs[:B], gs[:B], cfg)
                tok = fs(hs[:B], gs[:B], cfg)[0]
                gr = Graphed(lambda h, g, c=cfg: fs(h, g, c), hs[:B], gs[:B])
                tg = gr(hs[:B], gs[:B])[0]
                syn[f"{cfg['rule']}_T{cfg.get('T')}_cap{cap}_B{B}"] = [int((tok != ref).sum()), int((tg != ref).sum())]
    out["checks"]["synthetic_mismatches_eager_graph"] = syn
    print("synthetic mismatches:", {k: v for k, v in syn.items() if any(v)} or "none", flush=True)
    del chs, fs, Ws
    torch.cuda.empty_cache()

    c = CFG[model]
    W = load_head(c).cuda()
    p = torch.load(f"{VOL}/hdir/{model}/P.pt")[:, 1].cuda() if c["k2"] else None
    ch = CertHead2(W, t=c["t"], p=p)
    del W
    torch.cuda.empty_cache()
    fc = FusedCert2(ch)
    V, d = ch.V, ch.d
    out.update(V=V, d=d, t=c["t"], k2=c["k2"])
    gen = torch.Generator().manual_seed(0)
    out["rules"], out["profile"] = {}, {}

    def base(h, g, cfg):
        cz = softcap((h.bfloat16() @ ch.W.T).float(), cfg["cap"])
        return (cz if cfg["rule"] == "greedy" else cz / cfg["T"] + g).argmax(1)

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

        # 2. intervals on 64 states, against the reference logits and the plain-PyTorch formula
        chk = dict(states=0, first_outside=0, int8_outside=0, logit_dev_over_bound_max=0.0, bound_ratio_min=1e9,
                   bound_ratio_max=0.0, int8_rows=0)
        for i0 in (0, 32):
            h, g = Hs[i0:i0 + 32], cc.gumbel(keys[i0:i0 + 32], V, "cuda")
            fc(h, g, cfg)
            bf = fc._buf[32]
            Z, BND = bf["Z"].clone(), bf["BND"].clone()
            cnt, LIST, L8, U8 = bf["CNT"].clone(), bf["LIST"].clone(), bf["L8"].clone(), bf["U8"].clone()
            zt, _ = reference(ch.W, h, g, cfg)
            chk["first_outside"] += int(((zt - Z).abs() > BND).sum())
            si = ch.state_inputs(h)
            hn = si["hn"][:, None]
            for r0 in range(0, V, 16384):
                r1 = min(r0 + 16384, V)
                wq = ch.dequant4(r0, r1)
                s = ch.s4[r0:r1].float().clamp_min(1e-12)
                zf = h @ wq.T + si["csh"][:, None]
                bfm = ch.C * torch.sqrt(si["h2sq"] @ (s ** 2).T) + ch.gamma * hn * (ch.w_norm[r0:r1] + ch.wq4_norm[r0:r1])[None, :] \
                    + 1.01 * U32 * hn * ch.wt_norm[None, r0:r1] + 2 * U32 * (zf.abs() + si["csh"].abs()[:, None])
                if ch.p is not None:
                    corr = si["alpha"][:, None] * ch.r4[None, r0:r1]
                    zf = zf - corr
                    bfm = bfm + 3 * U32 * si["alpha"].abs()[:, None] * ch.r4[None, r0:r1].abs() + 2 * U32 * corr.abs()
                ratio = BND[:, r0:r1] / bfm
                chk["logit_dev_over_bound_max"] = max(chk["logit_dev_over_bound_max"],
                                                      float(((Z[:, r0:r1] - zf).abs() / bfm).max()))
                chk["bound_ratio_min"] = min(chk["bound_ratio_min"], float(ratio.min()))
                chk["bound_ratio_max"] = max(chk["bound_ratio_max"], float(ratio.max()))
            for b in range(h.shape[0]):
                n = int(cnt[b])
                rws = LIST[b, :n].long()
                zb = zt[b, rws]
                chk["int8_outside"] += int(((zb < L8[b, :n]) | (zb > U8[b, :n])).sum())
                chk["int8_rows"] += n
            chk["states"] += h.shape[0]
        out["checks"][f"intervals_{rn}"] = chk
        print(model, rn, "intervals", json.dumps(chk), flush=True)

        # 3. tokens and timing
        res, prof_all = {}, {}
        for B, nb in BATCHES.items():
            nb = min(nb, len(pick) // B)
            h0, g0 = Hs[:B], cc.gumbel(keys[:B], V, "cuda")
            fc(h0, g0, cfg); base(h0, g0, cfg)
            gf = Graphed(lambda h, g: fc(h, g, cfg), h0, g0)
            gb = Graphed(lambda h, g: base(h, g, cfg), h0, g0)
            rec = dict(ms={k: [] for k in VARIANTS}, n8=[], nr=[], mism=0, mism_graph=0, mism_base=0, rows_differ=0)
            for it in range(-3, nb):
                i0 = max(it, 0) * B
                h, g = Hs[i0:i0 + B], cc.gumbel(keys[i0:i0 + B], V, "cuda")
                _, ref = reference(ch.W, h, g, cfg)
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
                    first = first or cur
                    rec["rows_differ"] += int(cur != first or cur[0] != cur[2] or cur[1] != cur[3])
                if it < 0:
                    continue
                for k in VARIANTS:
                    rec["ms"][k].append(t[k])
                rec["n8"].append(first[0]); rec["nr"].append(first[1])
            res[str(B)] = dict(batches=nb, states=nb * B, ms=rec["ms"], int8_rows=rec["n8"], exact_rows=rec["nr"],
                               mismatches=rec["mism"], mismatches_graph=rec["mism_graph"],
                               mismatches_bf16_baseline=rec["mism_base"], rows_differ_across_reps=rec["rows_differ"])
            # 4. per-kernel breakdown, eager with a sync between kernels
            prof, nprof = {}, min(nb, 16)
            for it in range(nprof):
                h, g = Hs[it * B:(it + 1) * B], cc.gumbel(keys[it * B:(it + 1) * B], V, "cuda")
                fc(h, g, cfg, prof=prof)
            prof_all[str(B)] = {k: v / nprof for k, v in prof.items()}
            fg = sorted(sorted(x)[REPS // 2] for x in rec["ms"]["fused_graph"])
            bg = sorted(sorted(x)[REPS // 2] for x in rec["ms"]["base_graph"])
            print(model, rn, B, "ms median: BF16", round(bg[len(bg) // 2], 3), "fused2", round(fg[len(fg) // 2], 3), "max",
                  round(fg[-1], 3), "| mismatches", rec["mism"], rec["mism_graph"], "| profile",
                  {k: round(v, 3) for k, v in prof_all[str(B)].items()}, flush=True)
            del gf, gb
        out["rules"][rn], out["profile"][rn] = res, prof_all
    return out


@app.local_entrypoint()
def main(models: str = "qwen3_8b,gemma4_e4b,sarvam_30b", gpu: str = "A100-80GB", smoke: bool = False):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for res in run.with_options(gpu=gpu).starmap([(m, smoke) for m in models.split(",")], order_outputs=False,
                                                 return_exceptions=True):
        if not isinstance(res, dict):
            print(f"FAILED {res!r}"[:1500], flush=True)
            continue
        path = os.path.join(HERE, "results", f"fused2_{res['model']}_{gpu}{'_smoke' if smoke else ''}.json")
        json.dump(res, open(path, "w"))
        print("wrote", path, flush=True)
