"""K2 on Modal: state-direction correction of the dithered first pass (hdir_rule.json).

  modal run hdir_modal.py::main --model qwen3_8b --smoke      # first 512 states per mode
  modal run hdir_modal.py::main --model sarvam_30b            # also qwen3_8b, gemma4_e4b
  modal run hdir_modal.py::robust --model sarvam_30b          # addendum: five rules, leave-one-language-out
  modal volume get exp01-repro hdir/sarvam_30b ./results/hdir/
  python3 -I hdir_summarize.py results/hdir/sarvam_30b "Sarvam-30B" 262144
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "hdir_rule.json")
app = modal.App("exp01-hdir")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "hdir_lib.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

T_STAR = {"qwen3_8b": 0.5, "gemma4_e4b": 1.5, "sarvam_30b": 0.5}   # the B1 greedy shifts
CHUNK = 4096                                                         # B1's job size, for control H0
RULES = {"greedy": ("greedy", dict(rule="greedy")), "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0))}


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3600)
def prep(model: str, rule_sha: str):
    """Directions from the head's mean row and from the greedy states of the even prompt ids."""
    import torch

    import a3_lib
    import hdir_lib
    from a3_modal import MODELS, _load_head

    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_greedy.pt")
    pid = torch.tensor([json.loads(l)["pid"] for l in open(f"{sdir}/rows_greedy.jsonl")])
    W = _load_head(MODELS[model]).cuda().float()
    mu64 = a3_lib.mean_row(W)
    Pall, info = hdir_lib.fit_directions(mu64, H[pid % 2 == 0].cuda().float())
    os.makedirs(f"{VOL}/hdir/{model}", exist_ok=True)
    torch.save(Pall.cpu(), f"{VOL}/hdir/{model}/P.pt")
    json.dump(dict(info, rule_sha=rule_sha), open(f"{VOL}/hdir/{model}/P.json", "w"))
    vol.commit()
    return info


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3 * 3600)
def chunk(model: str, mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import numpy as np
    import torch

    import a3_lib
    import hdir_lib
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    mu64 = a3_lib.mean_row(W)
    w_norm = W.norm(dim=1)
    Pall = torch.load(f"{VOL}/hdir/{model}/P.pt").cuda()
    P = hdir_lib.split(Pall)
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in RULES.items() if md == mode}
    out, stats = {}, None
    for t in (0.0, T_STAR[model]):
        vs = a3_lib.build(W, mu64, t)
        del vs["int4_g128"]
        Wt = a3_lib.shifted(W, mu64, t)
        rall = hdir_lib.err_proj(vs["int4d_g128"].Wq, Wt, Pall)
        del Wt
        torch.cuda.empty_cache()
        R = {n: (rall[:, :1] if n == "mu" else rall[:, 1:1 + P[n].shape[1]]) for n in P}
        for c0 in range(0, len(rows), 128):
            rec, st = hdir_lib.run_chunk(W, mu64, w_norm, vs, P, R, H[c0:c0 + 128].cuda().float(),
                                         rows[c0:c0 + 128], rules)
            for k, x in rec.items():
                out.setdefault(k, []).append(x)
            if stats is None:
                stats = st
            else:
                for f in ("viol", "mismatch"):
                    for k, v in st[f].items():
                        stats[f][k] = stats[f].get(k, 0) + v
                for k, v in st["viol_excess_max"].items():
                    stats["viol_excess_max"][k] = max(stats["viol_excess_max"].get(k, -1e30), v)
        del vs, rall, R
        torch.cuda.empty_cache()
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    n = len(rows)
    mine = [(r["pid"], r["group"], r["step"]) for r in rows]

    # control H0: with no correction, bytes and S are B1's
    f = f"{VOL}/subg/{model}/{mode}_{start:06d}"
    assert [tuple(r) for r in json.load(open(f + ".json"))["rows"]][:n] == mine, "state order differs from B1"
    b1 = np.load(f + ".npz")
    h0 = 0
    for t in (0.0, T_STAR[model]):
        for rn in rules:
            for q in ("S", "bytes"):
                h0 += int((b1[f"t{t:g}/int4d_g128/subg/casc/{rn}/{q}"][:n] != out[f"t{t:g}/none/{rn}/{q}"]).sum())
    stats["h0_diff"] = h0
    rdir = f"{VOL}/hdir/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: v.astype(np.float32) for k, v in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, t_star=T_STAR[model], rows=mine),
              open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    ts = f"t{T_STAR[model]:g}"
    rn = next(iter(rules))
    return dict(mode=mode, start=start, n=n, h0=h0, viol=sum(stats["viol"].values()),
                mism=sum(stats["mismatch"].values()),
                mean_bytes={k: round(float(out[f"{ts}/{k}/{rn}/bytes"].mean()), 4) for k in ("none", "mu", "pca16")},
                rho_mu=round(float(np.median(out[f"{ts}/mu/rho"])), 3))


ALL_RULES = {
    "greedy": ("greedy", dict(rule="greedy")),
    "gumbel_0.6": ("t1", dict(rule="gumbel", T=0.6)),
    "gumbel_1.0": ("t1", dict(rule="gumbel", T=1.0)),
    "top50": ("t1", dict(rule="topk", k=50, T=1.0)),
    "minp_0.05": ("t1", dict(rule="minp", p=0.05, T=1.0)),
}


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=3600)
def prep_lolo(model: str, rule_sha: str):
    """hdir_addendum_robust.json: for each language group, directions fitted on the other groups' greedy states."""
    import torch

    import a3_lib
    import hdir_lib
    from a3_modal import MODELS, _load_head

    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_greedy.pt").cuda().float()
    grp = [json.loads(l)["group"] for l in open(f"{sdir}/rows_greedy.jsonl")]
    mu64 = a3_lib.mean_row(_load_head(MODELS[model]).cuda().float())
    pc1 = torch.load(f"{VOL}/hdir/{model}/P.pt").cuda()[:, 1]
    out, info = {}, {}
    for gname in sorted(set(grp)):
        keep = torch.tensor([x != gname for x in grp], device="cuda")
        Pa, inf = hdir_lib.fit_directions(mu64, H[keep], kmax=4)
        out[gname] = Pa[:, 1:5].cpu()
        info[gname] = dict(n_cal=inf["n_cal"], share=inf["share"], cos_with_pca1=float(Pa[:, 1] @ pc1))
    torch.save(out, f"{VOL}/hdir/{model}/P_lolo.pt")
    json.dump(dict(info=info, rule_sha=rule_sha), open(f"{VOL}/hdir/{model}/P_lolo.json", "w"))
    vol.commit()
    return info


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=4 * 3600)
def chunk_robust(model: str, mode: str, start: int, stop: int, smoke: bool, rule_sha: str):
    import numpy as np
    import torch

    import a3_lib
    import hdir_lib
    from a3_modal import MODELS, _load_head

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MODELS[model]
    sdir = f"{VOL}/states/{model}"
    H = torch.load(f"{sdir}/h_{mode}.pt")[start:stop]
    rows = [json.loads(l) for l in open(f"{sdir}/rows_{mode}.jsonl")][start:stop]
    W = _load_head(cfg).cuda().float()
    mu64 = a3_lib.mean_row(W)
    w_norm = W.norm(dim=1)
    Pall = torch.load(f"{VOL}/hdir/{model}/P.pt").cuda()[:, :5]         # mu, pca1..4
    Pl = {k: x.cuda() for k, x in torch.load(f"{VOL}/hdir/{model}/P_lolo.pt").items()}
    gnames = sorted(Pl)
    Pcat = torch.cat([Pall] + [Pl[k] for k in gnames], 1)               # 5 + 4 per group columns
    rules = {k: dict(c, cap=cfg["cap"]) for k, (md, c) in ALL_RULES.items() if md == mode}
    out, stats = {}, None
    for t in (0.0, T_STAR[model]):
        vs = a3_lib.build(W, mu64, t)
        del vs["int4_g128"]
        Wt = a3_lib.shifted(W, mu64, t)
        r = hdir_lib.err_proj(vs["int4d_g128"].Wq, Wt, Pcat)
        del Wt
        torch.cuda.empty_cache()
        col = {k: slice(5 + 4 * i, 9 + 4 * i) for i, k in enumerate(gnames)}
        sets = {"mu": (Pall[:, :1], r[:, :1]), "pca1": (Pall[:, 1:2], r[:, 1:2]), "pca4": (Pall[:, 1:5], r[:, 1:5]),
                "lolo1": {k: (Pl[k][:, :1], r[:, col[k]][:, :1]) for k in gnames},
                "lolo4": {k: (Pl[k], r[:, col[k]]) for k in gnames}}
        for c0 in range(0, len(rows), 128):
            rec, st = hdir_lib.run_chunk_sets(W, mu64, w_norm, vs, sets, H[c0:c0 + 128].cuda().float(),
                                              rows[c0:c0 + 128], rules)
            for k, x in rec.items():
                out.setdefault(k, []).append(x)
            if stats is None:
                stats = st
            else:
                for f in ("viol", "mismatch"):
                    for k, v in st[f].items():
                        stats[f][k] = stats[f].get(k, 0) + v
                for k, v in st["viol_excess_max"].items():
                    stats["viol_excess_max"][k] = max(stats["viol_excess_max"].get(k, -1e30), v)
        del vs, r, sets
        torch.cuda.empty_cache()
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    n = len(rows)
    mine = [(r_["pid"], r_["group"], r_["step"]) for r_ in rows]
    f = f"{VOL}/subg/{model}/{mode}_{start:06d}"
    assert [tuple(x) for x in json.load(open(f + ".json"))["rows"]][:n] == mine, "state order differs from B1"
    b1 = np.load(f + ".npz")
    h0 = 0
    for t in (0.0, T_STAR[model]):
        for rn in rules:
            for qn in ("S", "bytes"):
                h0 += int((b1[f"t{t:g}/int4d_g128/subg/casc/{rn}/{qn}"][:n] != out[f"t{t:g}/none/{rn}/{qn}"]).sum())
    stats["h0_diff"] = h0
    rdir = f"{VOL}/hdir_robust/{model}{'_smoke' if smoke else ''}"
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{mode}_{start:06d}.npz", **{k: v.astype(np.float32) for k, v in out.items()})
    json.dump(dict(stats=stats, rule_sha=rule_sha, t_star=T_STAR[model], rows=mine),
              open(f"{rdir}/{mode}_{start:06d}.json", "w"))
    vol.commit()
    ts = f"t{T_STAR[model]:g}"
    return dict(mode=mode, start=start, n=n, h0=h0, viol=sum(stats["viol"].values()),
                mism=sum(stats["mismatch"].values()),
                mean_bytes={rn: {k: round(float(out[f"{ts}/{k}/{rn}/bytes"].mean()), 4) for k in ("none", "pca1", "lolo1")}
                            for rn in rules})


@app.local_entrypoint()
def robust(model: str = "qwen3_8b", smoke: bool = False):
    """modal run hdir_modal.py::robust --model sarvam_30b [--smoke]"""
    add = os.path.join(HERE, "hdir_addendum_robust.json")
    sha = hashlib.sha256(open(add, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"hdir_addendum_robust.json sha256 {sha[:12]} is not logged")
    print("lolo directions", json.dumps(prep_lolo.remote(model, sha)), flush=True)
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{model}/rows_{mode}.jsonl")).count(b"\n")
        n = min(n, 512) if smoke else n
        jobs += [(model, mode, s, min(s + CHUNK, n), smoke, sha) for s in range(0, n, CHUNK)]
    print(f"{len(jobs)} jobs, addendum {sha[:12]}")
    tot = dict(n=0, h0=0, viol=0, mism=0)
    for r in chunk_robust.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)


@app.local_entrypoint()
def main(model: str = "qwen3_8b", smoke: bool = False):
    sha = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    if sha not in open(os.path.join(HERE, "logs", "a3_decision_rule.log")).read():
        raise SystemExit(f"hdir_rule.json sha256 {sha[:12]} is not logged")
    print("directions", prep.remote(model, sha), flush=True)
    jobs = []
    for mode in ("greedy", "t1"):
        n = b"".join(vol.read_file(f"states/{model}/rows_{mode}.jsonl")).count(b"\n")
        n = min(n, 512) if smoke else n
        jobs += [(model, mode, s, min(s + CHUNK, n), smoke, sha) for s in range(0, n, CHUNK)]
    print(f"{len(jobs)} jobs, rule {sha[:12]}")
    tot = dict(n=0, h0=0, viol=0, mism=0)
    for r in chunk.starmap(jobs, order_outputs=False):
        for k in tot:
            tot[k] += r[k]
        print(r, flush=True)
    print("TOTAL", tot)
