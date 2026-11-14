# %% [markdown]
# # Mean row vs typical row in three LM heads
#
# Task from the project supervisor (week of 9 Nov 2026): load **only the head weights** of Gemma 4 E4B,
# Qwen3-8B and Sarvam-30B and compare the size of the vocabulary-mean row
# $\mu = \frac{1}{V} W^\top \mathbf{1}$ with a typical row.
#
# Why it matters (Kadav et al., arXiv:2609.31291): every logit contains the shared term $\mu^\top h$.
# Subtracting $t\mu$ from every row leaves softmax (and greedy / Gumbel / top-k / min-p decisions)
# unchanged, but changes the INT4 group scales and hence our certificate widths.
# Sarvam-30B's top logit is ~200 against 25-38 for the other two; a large $\mu$ is one explanation.
#
# The notebook reads only the head tensor out of the safetensors file with HTTP range requests,
# so no full checkpoint is downloaded. Disk needed: ~1.3 GB (Gemma), 1.2 GB (Qwen), 4.3 GB (Sarvam, stored as F32).
# Runs on a free Colab CPU runtime; no GPU needed.

# %%
import json, os, struct, time
import numpy as np
import requests

OUTDIR = os.environ.get("HEAD_OUTDIR", "/content/heads")
os.makedirs(OUTDIR, exist_ok=True)

# Revisions: the lab note used Gemma 4ecc2ba6, Qwen b968826d, Sarvam 071ae95e (short hashes).
# Put full commit SHAs here to pin; "main" resolves to the latest and the commit is printed.
MODELS = {
    "gemma4_e4b": dict(repo="google/gemma-4-E4B-it", rev="main", file="model.safetensors",
                       tensor="model.language_model.embed_tokens.weight", softcap=30.0, tied=True),
    "qwen3_8b":   dict(repo="Qwen/Qwen3-8B", rev="main", file="model-00005-of-00005.safetensors",
                       tensor="lm_head.weight", softcap=None, tied=False),
    "sarvam_30b": dict(repo="sarvamai/sarvam-30b", rev="main", file="model-00026-of-00026.safetensors",
                       tensor="lm_head.weight", softcap=None, tied=False),
}
RUN = os.environ.get("HEAD_MODELS", ",".join(MODELS)).split(",")

# %%
def fetch_head(name, cfg):
    """Range-read one tensor out of a safetensors file on the HF Hub into a local .bin; return a memmap."""
    url = f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['rev']}/{cfg['file']}"
    s = requests.Session()
    r = s.get(url, headers={"Range": "bytes=0-7"}); r.raise_for_status()
    n = struct.unpack("<Q", r.content)[0]
    commit = s.head(url, allow_redirects=False).headers.get("X-Repo-Commit", "?")
    hdr = json.loads(s.get(url, headers={"Range": f"bytes=8-{7+n}"}).content)
    t = hdr[cfg["tensor"]]
    a, b = t["data_offsets"]
    path = f"{OUTDIR}/{name}.bin"
    if not (os.path.exists(path) and os.path.getsize(path) == b - a):
        t0 = time.time()
        with s.get(url, headers={"Range": f"bytes={8+n+a}-{8+n+b-1}"}, stream=True) as r, open(path, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 24):
                f.write(chunk)
        print(f"  downloaded {(b-a)/2**30:.2f} GiB in {time.time()-t0:.0f}s")
    np_dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[t["dtype"]]
    W = np.memmap(path, dtype=np_dt, mode="r", shape=tuple(t["shape"]))
    return W, t["dtype"], commit


def to_f32(x, dtype):
    if dtype == "BF16":
        return (x.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float32)

# %%
def head_stats(W, dtype, chunk=8192, G=128, tgrid=(0.0, 0.5, 1.0, 1.5, 2.0)):
    V, d = W.shape
    # Pass 1: mean row, row norms, second-moment matrix W^T W (for the top direction).
    s = np.zeros(d, np.float64)
    C = np.zeros((d, d), np.float64)
    norms = np.empty(V, np.float32)
    for i in range(0, V, chunk):
        x = to_f32(W[i:i+chunk], dtype)
        s += x.sum(0, dtype=np.float64)
        norms[i:i+len(x)] = np.linalg.norm(x, axis=1)
        C += (x.T @ x).astype(np.float64)
    mu = s / V
    mu_n = np.linalg.norm(mu)
    mu32 = mu.astype(np.float32)

    # Pass 2: cosine to mu, centered norms, INT4 group-128 scales (max |w| per group) for W - t*mu.
    cos = np.empty(V, np.float32)
    cnorm = np.empty(V, np.float32)
    gscale = {t: [] for t in tgrid}
    rng = np.random.default_rng(0)
    for i in range(0, V, chunk):
        x = to_f32(W[i:i+chunk], dtype)
        dot = x @ mu32
        nn = norms[i:i+len(x)]
        cos[i:i+len(x)] = dot / (nn * mu_n + 1e-30)
        cnorm[i:i+len(x)] = np.sqrt(np.maximum(nn**2 - 2*dot + mu_n**2, 0))
        sub = rng.choice(len(x), size=min(1024, len(x)), replace=False)  # subsample rows for group scales
        for t in tgrid:
            g = np.abs(x[sub] - t*mu32).reshape(len(sub), d // G, G).max(-1)
            gscale[t].append(g.ravel())
    gscale = {t: np.concatenate(v) for t, v in gscale.items()}

    evals, evecs = np.linalg.eigh(C)
    v1 = evecs[:, -1]
    fro2 = np.trace(C)
    return dict(
        V=V, d=d, mu=mu, norms=norms, cos=cos, cnorm=cnorm, gscale=gscale,
        mu_norm=mu_n,
        row_norm_median=float(np.median(norms)),
        row_norm_p10=float(np.percentile(norms, 10)),
        row_norm_p90=float(np.percentile(norms, 90)),
        mu_over_median_row=float(mu_n / np.median(norms)),
        energy_share_mu=float(V * mu_n**2 / fro2),           # Kadav eq. 5: share of ||W||_F^2 in the shared component
        centered_over_raw_median=float(np.median(cnorm) / np.median(norms)),
        cos_median=float(np.median(cos)), cos_p10=float(np.percentile(cos, 10)), cos_p90=float(np.percentile(cos, 90)),
        frac_rows_cos_gt_0=float((cos > 0).mean()),
        top_eig_share=float(evals[-1] / fro2),
        cos_v1_mu=float(abs(v1 @ mu) / mu_n),
        mu_linf_over_l2=float(np.abs(mu).max() / mu_n),       # 1/sqrt(d) if spread evenly, 1 if one dimension
        mu_top8_dims=np.argsort(-np.abs(mu))[:8].tolist(),
        mu_top8_share=float(np.sort(mu**2)[-8:].sum() / mu_n**2),
        gscale_median={t: float(np.median(v)) for t, v in gscale.items()},
        gscale_mean={t: float(np.mean(v)) for t, v in gscale.items()},
    )

# %%
results = {}
for name in RUN:
    cfg = MODELS[name]
    print(f"== {name}: {cfg['repo']} :: {cfg['tensor']}")
    W, dtype, commit = fetch_head(name, cfg)
    print(f"  commit {commit}  dtype {dtype}  shape {W.shape}")
    t0 = time.time()
    results[name] = st = head_stats(W, dtype)
    st["commit"], st["dtype"] = commit, dtype
    print(f"  stats in {time.time()-t0:.0f}s")

# %%
SCALARS = ["V", "d", "dtype", "mu_norm", "row_norm_median", "row_norm_p10", "row_norm_p90",
           "mu_over_median_row", "energy_share_mu", "centered_over_raw_median",
           "cos_median", "cos_p10", "cos_p90", "frac_rows_cos_gt_0",
           "top_eig_share", "cos_v1_mu", "mu_linf_over_l2", "mu_top8_share", "mu_top8_dims"]
print(f"{'':28s}" + "".join(f"{n:>16s}" for n in results))
for k in SCALARS:
    row = []
    for n in results:
        v = results[n][k]
        row.append(f"{v:>16.4g}" if isinstance(v, float) else f"{str(v):>16s}")
    print(f"{k:28s}" + "".join(row))
print("\nINT4 group-128 scale (median max|w| per group) for W - t*mu, relative to t=0:")
for n, st in results.items():
    base = st["gscale_median"][0.0]
    print(f"  {n:12s} " + "  ".join(f"t={t}: {v/base:.3f}" for t, v in st["gscale_median"].items()))

summary = {n: {k: st[k] for k in SCALARS + ["commit", "gscale_median", "gscale_mean"]} for n, st in results.items()}
with open(f"{OUTDIR}/head_mean_row_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)

# %%
import matplotlib
if not os.environ.get("DISPLAY") and "google.colab" not in str(globals().get("get_ipython", lambda: "")()):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(len(results), 3, figsize=(15, 3.6 * len(results)), squeeze=False)
for r, (n, st) in enumerate(results.items()):
    a = ax[r, 0]
    a.hist(st["norms"], bins=200, color="0.4")
    a.axvline(st["mu_norm"], color="C3", label=f"||mu|| = {st['mu_norm']:.3g}")
    a.set_title(f"{n}: row norms"); a.legend(); a.set_yscale("log")
    a = ax[r, 1]
    a.hist(st["cos"], bins=200, color="0.4")
    a.set_title(f"{n}: cos(w_i, mu)"); a.set_yscale("log")
    a = ax[r, 2]
    a.plot(np.sort(np.abs(st["mu"]))[::-1], color="C0")
    a.set_yscale("log"); a.set_title(f"{n}: |mu_j| sorted (d={st['d']})")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/head_mean_row.png", dpi=110)
plt.show()
