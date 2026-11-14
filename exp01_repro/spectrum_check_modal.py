"""Independent check of the one-direction share, with numpy only and none of the project's analysis code.

  modal run spectrum_check_modal.py
Answers: is Sarvam-30B's dominant state direction real, or an artifact of how we measure it?
For each model's saved greedy states it recomputes the top-1 share of the uncentred second moment, shows what
the same measure gives on states whose coordinates are shuffled (a null), and looks at the direction by language
group and by position in the answer. Diagnostic, run after K2 and K3; no rule file.
"""
import json
import os

import modal

app = modal.App("exp01-spectrum-check")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=32 * 1024, timeout=3600)
def check(model: str):
    import numpy as np
    import torch

    H = torch.load(f"{VOL}/states/{model}/h_greedy.pt").float().numpy().astype(np.float64)
    rows = [json.loads(l) for l in open(f"{VOL}/states/{model}/rows_greedy.jsonl")]
    pid = np.array([r["pid"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    step = np.array([r["step"] for r in rows])
    n, d = H.shape

    def top(X):
        """Top-1 share of the uncentred second moment and its direction, by eigen-decomposition in float64."""
        w, v = np.linalg.eigh(X.T @ X)
        return float(w[-1] / w.sum()), v[:, -1]

    even = pid % 2 == 0
    share, p = top(H[even])
    # the same share from singular values of a random subsample (a different numerical route)
    rng = np.random.default_rng(0)
    sub = H[rng.choice(n, 4096, replace=False)]
    sv = np.linalg.svd(sub, compute_uv=False)
    # null: shuffle each state's coordinates independently, which keeps every norm and destroys any direction
    Hn = H[even].copy()
    for i in range(len(Hn)):
        rng.shuffle(Hn[i])
    null_share, _ = top(Hn)
    mean = H[even].mean(0)
    energy = (H[even] ** 2).sum(1).mean()
    out = dict(model=model, n=n, d=d, top1_share_even=share, top1_share_svd_subsample=float(sv[0] ** 2 / (sv ** 2).sum()),
               null_share_shuffled_coordinates=null_share, one_over_d=1 / d,
               mean_state_energy_share=float(mean @ mean / energy),
               cos_mean_state_top_direction=float(abs(mean @ p) / np.linalg.norm(mean)),
               norm_mean_over_mean_norm=float(np.linalg.norm(mean) / np.sqrt(energy)),
               largest_coordinates_of_direction=[(int(i), float(p[i])) for i in np.argsort(-np.abs(p))[:5]],
               energy_of_direction_in_top5_coordinates=float(np.sort(p ** 2)[-5:].sum()))
    rho = (H @ p) ** 2 / (H ** 2).sum(1)                      # share of each state along the direction (all states)
    out["rho_median_odd"] = float(np.median(rho[~even]))
    out["by_group"] = {}
    for g in sorted(set(grp)):
        m = grp == g
        own, pg = top(H[m])                                   # fitted within the group alone
        out["by_group"][g] = dict(states=int(m.sum()), own_top1_share=own, cos_own_with_global=float(abs(pg @ p)),
                                  rho_median=float(np.median(rho[m])))
    out["by_step"] = {f"{a}-{b - 1}": dict(states=int(((step >= a) & (step < b)).sum()),
                                           rho_median=float(np.median(rho[(step >= a) & (step < b)])))
                      for a, b in ((0, 1), (1, 8), (8, 32), (32, 128))}
    return out


@app.local_entrypoint()
def main(models: str = "sarvam_30b,qwen3_8b,gemma4_e4b"):
    here = os.path.dirname(os.path.abspath(__file__))
    for r in check.map(models.split(",")):
        json.dump(r, open(os.path.join(here, "logs", f"spectrum_check_{r['model']}.json"), "w"), indent=1)
        print(json.dumps(r, indent=1), flush=True)
