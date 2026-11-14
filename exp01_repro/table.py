"""Lab-note style main table for our run. Usage: python3 -I table.py results/qwen3_8b [Model name]"""
import sys, numpy as np
sys.argv += [] ; rdir = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-8B"
import importlib.util
spec = importlib.util.spec_from_file_location("s", __file__.rsplit("/",1)[0] + "/summarize.py")
# reuse loader without printing: re-implement small load
import glob, json
def load(mode):
    arrs = {}
    for f in sorted(glob.glob(f"{rdir}/{mode}_*.npz")):
        a = np.load(f)
        for k in a.files: arrs.setdefault(k, []).append(a[k])
    return {k: np.concatenate(v) for k, v in arrs.items()}
G, T = load("greedy"), load("t1")
V = 151936 if "qwen" in rdir else 262144
RULES = [("greedy","greedy",G),("Gumbel T = 0.6","gumbel_0.6",T),("Gumbel T = 1","gumbel_1.0",T),("top-50","top50",T),("min-p 0.05","minp_0.05",T)]
def status(R, B):
    if np.median(B) >= 0.5: return "DEAD"
    if np.median(R) <= 0.001*V and np.percentile(R,99) <= 0.01*V and np.percentile(B,99) < 0.5: return "GATES_PASS"
    return "UNDER_HALF"
print("| Model | Rule | Status | S | R | Bytes median, p99, mean | INT8 bytes median, p99 |")
print("|---|---|---|---|---|---|---|")
for lab, rn, A in RULES:
    k = f"int4d_g128/hoeff/casc/{rn}"
    S, R, B = A[k+"/S"], A[k+"/R"], A[k+"/bytes"]
    i8 = min((A[f"int8_row/{b}/single/{rn}/bytes"] for b in ("wc","res")), key=np.median)
    f = lambda x: f"{np.median(x):,.0f}, {np.percentile(x,99):,.0f}"
    print(f"| {name} | {lab} | {status(R,B)} | {f(S)} | {f(R)} | {np.median(B):.3f}, {np.percentile(B,99):.3f}, {B.mean():.3f} | {np.median(i8):.3f}, {np.percentile(i8,99):.3f} |")
