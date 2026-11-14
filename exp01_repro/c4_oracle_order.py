"""C4 oracle order, from the saved results: on greedy states, the recompute set under the oracle bound (true error)
is never larger than the set of any single-stage certificate on the same head. The other rules are a readout.

  python3 -I c4_oracle_order.py
"""
import glob
import json

import numpy as np

out = {}
for model in ("qwen3_8b", "gemma4_e4b"):
    res = {}
    for mode in ("greedy", "t1"):
        arrs = {}
        for f in sorted(glob.glob(f"results/{model}/{mode}_*.npz")):
            a = np.load(f)
            for k in a.files:
                if "/single/" in k and k.endswith("/R"):
                    arrs.setdefault(k, []).append(a[k])
        arrs = {k: np.concatenate(v) for k, v in arrs.items()}
        for k, R in arrs.items():
            vn, bn, _, rn, _ = k.split("/")
            if bn == "oracle":
                continue
            orc = arrs[f"{vn}/oracle/single/{rn}/R"]
            res[f"{rn}/{vn}/{bn}"] = dict(states=int(len(R)), oracle_larger=int((orc > R).sum()),
                                          median_oracle=float(np.median(orc)), median_cert=float(np.median(R)))
    gate = sum(v["oracle_larger"] for k, v in res.items() if k.startswith("greedy/"))
    other = sum(v["oracle_larger"] for k, v in res.items() if not k.startswith("greedy/"))
    out[model] = dict(c4_pass=gate == 0, greedy_oracle_larger=gate, other_rules_oracle_larger=other, cells=res)
    print(model, "C4", "PASS" if gate == 0 else "FAIL", "| greedy cells", sum(k.startswith("greedy/") for k in res),
          "states with oracle larger:", gate, "| other rules (readout):", other)
    for k, v in res.items():
        if v["oracle_larger"]:
            print("   ", k, v)
json.dump(out, open("results/controls_c4.json", "w"), indent=1)
