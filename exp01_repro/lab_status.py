import glob, json, sys, numpy as np
GROUPS = ["english","code","hindi","bengali","tamil","telugu"]
P = lambda x,q: float(np.percentile(x,q,method="inverted_cdf"))
for rdir,V in (("results/qwen3_8b",151936),("results/gemma4_e4b",262144)):
    def load(mode):
        arrs, groups = {}, []
        for f in sorted(glob.glob(f"{rdir}/{mode}_*.npz")):
            a = np.load(f)
            for k in a.files:
                if k.startswith("int4d_g128/hoeff/casc/") or k.startswith("int4d_g128/hoeff/single/"): arrs.setdefault(k, []).append(a[k])
            groups += [r[1] for r in json.load(open(f[:-4]+".json"))["rows"]]
        return {k: np.concatenate(v) for k,v in arrs.items()}, np.array(groups)
    G,gG = load("greedy"); T,gT = load("t1")
    print(rdir, "states by group greedy", {g:int((gG==g).sum()) for g in GROUPS})
    for rn,(A,g) in {"greedy":(G,gG),"gumbel_0.6":(T,gT),"gumbel_1.0":(T,gT),"top50":(T,gT),"minp_0.05":(T,gT)}.items():
        k=f"int4d_g128/hoeff/casc/{rn}"; R,B,S=A[k+"/R"],A[k+"/bytes"],A[k+"/S"]
        per=[(P(R[g==x],50),P(R[g==x],99),P(B[g==x],50)) for x in GROUPS]
        gates=all(r50<=0.001*V and r99<=0.01*V and b<=0.35 for r50,r99,b in per)
        cheap=all(b<=0.35 for *_,b in per); uh=all(b<0.5 for *_,b in per)
        st="GATES_PASS" if gates else "CHEAP" if cheap else "UNDER_HALF" if uh else "MIXED" if P(B,50)<0.5 else "DEAD"
        s1=A[f"int4d_g128/hoeff/single/{rn}/R"]
        print(f"  {rn:11s} lab-rule status {st:10s} bytes med by group "+" ".join(f"{b:.3f}" for *_,b in per)+f" | max group R p99 {max(r for _,r,_ in per):.0f} | all: med {P(B,50):.3f} p99 {P(B,99):.3f} | dither-only R med {P(s1,50):.0f} p99 {P(s1,99):.0f}")
