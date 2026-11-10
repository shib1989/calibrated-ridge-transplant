import pickle,glob,numpy as np
from scipy.stats import norm
A="../results/runs16/"
runs={}
for f in sorted(glob.glob(A+"*.pkl")):
    try: r=pickle.load(open(f,"rb"))
    except Exception: continue
    if not(isinstance(r,dict) and "curves" in r and "OMP-64 centred (raw)" in r["curves"] and r.get("train_steps")==200 and "LS (cal)" not in r["curves"]): continue
    dom="PubMed" if " apoptosis" in r["terms"][:6] else "arXiv"
    runs[(r["pair"],dom,r["seed"])]=r
print(len(runs))
rng=np.random.default_rng(0)
def ps(d):
    n=len(d); bm=d[rng.integers(0,n,(5000,n))].mean(1); se=bm.std(); return d.mean(),se
res=[]
for k,r in runs.items():
    cv,sel=r["curves"],r["sel"]
    for st in (0,200):
        for b,lab in (("OMP-64 centred (raw)","64c"),("OMP-64 centred (cal)","64cal")):
            d=cv[sel][st]["t_eval"]["tot"]-cv[b][st]["t_eval"]["tot"]; m,se=ps(d); res.append((k,st,lab,m,se,2*(1-norm.cdf(abs(m/se)))))
for lab in ("64c","64cal"):
    sub=[x for x in res if x[2]==lab]; K=len(sub); idx=np.argsort([x[5] for x in sub]); rej=0
    for r_,i in enumerate(idx):
        if sub[i][5]<=0.05/(K-r_): rej+=1
        else: break
    print(lab,"n",K,"all negative",all(x[3]<0 for x in sub),"max p",max(x[5] for x in sub),"Holm rejects",rej)
# pair-level: per pair mean over 4 runs
import collections
for lab in ("64c","64cal"):
    for st in (0,200):
        by=collections.defaultdict(list)
        for k,s,l,m,se,p in res:
            if l==lab and s==st: by[k[0]].append(m)
        print(lab,st,{p[:14]:round(np.mean(v),2) for p,v in by.items()})
