import pickle,glob,os,numpy as np,collections
from scipy.stats import norm
from cluster_boot import ids_for, boot_se
A="../results/runs16/"; runs={}
for f in sorted(glob.glob(A+"*.pkl")):
    r=pickle.load(open(f,"rb"))
    if not("OMP-64 centred (raw)" in r["curves"] and r.get("train_steps")==200 and "LS (cal)" not in r["curves"]): continue
    runs[os.path.basename(f)]=r
print(len(runs),"runs")
rng=np.random.default_rng(0); res=[]
for fn,r in runs.items():
    cv,sel=r["curves"],r["sel"]
    try: ids=ids_for("runs16",fn)
    except FileNotFoundError: ids=None          # arXiv runs: no abstract IDs recovered, sentence-level bootstrap kept
    for st in (0,200):
        for b,lab in (("OMP-64 centred (raw)","64c"),("OMP-64 centred (cal)","64cal")):
            d=cv[sel][st]["t_eval"]["tot"]-cv[b][st]["t_eval"]["tot"]
            m,se=boot_se(d,ids,rng=rng); m0,se0=boot_se(d,None,rng=rng)
            res.append((fn,st,lab,m,se,se0,ids is not None,2*(1-norm.cdf(abs(m/se)))))
for lab in ("64c","64cal"):
    sub=[x for x in res if x[2]==lab]; K=len(sub); idx=np.argsort([x[7] for x in sub]); rej=0
    for r_,i in enumerate(idx):
        if sub[i][7]<=0.05/(K-r_): rej+=1
        else: break
    cl=[x for x in sub if x[6]]
    print(lab,"n",K,"(cluster-bootstrapped:",len(cl),") all negative",all(x[3]<0 for x in sub),"max p %.2g"%max(x[7] for x in sub),"Holm rejects",rej,
          "| median SE ratio cluster/sentence %.2f"%np.median([x[4]/x[5] for x in cl]))
