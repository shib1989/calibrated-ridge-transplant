import pickle,numpy as np,re
from scipy.stats import norm
A="../results/v5_small_and_scale/"
PK=[("GPT-2<-Pythia","pubmed__GPT-2_from_Pythia-160m__seed1.pkl"),("GPT-2<-Qwen","pubmed__GPT-2_from_Qwen2.5-0.5B__seed1.pkl"),("OPT<-Pythia","pubmed__OPT-125m_from_Pythia-160m__seed1.pkl"),("OPT<-Qwen","pubmed__OPT-125m_from_Qwen2.5-0.5B__seed1.pkl"),("GPT-2L<-Qwen","pubmed__GPT-2_large_from_Qwen2.5-0.5B__seed1.pkl"),("Pythia1B","pubmed__Pythia-1B_from_Qwen2.5-0.5B__seed1.pkl")]
FAM=["OMP-64 centred (cal)","OMP-64 as shipped (raw)","LS (cal)","Procrustes (cal)","SALT-ours (cal)","WECHSEL (cal)","OMP-32"]
rng=np.random.default_rng(0); rows=[]   # (pair,step,base,mean,se)
for nm,f in PK:
    d=pickle.load(open(A+f,'rb')); c=d['curves']; sel=d['sel']
    for st in (0,200):
        for b in FAM:
            df=c[sel][st]['t_eval']['tot']-c[b][st]['t_eval']['tot']; n=len(df)
            bm=df[rng.integers(0,n,(5000,n))].mean(1); rows.append((nm,st,b,df.mean(),bm.std()))
t=open(A+"qwen2.5-1.5B_from_pythia-410m_printed_report.txt").read().split("\n")
key={"OMP-64 centred (cal)":"OMP-64 centred (cal)","OMP-64 as shipped (raw)":"OMP-64 as shipped (raw)","LS (cal)":"LS (cal)","Procrustes (cal)":"Procrustes (cal)","SALT-ours (cal)":"SALT-ours (cal)","WECHSEL (cal)":"WECHSEL (cal)","OMP-32":"OMP-32"}
for st,(a,b) in {0:(24,40),200:(56,70)}.items():
    for l in t[a-1:b]:
        m=re.match(r"\s+ridge\[freq5000\] - (.+?): ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]",l)
        if m and m.group(1) in key:
            mean,lo,hi=map(float,m.groups()[1:]); rows.append(("Qwen1.5B",st,m.group(1),mean,(hi-lo)/3.92))
print(len(rows))
M=0.5
out=[]
for p,st,b,m,se in rows:
    z=m/se; pv=2*(1-norm.cdf(abs(z))); lo90=m-1.645*se; hi90=m+1.645*se
    out.append(dict(p=p,st=st,b=b,m=m,se=se,pv=pv,eq=(lo90>-M and hi90<M)))
# Holm over 98
idx=np.argsort([o['pv'] for o in out]); K=len(out); rej=np.zeros(K,bool)
for r,i in enumerate(idx):
    if out[i]['pv']<=0.05/(K-r): rej[i]=True
    else: break
for o,r in zip(out,rej): o['holm']=r
better=[o for o in out if o['m']<0 and o['pv']<0.05]; worse=[o for o in out if o['m']>0 and o['pv']<0.05]
print("raw-significant better",len(better),"after Holm",sum(o['holm'] for o in better))
print("raw-significant worse",[(o['p'],o['st'],o['b'],round(o['m'],2),round(o['pv'],4),o['holm']) for o in worse])
ns=[o for o in out if o['pv']>=0.05]
print("not significant",len(ns),"equivalent within ±%.1f"%M,sum(o['eq'] for o in ns))
for o in ns: print("  ",o['p'],o['st'],o['b'],round(o['m'],2),"se",round(o['se'],2),"eq",o['eq'])
pickle.dump(out,open('fam.pkl','wb'))
