import pickle,numpy as np
from scipy.stats import norm
from cluster_boot import ids_for, boot_se
A="../results/v5_small_and_scale/"
PK=[("GPT-2<-Pythia","pubmed__GPT-2_from_Pythia-160m__seed1.pkl"),("GPT-2<-Qwen","pubmed__GPT-2_from_Qwen2.5-0.5B__seed1.pkl"),("OPT<-Pythia","pubmed__OPT-125m_from_Pythia-160m__seed1.pkl"),("OPT<-Qwen","pubmed__OPT-125m_from_Qwen2.5-0.5B__seed1.pkl"),("GPT-2L<-Qwen","pubmed__GPT-2_large_from_Qwen2.5-0.5B__seed1.pkl"),("Pythia1B","pubmed__Pythia-1B_from_Qwen2.5-0.5B__seed1.pkl"),("Qwen1.5B","pubmed__Qwen2.5-1.5B_from_Pythia-410m__seed1.pkl")]
FAM=["OMP-64 centred (cal)","OMP-64 as shipped (raw)","LS (cal)","Procrustes (cal)","SALT-ours (cal)","WECHSEL (cal)","OMP-32"]
rng=np.random.default_rng(0); out=[]; M=0.5
for nm,f in PK:
    d=pickle.load(open(A+f,'rb')); c=d['curves']; sel=d['sel']; ids=ids_for("v5_small_and_scale",f)
    for st in (0,200):
        for b in FAM:
            df=c[sel][st]['t_eval']['tot']-c[b][st]['t_eval']['tot']; m,se=boot_se(df,ids,rng=rng); pl,ph=boot_se.last; _,se0=boot_se(df,None,rng=rng)
            pv=2*(1-norm.cdf(abs(m/se))); out.append(dict(pl=pl,ph=ph,p=nm,st=st,b=b,m=m,se=se,se0=se0,pv=pv,lo=m-1.96*se,hi=m+1.96*se,eq=(m-1.645*se>-M and m+1.645*se<M)))
K=len(out); print(K,"comparisons"); idx=np.argsort([o['pv'] for o in out]); rej=np.zeros(K,bool)
for r,i in enumerate(idx):
    if out[i]['pv']<=0.05/(K-r): rej[i]=True
    else: break
for o,r in zip(out,rej): o['holm']=r
sig=lambda o:(o['pl']>0 or o['ph']<0)   # significant = the 95% percentile interval excludes 0 (as in the tables); Holm uses the normal-approximation p
better=[o for o in out if o['m']<0 and sig(o)]; worse=[o for o in out if o['m']>0 and sig(o)]; ns=[o for o in out if not sig(o)]
print("ridge significantly better",len(better),"after Holm",sum(o['holm'] for o in better))
print("wins not surviving Holm: |diff| %s"%sorted(round(abs(o['m']),2) for o in better if not o['holm']))
print("baseline significantly better:")
for o in worse: print("  ",o['p'],o['st'],o['b'],"ridge-base %+.2f [%+.2f,%+.2f] p=%.3g Holm %s"%(o['m'],o['lo'],o['hi'],o['pv'],o['holm']))
print("not significant",len(ns),"equivalent within +-0.5:",sum(o['eq'] for o in ns))
print("median SE ratio cluster/sentence %.2f"%np.median([o['se']/o['se0'] for o in out]))
import collections
per=collections.defaultdict(lambda:[0,0,0])
for o in out: per[o['b']][0 if (o['m']<0 and sig(o)) else 2 if (o['m']>0 and sig(o)) else 1]+=1
for b,v in per.items(): print(b,"better/ns/worse",v)
pickle.dump(out,open('fam_cluster.pkl','wb'))
