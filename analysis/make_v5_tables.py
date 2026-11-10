"""Builds tables/scale_baselines.tex, scale_ci_a/b.tex and scale_bpb.tex from the seven v5 pkls. Intervals: percentile bootstrap resampling ABSTRACTS (ids in ../results/abstract_ids)."""
import pickle, re, numpy as np
from cluster_boot import ids_for
A="../results/v5_small_and_scale/"
PK=[("GPT-2 $\\leftarrow$ Pythia-160m","pubmed__GPT-2_from_Pythia-160m__seed1.pkl"),
    ("GPT-2 $\\leftarrow$ Qwen2.5-0.5B","pubmed__GPT-2_from_Qwen2.5-0.5B__seed1.pkl"),
    ("OPT-125m $\\leftarrow$ Pythia-160m","pubmed__OPT-125m_from_Pythia-160m__seed1.pkl"),
    ("OPT-125m $\\leftarrow$ Qwen2.5-0.5B","pubmed__OPT-125m_from_Qwen2.5-0.5B__seed1.pkl"),
    ("GPT-2 large $\\leftarrow$ Qwen2.5-0.5B","pubmed__GPT-2_large_from_Qwen2.5-0.5B__seed1.pkl"),
    ("Pythia-1B (untied) $\\leftarrow$ Qwen2.5-0.5B","pubmed__Pythia-1B_from_Qwen2.5-0.5B__seed1.pkl"),
    ("Qwen2.5-1.5B $\\leftarrow$ Pythia-410m","pubmed__Qwen2.5-1.5B_from_Pythia-410m__seed1.pkl")]
BASE=[("OMP-64 centred (cal)","OMP-64c cal."),("OMP-64 as shipped (raw)","OMP-64s"),("LS (cal)","LS"),("Procrustes (cal)","Procrustes"),("SALT-ours (cal)","SALT-ours"),("WECHSEL (cal)","WECHSEL"),("original","Unch.")]
rng=np.random.default_rng(0)
def boot(df,ids,B=5000):
    u,inv=np.unique(ids,return_inverse=True); sm=np.bincount(inv,weights=df); c=np.bincount(inv).astype(float); k=len(u)
    draw=rng.integers(0,k,(B,k)); m=sm[draw].sum(1)/c[draw].sum(1); return df.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
R={}   # R[pair][step][base]=(mean,lo,hi,bpb)
for nm,f in PK:
    d=pickle.load(open(A+f,'rb')); c=d['curves']; sel=d['sel']; ids=ids_for('v5_small_and_scale',f); nb=c[sel][0]['t_eval']['nbytes'].mean()
    R[nm]={}
    for st in (0,200):
        R[nm][st]={}
        for b,_ in BASE:
            o=c['original'][0]['t_eval']['tot'] if b=='original' else c[b][st]['t_eval']['tot']
            df=c[sel][st]['t_eval']['tot']-o; m,lo,hi=boot(df,ids); R[nm][st][b]=(m,lo,hi,m/np.log(2)/nb)
    R[nm]['sel']=sel; R[nm]['K']=d['K']
ORDER=[p for p,_ in PK]
def sg(x): return ("$+$" if x>0 else "$-$")+f"{abs(x):.2f}"
def cell(t): 
    m,lo,hi,_=t; s="*" if (lo>0 or hi<0) else ""
    return f"{sg(m)}\\,[{sg(lo)},{sg(hi)}]"+("$^{*}$" if s else "")
def mean_cell(t):
    m,lo,hi,_=t; return (f"{sg(m)}"+("$^{*}$" if (lo>0 or hi<0) else ""))
def table(cols,fn,name,first=True):
    s="\\begin{tabular}{ll"+"r"*len(cols)+"}\n\\toprule\nPair & Steps & "+" & ".join(dict(BASE)[b] for b in cols)+" \\\\\n\\midrule\n"
    for p in ORDER:
        for i,st in enumerate((0,200)):
            s+=(p if i==0 else "")+f" & {st} & "+" & ".join(fn(R[p][st][b]) for b in cols)+" \\\\\n"
        s+="\\addlinespace\n"
    s=s.rsplit("\\addlinespace\n",1)[0]+"\\bottomrule\n\\end{tabular}\n"
    open("../manuscript_source/tables/"+name+".tex","w").write(s)
ALL=[b for b,_ in BASE]
table(ALL,mean_cell,"scale_baselines")
table(ALL[:3],cell,"scale_ci_a")
table(ALL[3:],cell,"scale_ci_b")
s="\\begin{tabular}{llccccccc}\n\\toprule\nPair & Steps & "+" & ".join(b[1] for b in BASE)+" \\\\\n\\midrule\n"
def b4(x): return ("$+$" if x>0 else "$-$")+f"{abs(x):.4f}"
for p in ORDER:
    for i,st in enumerate((0,200)):
        s+=(p if i==0 else "")+f" & {st} & "+" & ".join(b4(R[p][st][b][3]) for b,_ in BASE)+" \\\\\n"
    s+="\\addlinespace\n"
s=s.rsplit("\\addlinespace\n",1)[0]+"\\bottomrule\n\\end{tabular}\n"
open("../manuscript_source/tables/scale_bpb.tex","w").write(s)
pickle.dump(R,open("R.pkl","wb"))
# summary for text
for p in ORDER:
    print(p,R[p]['sel'])
    for st in (0,200):
        print("  ",st," ".join(f"{b[:14]}:{R[p][st][b][0]:+.2f}{'*' if (R[p][st][b][1]>0 or R[p][st][b][2]<0) else ''}" for b,_ in BASE))
