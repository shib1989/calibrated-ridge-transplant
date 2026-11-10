# Run in the Kaggle notebook (needs torch, transformers). Compares ridge directions in float32 vs float64 (fixed lambda=0.01, same anchors).
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
PAIRS=[("gpt2","EleutherAI/pythia-160m"),("gpt2","Qwen/Qwen2.5-0.5B")]
def emb(n):
    m=AutoModelForCausalLM.from_pretrained(n,torch_dtype=torch.float32); return m.get_input_embeddings().weight.detach().double(),AutoTokenizer.from_pretrained(n)
def ridge(X,Y,Xn,dt,lam=1e-2):
    X,Y,Xn=X.to(dt),Y.to(dt),Xn.to(dt)
    mx,my=X.mean(0),Y.mean(0); Xc,Yc=X-mx,Y-my; s=Xc.norm(dim=1).mean(); Xc=Xc/s
    G=Xc.T@Xc; W=torch.linalg.solve(G+lam*G.trace()/G.shape[0]*torch.eye(G.shape[0],dtype=dt),Xc.T@Yc)
    D=((Xn-mx)/s)@W; return D/D.norm(dim=1,keepdim=True)
for rn,dn in PAIRS:
    R,rt=emb(rn); D,dt_=emb(dn); rv,dv=rt.get_vocab(),dt_.get_vocab()
    shared=[t for t in dv if t in rv]; new=[t for t in dv if t not in rv][:3000]
    ia=torch.tensor([dv[t] for t in shared]); ra=torch.tensor([rv[t] for t in shared]); inw=torch.tensor([dv[t] for t in new])
    X,Y,Xn=D[ia],R[ra],D[inw]
    a=ridge(X,Y,Xn,torch.float32).double(); b=ridge(X,Y,Xn,torch.float64)
    cos=(a*b).sum(1)
    print(rn,"<-",dn,"anchors",len(shared),"new",len(new),"| cosine fp32 vs fp64: min %.6f mean %.6f"%(cos.min(),cos.mean()))
