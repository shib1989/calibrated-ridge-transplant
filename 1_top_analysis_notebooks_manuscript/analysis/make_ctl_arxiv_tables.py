"""Tables for (a) the control "least squares on the same 5000 most frequent anchors" (PubMed, seed 0, four small pairs) and
(b) the arXiv second domain (four small pairs, seed 0, training-only word rule).
Abstract-level (cluster) paired bootstrap, B resamples; Holm over each family with a normal approximation to the bootstrap distribution.
Inputs (default: ../results/...; override with env vars CTL, AX, IDS, OUT):
  CTL = control_ls_freq5000/*.pkl, AX = arxiv/*.pkl, IDS = abstract_ids/*.json, v6 PubMed seed-0 pkls in V6 (to check the control reproduces v6).
Outputs: OUT/tables/control_ls.tex, OUT/tables/arxiv_summary.tex, OUT/stats_ctl_arxiv.json"""
import os, glob, json, pickle
import numpy as np
from math import erf, sqrt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
CTL = os.environ.get("CTL", os.path.join(R, "control_ls_freq5000")); AX = os.environ.get("AX", os.path.join(R, "arxiv"))
IDS = os.environ.get("IDS", os.path.join(R, "abstract_ids")); V6 = os.environ.get("V6", os.path.join(R, "v6_trainonly"))
OUT = os.environ.get("OUT", os.path.join(HERE, "out_ctl_arxiv")); os.makedirs(OUT + "/tables", exist_ok=True)
B = 20000
PAIRS = ["GPT-2_from_Pythia-160m", "GPT-2_from_Qwen2.5-0.5B", "OPT-125m_from_Pythia-160m", "OPT-125m_from_Qwen2.5-0.5B"]
LAB = {p: p.replace("_from_", " $\\leftarrow$ ") for p in PAIRS}

def boot(d, ids):
    rng = np.random.default_rng(0); u, inv = np.unique(ids, return_inverse=True)
    s = np.bincount(inv, weights=d); c = np.bincount(inv).astype(float); k = len(u); b = np.empty(B)
    for i in range(0, B, 2000):
        dr = rng.integers(0, k, (2000, k)); b[i:i + 2000] = s[dr].sum(1) / c[dr].sum(1)
    return d.mean(), b.std(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def pval(m, se): return 1 - erf(abs(m) / se / sqrt(2))
def holm(p):
    K = len(p); rej = np.zeros(K, bool)
    for r, i in enumerate(np.argsort(p)):
        if p[i] <= 0.05 / (K - r): rej[i] = True
        else: break
    return rej
def sg(x): return ("$+$" if x > 0 else "$-$") + f"{abs(x):.2f}"
def ids_for(folder, kind, pair): return np.array(json.load(open(f"{IDS}/{folder}__{kind}__{pair}__seed0.pkl.json"))["abstract_id"])
STATS = {}

# ------------------------------------------------------------ (a) control
rows = {}   # (pair, step, col) -> (m, se, lo, hi)
COLS = [("ridge[freq5000]", "LS[freq5000]"), ("LS[freq5000]", "LS (cal)"), ("ridge[all]", "LS (cal)")]
for p in PAIRS:
    c = pickle.load(open(f"{CTL}/pubmed__{p}__seed0.pkl", "rb")); cv = c["curves"]; ids = ids_for("v6_trainonly", "pubmed", p)
    v = pickle.load(open(f"{V6}/pubmed__{p}__seed0.pkl", "rb"))
    assert c["terms"] == v["terms"]                                                    # same words as v6 seed 0
    for k in ("ridge[all]", "ridge[freq5000]", "LS (cal)"):                            # the shared conditions reproduce v6 exactly
        for st in (0, 200): assert np.allclose(cv[k][st]["t_eval"]["tot"], v["curves"][k][st]["t_eval"]["tot"], atol=1e-3), (p, k, st)
    assert len(ids) == len(cv["original"][0]["t_eval"]["tot"])
    for st in (0, 200):
        for j, (a, b) in enumerate(COLS):
            rows[(p, st, j)] = boot(cv[a][st]["t_eval"]["tot"] - cv[b][st]["t_eval"]["tot"], ids)
keys = list(rows); rej = {}
for j in range(3):
    kj = [k for k in keys if k[2] == j]; r = holm(np.array([pval(rows[k][0], rows[k][1]) for k in kj]))
    rej.update(dict(zip(kj, r)))
def cell(k):
    m, se, lo, hi = rows[k]; s = f"{sg(m)}\\,[{sg(lo)},{sg(hi)}]"
    if lo > 0 or hi < 0: s += "$^{*}$"
    if rej[k]: s += "$^{\\dagger}$"
    return s
t = "\\begin{tabular}{llccc}\n\\toprule\nPair & Steps & ridge[freq5000] $-$ LS[freq5000] & LS[freq5000] $-$ LS (all anchors) & ridge[all] $-$ LS (all anchors) \\\\\n\\midrule\n"
for p in PAIRS:
    for i, st in enumerate((0, 200)): t += (LAB[p] if i == 0 else "") + f" & {st} & " + " & ".join(cell((p, st, j)) for j in range(3)) + " \\\\\n"
    t += "\\addlinespace\n"
t = t.rsplit("\\addlinespace\n", 1)[0] + "\\bottomrule\n\\end{tabular}\n"
open(OUT + "/tables/control_ls.tex", "w").write(t)
STATS["control"] = {f"{p}|{st}|{j}": dict(mean=rows[(p, st, j)][0], se=rows[(p, st, j)][1], lo=rows[(p, st, j)][2], hi=rows[(p, st, j)][3], holm=bool(rej[(p, st, j)])) for (p, st, j) in rows}
for j, (a, b) in enumerate(COLS):
    v = [rows[(p, st, j)] for p in PAIRS for st in (0, 200)]
    print(f"control col {j} {a} - {b}: ridge/first better (CI<0) in {sum(x[3] < 0 for x in v)}, worse (CI>0) {sum(x[2] > 0 for x in v)}, level {sum(x[2] <= 0 <= x[3] for x in v)}, Holm {sum(rej[(p, st, j)] for p in PAIRS for st in (0, 200))}; range {min(x[0] for x in v):+.2f}..{max(x[0] for x in v):+.2f}")

# ------------------------------------------------------------ (b) arXiv
BASE = [("OMP-64 centred (cal)", "OMP-64c cal."), ("OMP-64 as shipped (raw)", "OMP-64s"), ("LS (cal)", "LS"), ("Procrustes (cal)", "Procrustes"), ("SALT-ours (cal)", "SALT-ours"), ("WECHSEL (cal)", "WECHSEL"), ("original", "Unch.")]
ar = {}; sels = {}; Ks = {}
for p in PAIRS:
    r = pickle.load(open(f"{AX}/arxiv__{p}__seed0.pkl", "rb")); cv = r["curves"]; sel = r["sel"]; sels[p] = sel; Ks[p] = r["K"]
    ids = ids_for("v6_arxiv", "arxiv", p); assert len(ids) == len(cv["original"][0]["t_eval"]["tot"])
    for st in (0, 200):
        for b, _ in BASE:
            o = cv["original"][0]["t_eval"]["tot"] if b == "original" else cv[b][st]["t_eval"]["tot"]
            ar[(p, st, b)] = boot(cv[sel][st]["t_eval"]["tot"] - o, ids)
ka = list(ar); ra = dict(zip(ka, holm(np.array([pval(ar[k][0], ar[k][1]) for k in ka]))))
def acell(k):
    m, se, lo, hi = ar[k]; s = sg(m)
    if lo > 0 or hi < 0: s += "$^{*}$"
    if ra[k]: s += "$^{\\dagger}$"
    return s
t = "\\begin{tabular}{llrrrrrrr}\n\\toprule\nPair (selected variant) & Steps & " + " & ".join(n for _, n in BASE) + " \\\\\n\\midrule\n"
for p in PAIRS:
    for i, st in enumerate((0, 200)):
        t += ((LAB[p] + f" ({sels[p].replace('ridge', 'r')})") if i == 0 else "") + f" & {st} & " + " & ".join(acell((p, st, b)) for b, _ in BASE) + " \\\\\n"
    t += "\\addlinespace\n"
t = t.rsplit("\\addlinespace\n", 1)[0] + "\\bottomrule\n\\end{tabular}\n"
open(OUT + "/tables/arxiv_summary.tex", "w").write(t)
sig = [k for k in ka if ar[k][2] > 0 or ar[k][3] < 0]
better = [k for k in sig if ar[k][0] < 0]; worse = [k for k in sig if ar[k][0] > 0]
STATS["arxiv"] = dict(K=Ks, sel=sels, n_tests=len(ka), ci_sig=len(sig), better=len(better), worse=len(worse), holm=int(sum(ra.values())),
                      holm_better=int(sum(ra[k] for k in better)), holm_worse=int(sum(ra[k] for k in worse)),
                      worse_cells=[[k[0], k[1], k[2], ar[k][0], ar[k][2], ar[k][3]] for k in worse],
                      level_cells=[[k[0], k[1], k[2], ar[k][0], ar[k][2], ar[k][3]] for k in ka if k not in sig or not ra[k]],
                      cells={f"{p}|{st}|{b}": [ar[(p, st, b)][0], ar[(p, st, b)][2], ar[(p, st, b)][3], bool(ra[(p, st, b)])] for (p, st, b) in ar})
print("arxiv:", {k: v for k, v in STATS["arxiv"].items() if k not in ("cells",)})
json.dump(STATS, open(OUT + "/stats_ctl_arxiv.json", "w"), indent=1, default=str)
