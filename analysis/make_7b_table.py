"""Qwen2.5-7B <- Pythia-410m, zero-shot (fp16): table of ridge[selected] against each baseline and of each method against the unchanged receiver.
Abstract-level paired bootstrap (100000 resamples); the abstracts are those of the Qwen2.5-1.5B <- Pythia-410m seed-1 run (identical words and evaluation sentences: same Qwen tokenizer, same donor, same seed).
Holm over the five comparisons of ridge with the baselines (normal approximation to the bootstrap). Output: OUT/tables/b7_zeroshot.tex, OUT/stats_7b.json"""
import os, json, pickle, numpy as np
from math import erf, sqrt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
OUT = os.environ.get("OUT", os.path.join(HERE, "out_7b")); os.makedirs(OUT + "/tables", exist_ok=True)
r = pickle.load(open(f"{R}/v10_7b/pubmed__Qwen2.5-7B_from_Pythia-410m__seed1.pkl", "rb")); cv = r["curves"]; sel = r["sel"]
q = pickle.load(open(f"{R}/v6_trainonly/pubmed__Qwen2.5-1.5B_from_Pythia-410m__seed1.pkl", "rb")); assert q["terms"] == r["terms"]
assert np.array_equal(q["curves"]["original"][0]["t_eval"]["nbytes"], cv["original"][0]["t_eval"]["nbytes"])
ids = np.array(json.load(open(f"{R}/abstract_ids/v6_trainonly__pubmed__Qwen2.5-1.5B_from_Pythia-410m__seed1.pkl.json"))["abstract_id"]); assert len(ids) == 600
def boot(d, B=100000):
    rng = np.random.default_rng(0); u, inv = np.unique(ids, return_inverse=True); s = np.bincount(inv, weights=d); c = np.bincount(inv).astype(float); k = len(u); b = np.empty(B)
    for i in range(0, B, 2000):
        dr = rng.integers(0, k, (2000, k)); b[i:i + 2000] = s[dr].sum(1) / c[dr].sum(1)
    return d.mean(), b.std(), np.percentile(b, 2.5), np.percentile(b, 97.5)
tot = lambda c: cv[c][0]["t_eval"]["tot"]
def sg(x): return ("$+$" if x > 0 else "$-$") + f"{abs(x):.2f}"
M = [("mean-subword (cal)", "Subword mean (cal)"), ("OMP-64 as shipped (raw)", "OMP-64 as shipped"), ("OMP-64 centred (cal)", "OMP-64 centred, cal.\\ length"), ("LS (cal)", "LS (cal)"), ("ridge[all]", "ridge[all]"), ("ridge[freq5000]", "ridge[freq5000] (selected)")]
cmp_ = {b: boot(tot(sel) - tot(b)) for b, _ in M if b != sel}
p = {b: 1 - erf(abs(v[0]) / v[1] / sqrt(2)) for b, v in cmp_.items()}
order = sorted(p, key=p.get); adj = {}; run = 0
for i, b in enumerate(order): run = max(run, min(1, p[b] * (len(order) - i))); adj[b] = run
t = "\\begin{tabular}{lrrr}\n\\toprule\nMethod & Sentence NLL & Method $-$ unchanged receiver & Ridge $-$ method \\\\\n\\midrule\n"
t += f"Unchanged receiver (words split) & {tot('original').mean():.2f} & --- & --- \\\\\n"
out = {"selected": sel, "orig_nll": float(tot("original").mean()), "methods": {}}
for b, lab in M:
    m, se, lo, hi = boot(tot(b) - tot("original")); c1 = f"{sg(m)}\\,[{sg(lo)},{sg(hi)}]" + ("$^{*}$" if lo > 0 or hi < 0 else "")
    if b == sel: c2 = "---"; d = None
    else:
        mm, se2, lo2, hi2 = cmp_[b]; c2 = f"{sg(mm)}\\,[{sg(lo2)},{sg(hi2)}]" + ("$^{*}$" if lo2 > 0 or hi2 < 0 else "") + ("$^{\\dagger}$" if adj[b] < 0.05 else "")
        d = dict(mean=mm, lo=lo2, hi=hi2, p=p[b], holm=adj[b])
    t += f"{lab} & {tot(b).mean():.2f} & {c1} & {c2} \\\\\n"; out["methods"][b] = dict(nll=float(tot(b).mean()), vs_orig=[m, lo, hi], ridge_minus=d)
t += "\\bottomrule\n\\end{tabular}\n"; open(OUT + "/tables/b7_zeroshot.tex", "w").write(t)
out["map_s"] = r["map_s"]; out["scales"] = r["scales"]; out["minutes"] = r["minutes"]
out["controls_vs_orig"] = {k: {c: float(cv[k][0][c]["tot"].mean() - cv["original"][0][c]["tot"].mean()) for c in ("c_dom", "c_wiki")} for k in cv if k != "original"}
json.dump(out, open(OUT + "/stats_7b.json", "w"), indent=1, default=float); print(t)
