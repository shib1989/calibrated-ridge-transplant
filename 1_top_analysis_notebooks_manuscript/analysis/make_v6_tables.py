"""Builds the tables, the figure and the numbers quoted in the text for the v6 (training-only word selection) results, the v9
learned-baseline runs and the medical-abstracts downstream task.
Inputs (default: ../results/..., override with env vars V6, IDS, V9, DS, OUT):  V6 = folder with the 17 v6 pkls (names without ' (1)'),  IDS = folder with abstract_ids/*.json,
V9 = folder with the 5 v9 pkls, DS = folder with the downstream json files.
Outputs: tables/*.tex, figures/fig_realvocab.pdf and stats_v6.json (all numbers used in the text)."""
import pickle, json, glob, os, re, collections, sys
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
V6 = os.environ.get("V6", os.path.join(R, "v6_trainonly"))
IDS = os.environ.get("IDS", os.path.join(R, "abstract_ids"))
V9 = os.environ.get("V9", os.path.join(R, "v9_learned"))
DS = os.environ.get("DS", os.path.join(R, "downstream_med_abstracts"))
OUT = os.environ.get("OUT", os.path.join(HERE, "out_v6"))   # tables/, figures/ and stats_v6.json are written here
os.makedirs(OUT + "/tables", exist_ok=True); os.makedirs(OUT + "/figures", exist_ok=True)
sys.path.insert(0, HERE)
B = 5000
STATS = {}

def ids_for(fn):
    return np.array(json.load(open(f"{IDS}/v6_trainonly__{fn}.json"))["abstract_id"])

def boot(d, ids, rng, B=B):
    """paired difference d; cluster bootstrap by abstract id (ids=None: by sentence). Returns mean, se, lo, hi."""
    if ids is None:
        n = len(d); b = d[rng.integers(0, n, (B, n))].mean(1)
    else:
        u, inv = np.unique(ids, return_inverse=True)
        s = np.bincount(inv, weights=d); c = np.bincount(inv).astype(float)
        k = len(u); draw = rng.integers(0, k, (B, k)); b = s[draw].sum(1) / c[draw].sum(1)
    return d.mean(), b.std(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def holm(pvals):
    K = len(pvals); idx = np.argsort(pvals); rej = np.zeros(K, bool)
    for r, i in enumerate(idx):
        if pvals[i] <= 0.05 / (K - r): rej[i] = True
        else: break
    return rej

def sg(x, nd=2): return ("$+$" if x > 0 else "$-$") + f"{abs(x):.{nd}f}"
def sig(lo, hi): return lo > 0 or hi < 0
def fmt_big(v): return f"{v:+.0f}" if abs(v) >= 100 else (f"{v:+.1f}" if abs(v) >= 10 else f"{v:+.2f}")
def tex(s): return s.replace("-", "$-$", 1) if s.startswith("-") else s.replace("+", "$+$", 1)
def write(name, s): open(f"{OUT}/tables/{name}.tex", "w").write(s); print("wrote tables/%s.tex" % name)

def load_runs():
    runs = {}
    for f in sorted(glob.glob(f"{V6}/*.pkl")):
        fn = os.path.basename(f); runs[fn] = pickle.load(open(f, "rb"))
    return runs

runs = load_runs()
SMALL = [("GPT-2 <- Pythia-160m", "GPT-2_from_Pythia-160m"), ("GPT-2 <- Qwen2.5-0.5B", "GPT-2_from_Qwen2.5-0.5B"),
         ("OPT-125m <- Pythia-160m", "OPT-125m_from_Pythia-160m"), ("OPT-125m <- Qwen2.5-0.5B", "OPT-125m_from_Qwen2.5-0.5B")]
SCALE = [("GPT-2 large <- Qwen2.5-0.5B", "GPT-2_large_from_Qwen2.5-0.5B"), ("Pythia-1B <- Qwen2.5-0.5B", "Pythia-1B_from_Qwen2.5-0.5B"),
         ("Qwen2.5-1.5B <- Pythia-410m", "Qwen2.5-1.5B_from_Pythia-410m")]
fname = lambda key, seed: f"pubmed__{key}__seed{seed}.pkl"
tl = lambda p: p.replace(" <- ", " $\\leftarrow$ ")
tot = lambda r, k, st: r["curves"][k][st]["t_eval"]["tot"]
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ 1. real vocabulary expansion, 12 runs (4 pairs x seeds 0-2)
RV = []   # one record per (run, step)
for pn, key in SMALL:
    for seed in (0, 1, 2):
        fn = fname(key, seed); r = runs[fn]; sel = r["sel"]; ids = ids_for(fn)
        for st in (0, 200):
            rec = dict(pair=pn, seed=seed, st=st, K=r["K"], sel=sel)
            for nm, k in (("unch", "original"), ("omp64c", "OMP-64 centred (raw)"), ("omp64cal", "OMP-64 centred (cal)"), ("omp64s", "OMP-64 as shipped (raw)"),
                          ("omp32", "OMP-32"), ("msub", "mean-subword (cal)")):
                base = tot(r, "original", 0) if k == "original" else tot(r, k, st)
                m, se, lo, hi = boot(tot(r, sel, st) - base, ids, rng); rec[nm] = (m, lo, hi, se, 2 * (1 - norm.cdf(abs(m / se))))
            RV.append(rec)
for comp in ("omp64c", "omp64cal"):
    rej = holm([x[comp][4] for x in RV])
    STATS[f"rv_{comp}_n"] = len(RV); STATS[f"rv_{comp}_neg_ci"] = int(sum(x[comp][2] < 0 for x in RV))
    STATS[f"rv_{comp}_holm"] = int(rej.sum())
    for st in (0, 200):
        d = [x[comp][0] for x in RV if x["st"] == st]
        STATS[f"rv_{comp}_st{st}_range"] = (float(min(d)), float(max(d)), float(np.median(d)))
rows = []
comps = [("omp64c", "OMP-64, centred inputs (official)"), ("omp64cal", "OMP-64, centred, calibrated length"), ("omp64s", "OMP-64, as shipped (official)"),
         ("omp32", "OMP-32, centred (our scikit-learn)"), ("msub", "Mean of word pieces (calibrated)")]
s = "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Zero-shot} & \\multicolumn{3}{c}{After 200 steps} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nRidge minus & Median & Range & CI$<0$ & Median & Range & CI$<0$ \\\\\n\\midrule\n"
for k, n in comps:
    cells = []
    for st in (0, 200):
        xs = [x[k] for x in RV if x["st"] == st]; d = np.array([t[0] for t in xs])
        cells += [tex(fmt_big(np.median(d))), f"{tex(fmt_big(d.min()))} to {tex(fmt_big(d.max()))}", f"{sum(t[2] < 0 for t in xs)}/{len(xs)}"]
        STATS[f"rv_{k}_st{st}_cilt0"] = int(sum(t[2] < 0 for t in xs)); STATS[f"rv_{k}_st{st}_n"] = len(xs)
    s += n + " & " + " & ".join(cells) + " \\\\\n"
s += "\\midrule\n"
for st, lab in ((0, "Zero-shot"), (200, "After 200 steps")):
    xs = [x["unch"] for x in RV if x["st"] == st]
    better, worse = sum(t[2] < 0 for t in xs), sum(t[1] > 0 for t in xs)
    STATS[f"rv_unch_st{st}"] = dict(better=int(better), level=int(len(xs) - better - worse), worse=int(worse), n=len(xs))
    s += f"\\multicolumn{{7}}{{l}}{{{lab}, ridge vs.\\ unchanged receiver: better in {better}, level in {len(xs) - better - worse}, worse in {worse} of {len(xs)} runs}} \\\\\n"
write("realvocab_summary", s + "\\bottomrule\n\\end{tabular}\n")
# full table
fm = lambda t: tex(fmt_big(t[0])) + ("$^{*}$" if sig(t[1], t[2]) else "")
by = collections.OrderedDict()
for x in RV: by.setdefault((x["pair"], x["seed"]), {})[x["st"]] = x
s = "\\begin{tabular}{lrr" + "r" * 8 + "}\n\\toprule\n & & & \\multicolumn{4}{c}{Zero-shot} & \\multicolumn{4}{c}{After 200 steps} \\\\\n\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}\nReceiver$\\leftarrow$donor & Seed & Words & vs.\\ unch. & $-$OMP-64c & $-$OMP-64s & $-$OMP-32 & vs.\\ unch. & $-$OMP-64c & $-$OMP-64s & $-$OMP-32 \\\\\n\\midrule\n"
for (p, sd), v in by.items():
    pn = p.replace(" <- ", "$\\leftarrow$").replace("Qwen2.5-0.5B", "Qwen2.5").replace("-160m", "")
    s += f"{pn} & {sd} & {v[0]['K']} & " + " & ".join(fm(x[k]) for x in (v[0], v[200]) for k in ("unch", "omp64c", "omp64s", "omp32")) + " \\\\\n"
write("realvocab_full", s + "\\bottomrule\n\\end{tabular}\n")
# figure
BLUE, INK = "#1f5fa8", "#222222"
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.6), sharey=True)
keys = list(by.keys()); ys = np.arange(len(keys))[::-1]
lab = lambda k: f"{k[0].replace(' <- ', '←').replace('Qwen2.5-0.5B', 'Qwen2.5').replace('-160m', '')} s{k[1]}"
for ax, st, ttl in zip(axs, (0, 200), ("Zero-shot", "After 200 steps (new rows trained)")):
    for y, k in zip(ys, keys):
        t = by[k][st]["omp64c"]; ax.plot([t[1], t[2]], [y, y], color=BLUE, lw=1.3); ax.scatter(t[0], y, s=14, color=BLUE, zorder=3)
    ax.axvline(0, color=INK, lw=0.8); ax.set_title(ttl, fontsize=8.5); ax.grid(axis="x", color="#ececE8", lw=0.6); ax.set_xlabel("Ridge minus OMP-64 (centred), nats / sentence")
axs[0].set_yticks(ys); axs[0].set_yticklabels([lab(k) for k in keys], fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig_realvocab.pdf"); fig.savefig(f"{OUT}/figures/fig_realvocab.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 2. seven pairs, seed 1: baselines (98 comparisons)
FAM = ["OMP-64 centred (cal)", "OMP-64 as shipped (raw)", "LS (cal)", "Procrustes (cal)", "SALT-ours (cal)", "WECHSEL (cal)", "OMP-32"]
BASE = [("OMP-64 centred (cal)", "OMP-64c cal."), ("OMP-64 as shipped (raw)", "OMP-64s"), ("LS (cal)", "LS"), ("Procrustes (cal)", "Procrustes"),
        ("SALT-ours (cal)", "SALT-ours"), ("WECHSEL (cal)", "WECHSEL"), ("original", "Unch.")]
SC = {}; fam = []
ORDER = []
for pn, key in SMALL + SCALE:
    fn = fname(key, 1); r = runs[fn]; sel = r["sel"]; ids = ids_for(fn); nb = r["curves"][sel][0]["t_eval"]["nbytes"].mean()
    SC[pn] = {"sel": sel, "K": r["K"], "n": len(tot(r, sel, 0)), "nabs": len(set(ids))}; ORDER.append(pn)
    for st in (0, 200):
        SC[pn][st] = {}
        for b, _ in BASE + [("OMP-32", "OMP-32")]:
            o = tot(r, "original", 0) if b == "original" else tot(r, b, st)
            m, se, lo, hi = boot(tot(r, sel, st) - o, ids, rng); SC[pn][st][b] = (m, lo, hi, m / np.log(2) / nb, se, 2 * (1 - norm.cdf(abs(m / se))))
            if b in FAM: fam.append(dict(pair=pn, st=st, b=b, m=m, lo=lo, hi=hi, p=SC[pn][st][b][5]))
rej = holm([f["p"] for f in fam])
for f, rj in zip(fam, rej): f["holm"] = bool(rj)
wins = [f for f in fam if f["m"] < 0 and sig(f["lo"], f["hi"])]
base_wins = [f for f in fam if f["m"] > 0 and sig(f["lo"], f["hi"])]
nsig = [f for f in fam if not sig(f["lo"], f["hi"])]
STATS["fam"] = dict(n=len(fam), wins=len(wins), wins_holm=sum(f["holm"] for f in wins), ns=len(nsig), base_wins=len(base_wins),
                    base_wins_holm=sum(f["holm"] for f in base_wins),
                    nonholm_wins=sorted(round(abs(f["m"]), 2) for f in wins if not f["holm"]),
                    base_win_list=[(f["pair"], f["st"], f["b"], round(f["m"], 2), round(f["lo"], 2), round(f["hi"], 2), f["holm"]) for f in base_wins],
                    per_base={b: [sum(1 for f in fam if f["b"] == b and f["m"] < 0 and sig(f["lo"], f["hi"])),
                                  sum(1 for f in fam if f["b"] == b and not sig(f["lo"], f["hi"])),
                                  sum(1 for f in fam if f["b"] == b and f["m"] > 0 and sig(f["lo"], f["hi"]))] for b in FAM},
                    ns_list=[(f["pair"], f["st"], f["b"], round(f["m"], 2)) for f in nsig],
                    nonholm_wins_list=[(f["pair"], f["st"], f["b"], round(f["m"], 2), round(f["lo"], 2), round(f["hi"], 2)) for f in wins if not f["holm"]])
cell = lambda t: f"{sg(t[0])}\\,[{sg(t[1])},{sg(t[2])}]" + ("$^{*}$" if sig(t[1], t[2]) else "")
mcell = lambda t: (sg(t[0]) if abs(t[0]) < 100 else sg(t[0], 0)) + ("$^{*}$" if sig(t[1], t[2]) else "")
b4 = lambda x: ("$+$" if x > 0 else "$-$") + f"{abs(x):.4f}"
def table(cols, fn, name):
    s = "\\begin{tabular}{ll" + "r" * len(cols) + "}\n\\toprule\nPair & Steps & " + " & ".join(dict(BASE)[b] for b in cols) + " \\\\\n\\midrule\n"
    for p in ORDER:
        for i, st in enumerate((0, 200)):
            s += (tl(p) if i == 0 else "") + f" & {st} & " + " & ".join(fn(SC[p][st][b]) for b in cols) + " \\\\\n"
        s += "\\addlinespace\n"
    write(name, s.rsplit("\\addlinespace\n", 1)[0] + "\\bottomrule\n\\end{tabular}\n")
ALL = [b for b, _ in BASE]
table(ALL, mcell, "scale_baselines"); table(ALL[:3], cell, "scale_ci_a"); table(ALL[3:], cell, "scale_ci_b")
s = "\\begin{tabular}{llccccccc}\n\\toprule\nPair & Steps & " + " & ".join(b[1] for b in BASE) + " \\\\\n\\midrule\n"
for p in ORDER:
    for i, st in enumerate((0, 200)):
        s += (tl(p) if i == 0 else "") + f" & {st} & " + " & ".join(b4(SC[p][st][b][3]) for b, _ in BASE) + " \\\\\n"
    s += "\\addlinespace\n"
write("scale_bpb", s.rsplit("\\addlinespace\n", 1)[0] + "\\bottomrule\n\\end{tabular}\n")
STATS["sc"] = {p: {"sel": SC[p]["sel"], "K": SC[p]["K"], "n": SC[p]["n"], "nabs": SC[p]["nabs"],
                   **{f"st{st}": {b: [round(v, 3) for v in SC[p][st][b][:3]] for b in SC[p][st]} for st in (0, 200)}} for p in ORDER}

# ------------------------------------------------------------------ 3. three seeds of the four small pairs (168 comparisons)
fam3 = []
for pn, key in SMALL:
    for seed in (0, 1, 2):
        fn = fname(key, seed); r = runs[fn]; sel = r["sel"]; ids = ids_for(fn)
        for st in (0, 200):
            for b in FAM:
                m, se, lo, hi = boot(tot(r, sel, st) - tot(r, b, st), ids, rng)
                fam3.append(dict(pair=pn, seed=seed, st=st, b=b, m=m, lo=lo, hi=hi, p=2 * (1 - norm.cdf(abs(m / se)))))
rej = holm([f["p"] for f in fam3])
for f, rj in zip(fam3, rej): f["holm"] = bool(rj)
w3 = [f for f in fam3 if f["m"] < 0 and sig(f["lo"], f["hi"])]; b3 = [f for f in fam3 if f["m"] > 0 and sig(f["lo"], f["hi"])]
cells = collections.defaultdict(int)
for f in w3: cells[(f["pair"], f["st"], f["b"])] += 1
cnt = collections.Counter(cells.values()); ncell = len(SMALL) * 2 * len(FAM)
STATS["fam3"] = dict(n=len(fam3), wins=len(w3), wins_holm=sum(f["holm"] for f in w3), ns=sum(1 for f in fam3 if not sig(f["lo"], f["hi"])), base_wins=len(b3),
                     base_wins_holm=sum(f["holm"] for f in b3), cells=ncell, c3=cnt[3], c2=cnt[2], c1=cnt[1], c0=ncell - len(cells),
                     base_win_list=[(f["pair"], f["seed"], f["st"], f["b"], round(f["m"], 2), f["holm"]) for f in b3],
                     ns_list=[(f["pair"], f["seed"], f["st"], f["b"], round(f["m"], 2)) for f in fam3 if not sig(f["lo"], f["hi"]) and not (f["m"] > 0 and sig(f["lo"], f["hi"]))])

# ------------------------------------------------------------------ 4. WECHSEL (seed 1, four small pairs)
s = "\\begin{tabular}{llcccc}\n\\toprule\n & & \\multicolumn{2}{c}{Ridge minus WECHSEL} & WECHSEL minus & Ridge minus\\\\\nPair & Steps & raw length & calibrated length & unchanged receiver & unchanged receiver\\\\\n\\midrule\n"
W = {}
for pn, key in SMALL:
    fn = fname(key, 1); r = runs[fn]; sel = r["sel"]; ids = ids_for(fn); W[pn] = {}
    for st in (0, 200):
        tt = lambda t: f"{sg(t[0])}\\,[{sg(t[1])},{sg(t[2])}]" + ("$^{*}$" if sig(t[1], t[2]) else "")
        a = boot(tot(r, sel, st) - tot(r, "WECHSEL (raw)", st), ids, rng); b = boot(tot(r, sel, st) - tot(r, "WECHSEL (cal)", st), ids, rng)
        c = boot(tot(r, "WECHSEL (raw)", st) - tot(r, "original", 0), ids, rng); d = boot(tot(r, sel, st) - tot(r, "original", 0), ids, rng)
        W[pn][st] = [[round(v, 2) for v in x[:1] + x[2:]] for x in (a, b, c, d)]
        s += (tl(pn) if st == 0 else "") + f" & {st} & " + " & ".join(tt((x[0], x[2], x[3])) for x in (a, b, c, d)) + "\\\\\n"
    s += "\\addlinespace\n"
write("wechsel", s.rsplit("\\addlinespace\n", 1)[0] + "\\bottomrule\n\\end{tabular}\n"); STATS["wechsel"] = W
STATS["wechsel_fallback"] = {pn: runs[fname(key, 1)]["wechsel_fallback"] for pn, key in SMALL}

# ------------------------------------------------------------------ 5. sequence length (seed 1)
groups = [("GPT-2, OPT-125m $\\leftarrow$ Pythia-160m", ["GPT-2 <- Pythia-160m", "OPT-125m <- Pythia-160m"]),
          ("GPT-2, OPT-125m, GPT-2 large $\\leftarrow$ Qwen2.5-0.5B", ["GPT-2 <- Qwen2.5-0.5B", "OPT-125m <- Qwen2.5-0.5B", "GPT-2 large <- Qwen2.5-0.5B"]),
          ("Pythia-1B $\\leftarrow$ Qwen2.5-0.5B", ["Pythia-1B <- Qwen2.5-0.5B"]), ("Qwen2.5-1.5B $\\leftarrow$ Pythia-410m", ["Qwen2.5-1.5B <- Pythia-410m"])]
keyof = dict(SMALL + SCALE)
s = "\\begin{tabular}{lrrrrr}\n\\toprule\nReceiver $\\leftarrow$ donor & New words & Split & New tokens & Saved (\\%) & Saved tokens/byte \\\\\n\\midrule\n"
SQ = {}
for gname, ps in groups:
    vals = []
    for p in ps:
        r = runs[fname(keyof[p], 1)]; sel = r["sel"]; o = r["curves"]["original"][0]["t_eval"]; n = r["curves"][sel][0]["t_eval"]
        vals.append((o["ntok"].mean(), n["ntok"].mean(), (o["ntok"] - n["ntok"]).mean() / o["nbytes"].mean(), r["K"]))
        # control sentences must be tokenised identically before and after
        for ck in ("c_dom", "c_wiki"):
            SQ.setdefault("control_equal", []).append(bool(np.array_equal(r["curves"]["original"][0][ck]["ntok"], r["curves"][sel][0][ck]["ntok"])))
    assert max(v[0] for v in vals) - min(v[0] for v in vals) < 1e-9 and max(v[1] for v in vals) - min(v[1] for v in vals) < 1e-9, gname
    o_, n_, pb, K = vals[0]; SQ[gname] = (K, round(o_, 1), round(n_, 1), round(100 * (o_ - n_) / o_, 1), round(pb, 4))
    s += f"{gname} & {K} & {o_:.1f} & {n_:.1f} & {100 * (o_ - n_) / o_:.1f} & {pb:.3f} \\\\\n"
write("seqlen", s + "\\bottomrule\n\\end{tabular}\n"); STATS["seqlen"] = SQ

# ------------------------------------------------------------------ 6. learned baselines (v9)
V9F = {"GPT-2 <- Pythia-160m": "pubmed__GPT-2_from_Pythia-160m__seed1.pkl", "GPT-2 <- Qwen2.5-0.5B": "pubmed__GPT-2_from_Qwen2.5-0.5B__seed1.pkl",
       "OPT-125m <- Pythia-160m": "pubmed__OPT-125m_from_Pythia-160m__seed1.pkl", "OPT-125m <- Qwen2.5-0.5B": "pubmed__OPT-125m_from_Qwen2.5-0.5B__seed1.pkl",
       "GPT-2 large <- Qwen2.5-0.5B": "pubmed__GPT-2_large_from_Qwen2.5-0.5B__seed1.pkl"}
LB = [("AweDist (official)", "AweDist (official)"), ("AweDist (official, cal)", "AweDist, cal."), ("TokDist (ours)", "TokDist (ours)"), ("TokDist (ours, cal)", "TokDist, cal.")]
lrows, tim = [], {}
for pn, f in V9F.items():
    r = pickle.load(open(glob.glob(f"{V9}/**/{f}", recursive=True)[0], "rb")); sel = r["sel"]; r6 = runs[fname(keyof[pn], 1)]
    assert np.array_equal(r["curves"]["original"][0]["t_eval"]["ntok"], r6["curves"]["original"][0]["t_eval"]["ntok"])   # same evaluation sentences as v6 seed 1
    ids = ids_for(fname(keyof[pn], 1))
    tim[pn] = dict(ridge_map=r["map_s"]["ridge[all]"] + r["map_s"]["ridge[freq5000]"], ridge_cal=r["calib_s"]["ridge[all]"] + r["calib_s"]["ridge[freq5000]"],   # both anchor subsets: the selection needs both
                   ridge_sel_only=r["map_s"][sel] + r["calib_s"][sel], awedist=r["calib_s"]["AweDist (official)"], tokdist=r["calib_s"]["TokDist (ours)"], sel=sel)
    for st in (0, 200):
        for b, _ in LB:
            m, se, lo, hi = boot(tot(r, sel, st) - tot(r, b, st), ids, rng)
            lrows.append(dict(pair=pn, st=st, b=b, m=m, lo=lo, hi=hi, p=2 * (1 - norm.cdf(abs(m / se)))))
rej = holm([x["p"] for x in lrows])
for x, rj in zip(lrows, rej): x["holm"] = bool(rj)
STATS["learned"] = dict(n=len(lrows), wins=sum(1 for x in lrows if x["m"] < 0 and sig(x["lo"], x["hi"])), wins_holm=sum(1 for x in lrows if x["m"] < 0 and sig(x["lo"], x["hi"]) and x["holm"]),
                        not_sig=[(x["pair"], x["st"], x["b"], round(x["m"], 2), round(x["lo"], 2), round(x["hi"], 2)) for x in lrows if not sig(x["lo"], x["hi"])],
                        timing={p: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items()} for p, d in tim.items()},
                        rows=[(x["pair"], x["st"], x["b"], round(x["m"], 2), round(x["lo"], 2), round(x["hi"], 2), x["holm"]) for x in lrows])
s = "\\begin{tabular}{llrrrr}\n\\toprule\nPair & Steps & " + " & ".join(n for _, n in LB) + " \\\\\n\\midrule\n"
for p in V9F:
    for i, st in enumerate((0, 200)):
        cs = []
        for b, _ in LB:
            x = next(y for y in lrows if y["pair"] == p and y["st"] == st and y["b"] == b)
            cs.append((sg(x["m"]) if abs(x["m"]) < 100 else sg(x["m"], 0)) + ("$^{*}$" if sig(x["lo"], x["hi"]) else "") + ("$^{\\dagger}$" if (sig(x["lo"], x["hi"]) and x["holm"]) else ""))
        s += (tl(p) if i == 0 else "") + f" & {st} & " + " & ".join(cs) + " \\\\\n"
    s += "\\addlinespace\n"
write("learned_baselines", s.rsplit("\\addlinespace\n", 1)[0] + "\\bottomrule\n\\end{tabular}\n")
s = "\\begin{tabular}{lrrrr}\n\\toprule\nPair & Ridge (both subsets) & AweDist (official) & TokDist (ours) & Selected ridge \\\\\n\\midrule\n"
for p, d in tim.items():
    s += f"{tl(p)} & {d['ridge_map'] + d['ridge_cal']:.0f} & {d['awedist']:.0f} & {d['tokdist']:.0f} & {d['sel'].replace('ridge[', '').replace(']', '')} \\\\\n"
write("learned_time", s + "\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ 7. downstream, medical abstracts
PAIRS = [("GPT_2_Pythia_tokenizer", "GPT-2 $\\leftarrow$ Pythia-160m"), ("GPT_2_Qwen_tokenizer", "GPT-2 $\\leftarrow$ Qwen2.5-0.5B")]
M = collections.OrderedDict([("ridge", "ridge_sel_ours_calibrated"), ("OMP-64 centred", "OMP_64_official_centred_inputs"), ("OMP-64 as shipped", "OMP_64_official"),
     ("FOCUS", "FOCUS_official"), ("Subword mean", "subword_mean_official"), ("Mean fill", "mean_fill"), ("Native tokenizer", "own_tokenizer_reference")])
def dsload(p, m):
    out = [json.load(open(f"{DS}/{p}__{M[m]}__seed{s}.json")) for s in range(3)]; g = np.array(out[0]["gold"])
    assert all((np.array(o["gold"]) == g).all() for o in out); return out
def corr(p, m, ep): return np.mean([(np.array(o["epochs"][ep]["preds"]) == np.array(o["gold"])).astype(float) for o in dsload(p, m)], 0)
drows = []
for p, _ in PAIRS:
    for ep in (0, 2):
        r_ = corr(p, "ridge", ep)
        for m in list(M)[1:]:
            df = r_ - corr(p, m, ep); mm, se, lo, hi = boot(df, None, rng, B=100000)
            drows.append(dict(pair=p, ep=ep, m=m, d=mm, lo=lo, hi=hi, p=2 * (1 - norm.cdf(abs(mm / se)))))
hl = {}
for ep in (0, 2):   # Holm separately for the first and the last epoch: ridge against five baselines in two pairs = 10 comparisons each (the native tokenizer is a reference, not a baseline)
    ref = [x for x in drows if x["m"] != "Native tokenizer" and x["ep"] == ep]
    ps = np.array([x["p"] for x in ref]); order = np.argsort(ps); adj = np.zeros(len(ps)); run = 0.0
    for rank, i in enumerate(order): run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run   # Holm-adjusted p
    for x, a in zip(ref, adj): x["adj"] = float(a); hl[id(x)] = bool(a < 0.05)
STATS["downstream"] = dict(holm_family=10, rows=[(x["pair"], x["ep"] + 1, x["m"], round(100 * x["d"], 2), round(100 * x["lo"], 2), round(100 * x["hi"], 2), hl.get(id(x)), round(x.get("adj", float("nan")), 4)) for x in drows],
                           majority=0.351)
s = "\\begin{tabular}{l" + "rr" * 2 + "}\n\\toprule\n & \\multicolumn{2}{c}{GPT-2$\\leftarrow$Pythia-160m} & \\multicolumn{2}{c}{GPT-2$\\leftarrow$Qwen2.5-0.5B} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nMethod & Acc.\\ (\\%) & Ridge minus (points) & Acc.\\ (\\%) & Ridge minus (points) \\\\\n\\midrule\n"
for m in M:
    cs = []
    for p, _ in PAIRS:
        accs = [np.mean([o["epochs"][2]["acc"] for o in [dsload(p, m)[s_] for s_ in range(3)]]) * 100]
        a = accs[0]; sd = np.std([o["epochs"][2]["acc"] for o in dsload(p, m)], ddof=1) * 100
        if m == "ridge": cs += [f"{a:.1f}$\\pm${sd:.1f}", "--"]
        else:
            x = next(y for y in drows if y["pair"] == p and y["ep"] == 2 and y["m"] == m)
            mark = ("$^{*}$" if sig(x["lo"], x["hi"]) else "") + ("$^{\\dagger}$" if hl.get(id(x)) and sig(x["lo"], x["hi"]) else "")
            cs += [f"{a:.1f}$\\pm${sd:.1f}", f"{sg(100 * x['d'], 1)}\\,[{sg(100 * x['lo'], 1)},{sg(100 * x['hi'], 1)}]{mark}"]
    s += m.replace("ridge", "Ridge (ours)") + " & " + " & ".join(cs) + " \\\\\n"
write("downstream_med", s + "\\bottomrule\n\\end{tabular}\n")

json.dump(STATS, open(f"{OUT}/stats_v6.json", "w"), indent=1, default=str)
print("done")
