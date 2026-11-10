"""Builds the LaTeX tables (tables/*.tex) and two new figures (figures/*.pdf, .png) for main.tex from the saved result files."""
import csv, pickle, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "../paper1_pack/tables/"
BLUE, GRAY, INK, ORANGE = "#2a78d6", "#8a8f98", "#1a1a1a", "#d9822b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
arr = lambda s: s.replace("<-", r"$\leftarrow$").replace("←", r"$\leftarrow$")
import re
def w(name, s):
    s = re.sub(r"(?<![\w${}^.])-(?=\d)", "$-$", s); s = re.sub(r"(?<![\w${}^.])\+(?=\d)", "$+$", s)   # proper minus / plus signs in numbers
    open(f"tables/{name}.tex", "w").write(s)

# ---------- Table: zero-shot NLL ----------
rows = list(csv.DictReader(open(T + "table1_full_all_methods.csv")))
P = ["GPT-2 <- Pythia tokenizer", "GPT-2 <- Qwen tokenizer", "GPT-2 medium <- Pythia tokenizer", "Qwen2.5-0.5B <- GPT-2 tokenizer"]
M = [("base (own tokenizer)", "Unchanged model (own tokenizer)$^{\\dagger}$", False), ("OMP-64 (official)", "OMP-64, as shipped", True), ("OMP-64 (official, centred inputs)", "OMP-64, centred inputs", True),
     ("FOCUS (official)", "FOCUS", True), ("subword-mean (official)", "Subword mean", True), ("mean (official)", "Mean", True), ("john-hewitt (official)", "John Hewitt", True),
     ("common-interp (official)", "Common interpolation", True), ("ridge[all] uncalibrated", "Ridge, uncalibrated length", True), ("ridge[sel] (ours, calibrated)", "\\textbf{Ridge, calibrated (ours)}", True)]
V = {(r["pair"], r["method"]): (float(r["domain_NLL"]), float(r["general_NLL"])) for r in rows}
best = {(p, k): min(V[(p, m)][k] for m, _, tr in M if tr and (p, m) in V) for p in P for k in (0, 1)}
s = "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{GPT-2$\\leftarrow$Pythia} & \\multicolumn{2}{c}{GPT-2$\\leftarrow$Qwen2.5} & \\multicolumn{2}{c}{GPT-2 med.$\\leftarrow$Pythia} & \\multicolumn{2}{c}{Qwen2.5$\\leftarrow$GPT-2} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nMethod & Dom & Gen & Dom & Gen & Dom & Gen & Dom & Gen \\\\\n\\midrule\n"
for m, nm, tr in M:
    cells = []
    for p in P:
        for k in (0, 1):
            v = V[(p, m)][k]; t = f"{v:.1f}"
            if tr and abs(v - best[(p, k)]) < 1e-9: t = "\\textbf{" + t + "}"
            cells.append(t)
    s += nm + " & " + " & ".join(cells) + " \\\\\n"
    if m == "base (own tokenizer)": s += "\\midrule\n"
w("zero_shot_nll", s + "\\bottomrule\n\\end{tabular}\n")

# ---------- Table: fine-tuning ----------
s = "\\begin{tabular}{llrrr}\n\\toprule\nPair & Method & Seeds & Accuracy (mean $\\pm$ sd) & Macro-F1 \\\\\n\\midrule\n"
fr = list(csv.DictReader(open(T + "table2_finetuned.csv"))); prev = None
nm = {"Own tokenizer (reference)": "Native tokenizer (reference)", "ridge, calibrated (ours)": "\\textbf{Ridge, calibrated (ours)}", "OMP-64 (as shipped)": "OMP-64, as shipped", "OMP-64 (centred inputs)": "OMP-64, centred inputs", "subword-mean": "Subword mean", "mean-fill": "Mean fill"}
for r in fr:
    if prev and r["Pair"] != prev: s += "\\midrule\n"
    prev = r["Pair"]; s += f"{arr(r['Pair'])} & {nm.get(r['Method'], r['Method'])} & {r['Seeds']} & {r['Accuracy (mean ± sd)'].replace('±', '$\\pm$')} & {r['Macro-F1']} \\\\\n"
w("finetune", s + "\\bottomrule\n\\end{tabular}\n")

# ---------- Table: frozen probe accuracy (appendix) ----------
fp = list(csv.reader(open(T + "table3_frozen_probe.csv"))); hdr = fp[0]
s = "\\begin{tabular}{l" + "r" * 9 + "}\n\\toprule\n & \\multicolumn{3}{c}{GPT-2$\\leftarrow$Pythia} & \\multicolumn{3}{c}{GPT-2$\\leftarrow$Qwen2.5} & \\multicolumn{3}{c}{Qwen2.5$\\leftarrow$GPT-2} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nMethod & emb & mid & last & emb & mid & last & emb & mid & last \\\\\n\\midrule\n"
for r in fp[1:]:
    n = nm.get(r[0], r[0]).replace("OMP-64 (as shipped)", "OMP-64, as shipped").replace("OMP-64 (centred inputs)", "OMP-64, centred inputs")
    s += n + " & " + " & ".join(r[1:]) + " \\\\\n"
w("frozen_accuracy", s + "\\bottomrule\n\\end{tabular}\n")

# ---------- Table: frozen probe, ridge minus baseline, Holm ----------
NT = pickle.load(open("/tmp/new_tables.pkl", "rb")); R = NT["rows"]
B = [("OMP-64 (official, centred inputs)", "OMP-64 centred"), ("OMP-64 (official)", "OMP-64 shipped"), ("FOCUS (official)", "FOCUS"), ("subword-mean (official)", "Subw.\\ mean"), ("mean-fill", "Mean fill"), ("own tokenizer (reference)", "Native tok.")]
s = "\\begin{tabular}{llrrrrrr}\n\\toprule\nPair & Layer & " + " & ".join(b[1] for b in B) + " \\\\\n\\midrule\n"
PP = ["GPT-2 <- Pythia tokenizer", "GPT-2 <- Qwen tokenizer", "Qwen2.5-0.5B <- GPT-2 tokenizer"]; short = {PP[0]: "GPT-2$\\leftarrow$Pythia", PP[1]: "GPT-2$\\leftarrow$Qwen2.5", PP[2]: "Qwen2.5$\\leftarrow$GPT-2"}
for p in PP:
    for L in ("emb", "mid", "last"):
        c = []
        for m, _ in B:
            r = next(x for x in R if x[0] == p and x[1] == L and x[2] == m)
            c.append(f"{r[3]:+.2f}" + ("$^{\\dagger}$" if r[7] < .05 else "$^{*}$" if r[6] < .05 else ""))
        s += f"{short[p] if L == 'emb' else ''} & {L} & " + " & ".join(c) + " \\\\\n"
    if p != PP[-1]: s += "\\midrule\n"
w("frozen_holm", s + "\\bottomrule\n\\end{tabular}\n")

# ---------- Real vocabulary expansion ----------
rv = pickle.load(open("/tmp/rv/rows3.pkl", "rb"))
ci = lambda t: (t[1] > 0 or t[2] < 0)
def summ(key, stage):
    xs = [x[key] for x in rv if x["steps"] == stage]; d = np.array([t[0] for t in xs])
    return f"{np.median(d):+.2f}", f"{d.min():+.2f} to {d.max():+.2f}", f"{sum(t[2] < 0 for t in xs)}/{len(xs)}", f"{sum(t[0] < 0 for t in xs)}/{len(xs)}"
comps = [("r_omp64c", "OMP-64, centred inputs (official)"), ("r_omp64cal", "OMP-64, centred, calibrated length"), ("r_omp64s", "OMP-64, as shipped (official)"), ("r_omp32", "OMP-32, centred (our scikit-learn)"), ("r_msub", "Mean of word pieces (calibrated)")]
s = "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Zero-shot} & \\multicolumn{3}{c}{After 200 steps} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nRidge minus & Median & Range & CI$<0$ & Median & Range & CI$<0$ \\\\\n\\midrule\n"
def rng(k, st):
    d = [x[k][0] for x in rv if x["steps"] == st]; f = lambda v: f"{v:+.0f}" if abs(v) >= 100 else (f"{v:+.1f}" if abs(v) >= 10 else f"{v:+.2f}")
    return f"{f(min(d))} to {f(max(d))}", f(float(np.median(d)))
for k, n in comps:
    a, b = rng(k, 0), rng(k, 200)
    s += f"{n} & {a[1]} & {a[0]} & {summ(k, 0)[2]} & {b[1]} & {b[0]} & {summ(k, 200)[2]} \\\\\n"
s += "\\midrule\n"
for st, lab in ((0, "Zero-shot"), (200, "After 200 steps")):
    xs = [x["ridge_ref"] for x in rv if x["steps"] == st]
    better, worse = sum(t[2] < 0 for t in xs), sum(t[1] > 0 for t in xs)
    s += f"\\multicolumn{{7}}{{l}}{{{lab}, ridge vs.\\ unchanged receiver: better in {better}, level in {len(xs) - better - worse}, worse in {worse} of {len(xs)} runs}} \\\\\n"
w("realvocab_summary", s + "\\bottomrule\n\\end{tabular}\n")
fm = lambda t: (f"{t[0]:+.0f}" if abs(t[0]) >= 100 else f"{t[0]:+.2f}") + ("$^{*}$" if ci(t) else "")
by = collections.OrderedDict()
for x in rv: by.setdefault((x["pair"], x["domain"], x["seed"]), {})[x["steps"]] = x
s = "\\begin{tabular}{llrr" + "r" * 8 + "}\n\\toprule\n & & & & \\multicolumn{4}{c}{Zero-shot} & \\multicolumn{4}{c}{After 200 steps} \\\\\n\\cmidrule(lr){5-8}\\cmidrule(lr){9-12}\nReceiver$\\leftarrow$donor & Text & Seed & Words & vs.\\ unch. & $-$OMP-64c & $-$OMP-64s & $-$OMP-32 & vs.\\ unch. & $-$OMP-64c & $-$OMP-64s & $-$OMP-32 \\\\\n\\midrule\n"
for (p, d, sd), v in by.items():
    pn = p.replace(" <- ", "$\\leftarrow$").replace("Qwen2.5-0.5B", "Qwen2.5").replace("-160m", "")
    s += f"{pn} & {d} & {sd} & {v[0]['K']} & " + " & ".join(fm(x[k]) for x in (v[0], v[200]) for k in ("ridge_ref", "r_omp64c", "r_omp64s", "r_omp32")) + " \\\\\n"
w("realvocab_full", s + "\\bottomrule\n\\end{tabular}\n")

# ---------- Ablation ----------
ab = list(csv.DictReader(open(T + "table9_ablation.csv"))); AB = {(r["variant"], r["pair"], int(r["steps"])): (float(r["diff_vs_ridge_all_nats"]), float(r["ci_lo"]), float(r["ci_hi"])) for r in ab}
names = [("ridge[freq5000]", "Ridge on the 5000 most frequent anchors"), ("A no-centring (cal)", "No centring, calibrated length"), ("A no-centring (raw output)", "No centring, raw output"), ("B own magnitude (no unit, no cal)", "Own magnitude (no unit direction, no calibration)"),
         ("B fixed scale 1.0 (no cal)", "Unit direction, fixed length 1.0"), ("C lambda=0.0001 (cal)", "$\\lambda=10^{-4}$"), ("C lambda=0.001 (cal)", "$\\lambda=10^{-3}$"), ("C lambda=0.01 (cal)", "$\\lambda=10^{-2}$"), ("C lambda=0.1 (cal)", "$\\lambda=0.1$"), ("C lambda=1 (cal)", "$\\lambda=1$"), ("mean-subword (cal)", "Mean of word pieces (calibrated)")]
pr = ["GPT-2 <- Pythia", "GPT-2 <- Qwen"]
fa = lambda t: (f"{t[0]:+.0f}" if abs(t[0]) >= 100 else f"{t[0]:+.2f}") + ("$^{*}$" if (t[1] > 0 or t[2] < 0) else "")
s = "\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{GPT-2$\\leftarrow$Pythia} & \\multicolumn{2}{c}{GPT-2$\\leftarrow$Qwen2.5} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nVariant (change from ridge fitted on all anchors) & Zero-shot & 200 steps & Zero-shot & 200 steps \\\\\n\\midrule\n"
s += "Full ridge vs.\\ unchanged receiver (reference) & " + " & ".join(fa(AB[("ref", p, st)]) for p in pr for st in (0, 200)) + " \\\\\n\\midrule\n"
for k, n in names: s += n + " & " + " & ".join(fa(AB[(k, p, st)]) for p in pr for st in (0, 200)) + " \\\\\n"
w("ablation", s + "\\bottomrule\n\\end{tabular}\n")

# ---------- Speed ----------
sp = list(csv.DictReader(open(T + "table10_speed_benchmark_T4.csv"))); SP = {(r["pair"], int(r["K"]), r["device"], r["dtype"], r["method"]): float(r["median_s"]) for r in sp}
pp = ["GPT-2 <- Pythia-160m", "GPT-2 <- Qwen2.5-0.5B"]
s = "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{CPU (2 threads), s} & \\multicolumn{3}{c}{GPU (Tesla T4), s} & \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\nPair & $K$ & OMP-64 & Ridge fixed & Ridge search & Ratio & OMP-64 & Ridge fixed & Ridge search & Ratio \\\\\n\\midrule\n".replace("\\multicolumn{3}{c}{GPU (Tesla T4), s} & \\\\", "\\multicolumn{4}{c}{GPU (Tesla T4), s} \\\\").replace("\\cmidrule(lr){7-9}", "\\cmidrule(lr){7-10}").replace("{lrrrrrrrr}", "{lrrrrrrrrr}")
for p in pp:
    for K in (82, 300, 1000, 3000):
        c = []
        for dev in ("cpu", "cuda"):
            o, a, b = (SP[(p, K, dev, "float32", m)] for m in ("OMP-64 official", "ridge, fixed lambda (all anchors)", "ridge, lambda searched (as in paper)"))
            c += [f"{o:.2f}", f"{a:.3f}", f"{b:.2f}", f"{o / a:.0f}$\\times$"]
        s += f"{arr(p).replace('-160m','').replace('-0.5B','') if K == 82 else ''} & {K} & " + " & ".join(c) + " \\\\\n"
    if p != pp[-1]: s += "\\midrule\n"
w("speed", s + "\\bottomrule\n\\end{tabular}\n")

# ---------- Figure: speed scaling ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=False)
Ks = [82, 300, 1000, 3000]
for ax, dev, ttl in zip(axs, ("cpu", "cuda"), ("CPU (2 threads)", "GPU (Tesla T4)")):
    for m, lab, col, ls in (("OMP-64 official", "OMP-64 (official)", ORANGE, "-"), ("OMP-32 official", "OMP-32 (official)", ORANGE, "--"), ("ridge, lambda searched (as in paper)", "Ridge, $\\lambda$ searched", BLUE, "--"), ("ridge, fixed lambda (all anchors)", "Ridge, fixed $\\lambda$", BLUE, "-")):
        y = [np.mean([SP[(p, K, dev, "float32", m)] for p in pp]) for K in Ks]; ax.plot(Ks, y, ls, color=col, marker="o", ms=3, lw=1.3, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.set_xlabel("New tokens $K$"); ax.set_title(ttl, fontsize=8.5); ax.grid(color="#ececE8", lw=0.6)
axs[0].set_ylabel("Mapping time (s, float32)"); axs[0].legend(fontsize=6.8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig_speed.pdf"); fig.savefig("figures/fig_speed.png", dpi=200); plt.close(fig)

# ---------- Figure: real vocab, ridge minus official OMP-64 centred ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 4.0), sharey=True)
keys = list(by.keys()); ys = np.arange(len(keys))[::-1]
lab = lambda k: f"{k[0].replace(' <- ', '←').replace('Qwen2.5-0.5B', 'Qwen2.5').replace('-160m', '')} {k[1]} s{k[2]}"
for ax, st, ttl in zip(axs, (0, 200), ("Zero-shot", "After 200 steps (new rows trained)")):
    for y, k in zip(ys, keys):
        t = by[k][st]["r_omp64c"]; ax.plot([t[1], t[2]], [y, y], color=BLUE, lw=1.3); ax.scatter(t[0], y, s=14, color=BLUE, zorder=3)
    ax.axvline(0, color=INK, lw=0.8); ax.set_title(ttl, fontsize=8.5); ax.grid(axis="x", color="#ececE8", lw=0.6); ax.set_xlabel("Ridge minus OMP-64 (centred), nats / sentence")
axs[0].set_yticks(ys); axs[0].set_yticklabels([lab(k) for k in keys], fontsize=6.5)
fig.tight_layout(); fig.savefig("figures/fig_realvocab.pdf"); fig.savefig("figures/fig_realvocab.png", dpi=200); plt.close(fig)
print("assets ok")
