"""Tables and figure of the revision (Sections 5.3, 5.4, 5.7, 5.11, 5.12 and Fig. 4 of the paper).

Reads results/revision/ (stats CSVs from make_v11_stats.py and make_revision_stats.py, plus the
ablation and anchor-sweep result files) and writes LaTeX tables and the frequent-anchor figure
to analysis/out_revision/ (or to $OUT).
Usage (from the repository root): python analysis/make_revision_tables.py
"""
import csv, collections, os, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from cluster_boot import boot_se  # noqa: E402

R = Path("results/revision"); STATS = R / "stats"
ABL = R / "ablation_v2"; SWP = R / "anchor_sweep_v3"; V8C = R / "awedist_25snippets_v8c"
TAB = FIG = Path(os.environ.get("OUT", "analysis/out_revision")); TAB.mkdir(parents=True, exist_ok=True)
def f2(x): return ("$-$" if x < 0 else "$+$") + f"{abs(x):.2f}"
def rng_fmt(v):
    g = lambda x: ("$-$" if x < 0 else "$+$") + (f"{abs(x):.0f}" if abs(x) >= 100 else f"{abs(x):.2f}")
    return f"{g(min(v))} to {g(max(v))}"
def rows(name): return list(csv.DictReader(open(STATS / name)))
def tally(rs):
    b = sum(1 for r in rs if float(r["hi"]) < 0 and r["holm"] == "True")
    w = sum(1 for r in rs if float(r["lo"]) > 0 and r["holm"] == "True")
    return b, len(rs) - b - w, w

# ---------------------------------------------------------------- frequent anchors / global map
SPEC = [("ridge[sel] vs OMP-64 freq5000 centred (cal)", "Ridge $-$ OMP-64, centred, same anchors, cal.\\ length"),
        ("ridge[sel] vs OMP-64 freq5000 as shipped", "Ridge $-$ OMP-64 as shipped, same anchors"),
        ("ridge[sel] vs LS[freq5000]", "Ridge $-$ least squares, same anchors"),
        ("ridge: freq5000 vs all", "Ridge: frequent $-$ all anchors"),
        ("LS: freq5000 vs all", "Least squares: frequent $-$ all anchors"),
        ("OMP-64 centred: freq5000 vs all", "OMP-64 centred: frequent $-$ all anchors")]
main, scale = rows("v11_main_stats.csv"), rows("v11_scale_stats.csv")
L = ["\\begin{tabular}{lcrrcrr}", "\\toprule",
     " & \\multicolumn{3}{c}{12 runs (4 small pairs $\\times$ 3 seeds)} & \\multicolumn{3}{c}{3 larger receivers (seed 0)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "Comparison (negative = first better) & B / C / W & Median & Range & B / C / W & Median & Range \\\\", "\\midrule"]
for key, lab in SPEC:
    cells = []
    for src in (main, scale):
        rs = [r for r in src if r["test"] == key]
        v = [float(r["mean"]) for r in rs]; b, c, w = tally(rs)
        cells += [f"{b} / {c} / {w}", f2(np.median(v)), rng_fmt(v)]
    L.append(lab + " & " + " & ".join(cells) + " \\\\")
    if key == "ridge[sel] vs LS[freq5000]": L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
(TAB / "freq_global.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- length calibration (zero-shot)
SPEC = [("ridge[sel] cal vs c=1", "Ridge: calibrated $-$ fixed length $c=1$"),
        ("LS[freq5000] cal vs c=1", "Least squares: calibrated $-$ fixed length $c=1$"),
        ("ridge[sel] cal vs median-norm", "Ridge: calibrated $-$ median-norm rescaling"),
        ("LS[freq5000] cal vs median-norm", "Least squares: calibrated $-$ median-norm"),
        ("OMP-64 centred cal vs median-norm", "OMP-64 centred: calibrated $-$ median-norm"),
        ("ridge[sel] vs mean-subword median-norm", "Ridge $-$ subword mean with median-norm")]
L = ["\\begin{tabular}{lcrrcrr}", "\\toprule",
     " & \\multicolumn{3}{c}{12 runs (small pairs)} & \\multicolumn{3}{c}{3 larger receivers} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "Zero-shot comparison (negative = first better) & B / C / W & Median & Range & B / C / W & Median & Range \\\\", "\\midrule"]
for key, lab in SPEC:
    cells = []
    for src in (main, scale):
        rs = [r for r in src if r["test"] == key]
        v = [float(r["mean"]) for r in rs]; b, c, w = tally(rs)
        cells += [f"{b} / {c} / {w}", f2(np.median(v)), rng_fmt(v)]
    L.append(lab + " & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(TAB / "length_cal.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- learned initialisers (AweDist at 25 snippets)
aw = rows("v8c_awedist25_stats.csv")
def cell(pair, B, st):
    r = [r for r in aw if r["pair"] == pair and r["B"] == B and r["st"] == str(st)][0]
    s = f"{f2(float(r['mean']))}\\,[{f2(float(r['lo']))},{f2(float(r['hi']))}]"
    if float(r["lo"]) > 0 or float(r["hi"]) < 0: s += "$^{*}$"
    if r["holm"] == "True": s += "$^{\\dagger}$"
    return s
L = ["\\begin{tabular}{llccc}", "\\toprule",
     "Pair & Steps & AweDist (official, 25 snippets) & TokDist (ours) & OMP-64 centred, cal. \\\\", "\\midrule"]
for pair in sorted({r["pair"] for r in aw}, key=lambda p: (not p.startswith("GPT"), p)):
    for st in (0, 200):
        L.append(("" if st else pair.replace("<-", "$\\leftarrow$")) + f" & {st} & " + " & ".join(cell(pair, B, st) for B in ("AweDist (official)", "TokDist (ours)", "OMP-64 centred (cal)")) + " \\\\")
    L.append("\\addlinespace")
L[-1] = "\\bottomrule"; L.append("\\end{tabular}")
(TAB / "learned_baselines.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- ablation (training-only words, 4 pairs x 2 seeds)
ab = rows("ablation_v2_stats.csv")
fs = sorted(ABL.glob("*.pkl")); rng = np.random.default_rng(0)
lam_rows = collections.defaultdict(list)
for f in fs:  # lambda against ridge on all anchors (same anchors), so that the anchor choice does not enter
    d = pickle.load(open(f, "rb")); c = d["curves"]; ids = np.array(d["abs_ids"]["t_eval"])
    for lam in ("0.0001", "0.01", "1"):
        for st in (0, 200):
            diff = c[f"C lambda={lam} (cal)"][st]["t_eval"]["tot"] - c["ridge[all]"][st]["t_eval"]["tot"]
            m, se = boot_se(diff, ids, rng=rng); lo, hi = boot_se.last
            lam_rows[(lam, st)].append(dict(mean=m, lo=lo, hi=hi))
SPEC = [("no centring (cal)", "No centring, calibrated length"), ("no centring (raw)", "No centring, raw output"),
        ("own magnitude", "Own magnitude (no unit direction, no calibration)"), ("fixed scale 1.0", "Unit direction, fixed length 1.0"),
        ("all anchors (ridge[all])", "All anchors instead of the selected subset$^{a}$"), ("mean-subword", "Mean of word pieces (calibrated)")]
L = ["\\begin{tabular}{lcrrcrr}", "\\toprule",
     " & \\multicolumn{3}{c}{Zero-shot} & \\multicolumn{3}{c}{After 200 steps} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "Change (positive = ablation worse) & W / C / B & Median & Range & W / C / B & Median & Range \\\\", "\\midrule"]
for key, lab in SPEC:
    cells = []
    for st in ("0", "200"):
        rs = [r for r in ab if r["test"] == key and r["stage"] == st]
        v = [float(r["mean"]) for r in rs]
        w = sum(1 for r in rs if float(r["lo"]) > 0 and r["holm"] == "True"); b = sum(1 for r in rs if float(r["hi"]) < 0 and r["holm"] == "True")
        rng_s = f"{f2(min(v))} to {f2(max(v))}" if max(abs(x) for x in v) < 100 else f"{f2(min(v))} to $+${max(v):.0f}"
        cells += [f"{w} / {len(rs)-w-b} / {b}", f2(np.median(v)), rng_s]
    L.append(lab + " & " + " & ".join(cells) + " \\\\")
L.append("\\midrule")
for lam, lab in (("0.0001", "$\\lambda=10^{-4}$ vs.\\ chosen $\\lambda$, all anchors$^{b}$"), ("0.01", "$\\lambda=10^{-2}$ vs.\\ chosen $\\lambda$, all anchors$^{b}$"), ("1", "$\\lambda=1$ vs.\\ chosen $\\lambda$, all anchors$^{b}$")):
    cells = []
    for st in (0, 200):
        rs = lam_rows[(lam, st)]; v = [r["mean"] for r in rs]
        w = sum(1 for r in rs if r["lo"] > 0); b = sum(1 for r in rs if r["hi"] < 0)
        cells += [f"{w} / {len(rs)-w-b} / {b}", f2(np.median(v)), f"{f2(min(v))} to {f2(max(v))}"]
    L.append(lab + " & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(TAB / "ablation.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- scarce-anchor sweep (training-only words)
L = ["\\begin{tabular}{llrrr}", "\\toprule",
     "Pair & $n$ ($n/d$) & LS & Ridge & Ridge $-$ LS: mean [min, max over subsets] \\\\", "\\midrule"]
for f in sorted(SWP.glob("*.pkl")):
    d = pickle.load(open(f, "rb")); g = collections.defaultdict(list)
    for r in d["runs"]: g[r["n"]].append(r)
    first = True
    for n in sorted(g):
        rs = g[n]; dif = [(r["ridge"] - r["LS"]).mean() for r in rs]
        span = f"[{f2(min(dif))}, {f2(max(dif))}]" if len(rs) > 1 else "(one subset)"
        L.append((d["pair"].replace("<-", "$\\leftarrow$") if first else "") + f" & {n:,} ({n/d['d']:.1f}) & {np.mean([r['LS'].mean() for r in rs]):.1f} & {np.mean([r['ridge'].mean() for r in rs]):.1f} & {f2(np.mean(dif))} {span} \\\\")
        first = False
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L.append("\\end{tabular}")
(TAB / "anchor_sweep.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure: frequent anchors and the global map, per run
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
panels = [("ridge: freq5000 vs all", "Ridge", "#2a6fbb"), ("LS: freq5000 vs all", "Least squares", "#e08a1e"),
          ("OMP-64 centred: freq5000 vs all", "OMP-64 centred", "#3a9a4a"), ("ridge[sel] vs OMP-64 freq5000 centred (cal)", "Ridge $-$ OMP-64,\nsame anchors", "#7a4fa0")]
for ax, st in zip(axes, ("0", "200")):
    for i, (key, lab, col) in enumerate(panels):
        rs = [r for r in main if r["test"] == key and r["stage"] == st]
        for j, r in enumerate(rs):
            m, lo, hi = float(r["mean"]), float(r["lo"]), float(r["hi"]); y = i + (j - (len(rs) - 1) / 2) * 0.055
            filled = (lo > 0 or hi < 0)
            ax.plot([lo, hi], [y, y], color=col, lw=0.8, alpha=0.8)
            ax.plot(m, y, "o", ms=3.2, color=col, mfc=col if filled else "white")
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_title("Zero-shot" if st == "0" else "After 200 steps training the new rows")
    ax.set_xlabel("NLL difference per sentence (nats)")
    ax.grid(axis="x", color="0.9")
axes[0].set_yticks(range(len(panels))); axes[0].set_yticklabels([("Frequent $-$ all:\n" + p[1]) if i < 3 else p[1] for i, p in enumerate(panels)])
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / "fig_frequent_anchors.pdf"); fig.savefig(FIG / "fig_frequent_anchors.png", dpi=200)
print("tables and figure written")

# ---------------------------------------------------------------- initialisation time, learned run (one T4)
if V8C.exists():
    L = ["\\begin{tabular}{lrrrl}", "\\toprule", "Pair & Ridge (both subsets) & AweDist (official, 25 snippets) & TokDist (ours) & Selected ridge \\\\", "\\midrule"]
    for f in sorted(V8C.glob("**/*.pkl"), key=lambda p: (not p.name.split("__")[1].startswith("GPT"), p.name)):
        d = pickle.load(open(f, "rb")); cs, ms = d["calib_s"], d["map_s"]
        rid = cs["ridge[all]"] + cs["ridge[freq5000]"] + ms.get("ridge[all]", 0) + ms.get("ridge[freq5000]", 0)
        L.append(f"{d['pair'].replace('<-', '$\\leftarrow$')} & {rid:.0f} & {cs['AweDist (official)']:.0f} & {cs['TokDist (ours)']:.0f} & {d['sel'].replace('ridge[', '').rstrip(']')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "learned_time.tex").write_text("\n".join(L) + "\n")
    print("learned_time written")
