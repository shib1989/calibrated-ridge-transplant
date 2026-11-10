"""Figure: what separates ridge from least squares (PubMed, seed 0, four small pairs; abstract-level bootstrap 95 % intervals).
Three differences against plain LS on ALL anchors (calibrated length for all): anchor subset only (LS on the 5000 most frequent anchors),
penalty only (ridge on all anchors), both (ridge on the 5000 most frequent anchors). Output: OUT/fig_ls_decomp.pdf/png. Colours: validated categorical slots 1-3 plus marker shapes."""
import os, json, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
OUT = os.environ.get("OUT", os.path.join(HERE, "out_fig")); os.makedirs(OUT, exist_ok=True)
PAIRS = ["GPT-2_from_Pythia-160m", "GPT-2_from_Qwen2.5-0.5B", "OPT-125m_from_Pythia-160m", "OPT-125m_from_Qwen2.5-0.5B"]
def boot(d, ids, B=20000):
    rng = np.random.default_rng(0); u, inv = np.unique(ids, return_inverse=True); s = np.bincount(inv, weights=d); c = np.bincount(inv).astype(float); k = len(u); b = np.empty(B)
    for i in range(0, B, 2000):
        dr = rng.integers(0, k, (2000, k)); b[i:i + 2000] = s[dr].sum(1) / c[dr].sum(1)
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
SER = [("LS[freq5000]", "anchor subset only\n(LS on 5000 frequent anchors)", "#2a78d6", "o"),
       ("ridge[all]", "penalty only\n(ridge on all anchors)", "#eb6834", "s"),
       ("ridge[freq5000]", "both\n(ridge on 5000 frequent anchors)", "#1baf7a", "D")]
res = {}
for p in PAIRS:
    c = pickle.load(open(f"{R}/control_ls_freq5000/pubmed__{p}__seed0.pkl", "rb"))["curves"]
    ids = np.array(json.load(open(f"{R}/abstract_ids/v6_trainonly__pubmed__{p}__seed0.pkl.json"))["abstract_id"])
    for st in (0, 200):
        for k, *_ in SER: res[(p, st, k)] = boot(c[k][st]["t_eval"]["tot"] - c["LS (cal)"][st]["t_eval"]["tot"], ids)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True, facecolor="#fcfcfb")
ys = np.arange(len(PAIRS))[::-1]; off = {SER[0][0]: 0.22, SER[1][0]: 0.0, SER[2][0]: -0.22}
for ax, st, title in zip(axs, (0, 200), ("zero-shot", "after 200 steps")):
    ax.set_facecolor("#fcfcfb"); ax.axvline(0, color="#52514e", lw=1)
    for k, lab, col, mk in SER:
        for y, p in zip(ys, PAIRS):
            m, lo, hi = res[(p, st, k)]
            ax.plot([lo, hi], [y + off[k]] * 2, color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(m, y + off[k], marker=mk, ms=6.5, mfc=col, mec="#0b0b0b", mew=0.6, ls="none", zorder=3, label=lab if (p == PAIRS[0] and st == 0) else None)
    ax.set_title(title, fontsize=9.5, loc="left"); ax.set_xlabel("NLL difference to LS (nats)\n$\\leftarrow$ better than LS on all anchors", fontsize=8.5)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(ys); axs[0].set_yticklabels([p.replace("_from_", "\n$\\leftarrow$ ") for p in PAIRS], fontsize=8)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(OUT + "/fig_ls_decomp.pdf", bbox_inches="tight", facecolor=fig.get_facecolor()); fig.savefig(OUT + "/fig_ls_decomp.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
json.dump({f"{p}|{st}|{k}": list(map(float, v)) for (p, st, k), v in res.items()}, open(OUT + "/fig_ls_decomp_values.json", "w"), indent=1)
for (p, st, k), v in res.items():
    if st == 0 and p == PAIRS[0] or st == 200 and p == PAIRS[0]: print(p, st, k, [round(x, 2) for x in v])
