"""Statistics for the AweDist (25 snippets) runs and the training-only ablation.

Writes results/revision/stats/v8c_awedist25_stats.csv and ablation_v2_stats.csv.
Abstract-level paired bootstrap (cluster_boot.boot_se), two-sided normal p-values, Holm within each family.
Usage (from the repository root): python analysis/make_revision_stats.py
"""
import csv, glob, pickle, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))
from cluster_boot import boot_se  # noqa: E402

R = Path("results/revision"); OUT = R / "stats"; OUT.mkdir(exist_ok=True)

def holm(rows):
    p = np.array([r["p"] for r in rows]); idx = np.argsort(p); rej = np.zeros(len(p), bool)
    for k, i in enumerate(idx):
        if p[i] <= 0.05 / (len(p) - k): rej[i] = True
        else: break
    for r, h in zip(rows, rej): r["holm"] = bool(h)

def test(diff, ids, rng):
    m, se = boot_se(diff, ids, rng=rng); lo, hi = boot_se.last
    return m, lo, hi, (2 * (1 - norm.cdf(abs(m / se))) if se > 0 else 1.0)

# AweDist at its default 25 snippets: selected ridge minus each baseline (one Holm family over the three learned/OMP baselines)
rng = np.random.default_rng(0); rows = []
for f in sorted(glob.glob(str(R / "awedist_25snippets_v8c" / "*.pkl"))):
    d = pickle.load(open(f, "rb")); c = d["curves"]; ids = np.array(d["abs_ids"]["t_eval"]); sel = d["sel"]
    for B in ["AweDist (official)", "TokDist (ours)", "OMP-64 centred (cal)", "original"]:
        for st in (0, 200):
            if st not in c[B]: continue
            m, lo, hi, p = test(c[sel][st]["t_eval"]["tot"] - c[B][st]["t_eval"]["tot"], ids, rng)
            rows.append(dict(pair=d["pair"], B=B, st=st, mean=m, lo=lo, hi=hi, p=p))
fam = [r for r in rows if r["B"] != "original"]; holm(fam)
with open(OUT / "v8c_awedist25_stats.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pair", "B", "st", "mean", "lo", "hi", "p", "holm"]); w.writeheader()
    for r in rows: w.writerow({**r, "holm": r.get("holm", "")})

# Ablation (training-only words, 4 pairs x seeds 1 and 2): each ablation minus the selected ridge variant, one Holm family
rng = np.random.default_rng(0); rows = []
T = [("no centring (cal)", "A no-centring (cal)"), ("no centring (raw)", "A no-centring (raw output)"), ("own magnitude", "B own magnitude (no unit, no cal)"),
     ("fixed scale 1.0", "B fixed scale 1.0 (no cal)"), ("lambda=1e-4", "C lambda=0.0001 (cal)"), ("lambda=1", "C lambda=1 (cal)"),
     ("mean-subword", "mean-subword (cal)"), ("all anchors (ridge[all])", "ridge[all]")]
for f in sorted(glob.glob(str(R / "ablation_v2" / "*.pkl"))):
    d = pickle.load(open(f, "rb")); c = d["curves"]; ids = np.array(d["abs_ids"]["t_eval"]); ref = d["sel"]
    for lab, B in T:
        if B == ref: continue
        for st in (0, 200):
            m, lo, hi, p = test(c[B][st]["t_eval"]["tot"] - c[ref][st]["t_eval"]["tot"], ids, rng)
            rows.append(dict(test=lab, stage=st, pair=d["pair"], seed=d["seed"], ref=ref, mean=m, lo=lo, hi=hi, p=p))
holm(rows)
with open(OUT / "ablation_v2_stats.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", OUT / "v8c_awedist25_stats.csv", "and", OUT / "ablation_v2_stats.csv")
