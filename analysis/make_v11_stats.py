"""Statistics for the v11 revision runs (R5, R7, R8, R9, R10).
Usage:  python make_v11_stats.py ../results/v11_revision   [out.csv]
Every test is a paired comparison on the evaluation sentences that contain the new words (t_eval), difference = A - B in total NLL (nats; NEGATIVE = A better).
Resampling unit: the abstract (cluster bootstrap, 5000 draws, 95% percentile interval). Multiplicity: Holm step-down within each declared family, with
normal-approximation p-values. Seeds of one pair are treated as separate runs, as in the earlier analysis (they are repeated measures of one setup; state this in the paper).
Families are declared here, before any result is read.
"""
import sys, glob, pickle, csv, collections, numpy as np
from scipy.stats import norm
from cluster_boot import boot_se

SEL = "ridge[sel]"                       # resolved per run from res["sel"]
FAMILIES = {   # name -> list of (label, A, B, stages)
 "F1 ridge vs least squares on the SAME frequent anchors (R10)": [("ridge[sel] vs LS[freq5000]", SEL, "LS[freq5000] (cal)", (0, 200))],
 "F2 ridge vs official OMP-64 on the SAME frequent anchors (R7)": [("ridge[sel] vs OMP-64 freq5000 centred (cal)", SEL, "OMP-64 freq5000 centred (cal)", (0, 200)),
                                                                   ("ridge[sel] vs OMP-64 freq5000 as shipped", SEL, "OMP-64 freq5000 as shipped (raw)", (0, 200))],
 "F3 effect of the frequent-anchor choice, per method (R7, R10): frequent minus all": [("ridge: freq5000 vs all", "ridge[freq5000]", "ridge[all]", (0, 200)),
                                                                                       ("LS: freq5000 vs all", "LS[freq5000] (cal)", "LS (cal)", (0, 200)),
                                                                                       ("OMP-64 centred: freq5000 vs all", "OMP-64 freq5000 centred (cal)", "OMP-64 centred (cal)", (0, 200))],
 "F4 calibrated length vs uncalibrated length c=1, zero-shot (R9)": [("ridge[sel] cal vs c=1", SEL, SEL + " c=1 (raw)", (0,)),
                                                                     ("LS[freq5000] cal vs c=1", "LS[freq5000] (cal)", "LS[freq5000] (cal) c=1 (raw)", (0,))],
 "F5 our calibration vs median-norm rescaling, zero-shot (R8)": [("ridge[sel] cal vs median-norm", SEL, SEL + " median-norm (raw)", (0,)),
                                                                 ("LS[freq5000] cal vs median-norm", "LS[freq5000] (cal)", "LS[freq5000] (cal) median-norm (raw)", (0,)),
                                                                 ("OMP-64 centred cal vs median-norm", "OMP-64 centred (cal)", "OMP-64 centred median-norm (raw)", (0,)),
                                                                 ("ridge[sel] vs mean-subword median-norm", SEL, "mean-subword median-norm (raw)", (0,))],
}
def holm(p, alpha=0.05):
    idx = np.argsort(p); rej = np.zeros(len(p), bool)
    for r, i in enumerate(idx):
        if p[i] <= alpha / (len(p) - r): rej[i] = True
        else: break
    return rej

def main(folder, out_csv="v11_stats.csv"):
    runs = sorted(glob.glob(f"{folder}/*.pkl")); rng = np.random.default_rng(0); allrows = []
    print(f"{len(runs)} result files in {folder}")
    for fam, specs in FAMILIES.items():
        rows = []
        for f in runs:
            d = pickle.load(open(f, "rb")); c = d["curves"]; ids = np.array(d["abs_ids"]["t_eval"]); sel = d["sel"]
            for lab, A, B, stages in specs:
                A = sel if A == SEL else A; B = (sel + B[len(SEL):]) if B.startswith(SEL) else B
                for st in stages:
                    if A not in c or B not in c or st not in c[A] or st not in c[B]: continue
                    diff = c[A][st]["t_eval"]["tot"] - c[B][st]["t_eval"]["tot"]; m, se = boot_se(diff, ids, rng=rng); lo, hi = boot_se.last
                    rows.append(dict(family=fam, test=lab, run=f.split("/")[-1], stage=st, mean=m, lo=lo, hi=hi, p=2 * (1 - norm.cdf(abs(m / se))) if se > 0 else 1.0))
        if not rows: print(f"\n{fam}: no comparable runs found"); continue
        rej = holm(np.array([r["p"] for r in rows]))
        for r, h in zip(rows, rej): r["holm"] = bool(h)
        print(f"\n{fam}: {len(rows)} tests (Holm family)")
        by = collections.defaultdict(list)
        for r in rows: by[r["test"]].append(r)
        for lab, rs in by.items():
            better = [r for r in rs if r["hi"] < 0]; worse = [r for r in rs if r["lo"] > 0]; ns = len(rs) - len(better) - len(worse)
            print(f"  {lab:<46} A better {len(better)} (Holm {sum(r['holm'] for r in better)}) | comparable {ns} | A worse {len(worse)} (Holm {sum(r['holm'] for r in worse)}) | median diff {np.median([r['mean'] for r in rs]):+.2f} nats")
        allrows += rows
    if allrows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
        print(f"\nsaved {out_csv}")
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v11_stats.csv")
