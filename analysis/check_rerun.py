"""Compare a rerun result file with the saved one, sentence by sentence.
Usage: python check_rerun.py saved.pkl rerun.pkl
Prints, for every method and step, the largest absolute difference of the per-sentence NLL and the difference of the means.
GPU kernels, library versions and the unseeded parts of the data pipeline can cause small differences; a rerun 'matches' if the mean differences are far
below the differences between methods (we suggest below 0.05 nats per sentence) and the chosen ridge variant and calibrated scales are identical."""
import pickle, sys, numpy as np
a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
print("same words:", a["terms"] == b["terms"], "| same ridge variant:", a["sel"] == b["sel"], "| same scales:", a["scales"] == b["scales"])
worst = 0.0
for m in a["curves"]:
    for st in a["curves"][m]:
        x, y = a["curves"][m][st]["t_eval"]["tot"], b["curves"][m][st]["t_eval"]["tot"]
        d = float(abs(x.mean() - y.mean())); worst = max(worst, d)
        print(f"{m:<26} step {st:<3} max |sentence diff| {np.abs(x - y).max():8.4f}   |mean diff| {d:8.4f}")
print("largest |mean diff|:", round(worst, 4), "->", "MATCH" if worst < 0.05 else "CHECK")
