"""Real vocabulary expansion (final): 16 runs = PubMed seeds 0-2 + arXiv seed 0, x 4 pairs. Official OMP-64 (mergekit batch_omp), centred and as shipped.
Domain is read from the new words (files named 'arxiv' that contain medical words are PubMed seed 0 runs)."""
import pickle, glob, csv, numpy as np
A = "../results/runs16/"
def boot(d, n=5000, seed=0):
    r = np.random.default_rng(seed); m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
runs = {}
for f in sorted(glob.glob(A + "*.pkl")):
    try: r = pickle.load(open(f, "rb"))
    except Exception: continue
    if not (isinstance(r, dict) and "curves" in r and "OMP-64 centred (raw)" in r["curves"] and r.get("train_steps") == 200): continue
    dom = "PubMed" if " apoptosis" in r["terms"][:6] else "arXiv"
    runs[(r["pair"], dom, r["seed"])] = r          # duplicate files are identical
rows = []
for (pair, dom, seed), r in sorted(runs.items(), key=lambda kv: (kv[0][1] != "PubMed", kv[0][2], kv[0][0])):
    cv, sel = r["curves"], r["sel"]; o = cv["original"][0]["t_eval"]["tot"]
    for st in (0, 200):
        g = lambda m: cv[m][st]["t_eval"]["tot"]
        rows.append(dict(pair=pair, domain=dom, seed=seed, K=r["K"], steps=st, sel=sel, ridge_ref=boot(g(sel) - o),
            **{k: boot(g(sel) - g(m)) for k, m in [("r_omp32", "OMP-32"), ("r_omp64c", "OMP-64 centred (raw)"), ("r_omp64cal", "OMP-64 centred (cal)"), ("r_omp64s", "OMP-64 as shipped (raw)"), ("r_msub", "mean-subword (cal)")]}))
pickle.dump(rows, open("/tmp/rv/rows3.pkl", "wb"))
fm = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]" + ("*" if t[1] > 0 or t[2] < 0 else "")
with open("tables/table7_real_vocab_expansion_final.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["pair", "text", "seed", "words", "steps", "ridge_variant", "ridge_vs_unchanged_receiver", "ridge_minus_OMP32_own", "ridge_minus_OMP64_official_centred", "ridge_minus_OMP64_official_centred_cal", "ridge_minus_OMP64_official_as_shipped", "ridge_minus_meansubword"])
    for x in rows: w.writerow([x["pair"], x["domain"], x["seed"], x["K"], x["steps"], x["sel"]] + [fm(x[k]) for k in ("ridge_ref", "r_omp32", "r_omp64c", "r_omp64cal", "r_omp64s", "r_msub")])
print("runs", len(runs), "comparisons per competitor", len(rows))
for k in ("r_omp32", "r_omp64c", "r_omp64cal", "r_omp64s", "r_msub"):
    print(k, "ridge better", sum(x[k][0] < 0 for x in rows), "| CI<0", sum(x[k][2] < 0 for x in rows), "| ridge worse CI>0", sum(x[k][1] > 0 for x in rows), "| range", min(x[k][0] for x in rows), max(x[k][0] for x in rows))
for s in (0, 200):
    xs = [x for x in rows if x["steps"] == s]
    print("steps", s, "ridge vs ref: better(CI<0)", sum(x["ridge_ref"][2] < 0 for x in xs), "worse(CI>0)", sum(x["ridge_ref"][1] > 0 for x in xs), "level", sum(x["ridge_ref"][1] <= 0 <= x["ridge_ref"][2] for x in xs), "of", len(xs))
    for k in ("r_omp64c",): print("  omp64c range", min(x[k][0] for x in xs), max(x[k][0] for x in xs))
for x in rows: print(x["pair"][:21], x["domain"], x["seed"], x["steps"], "| ref", fm(x["ridge_ref"]), "| 64c", fm(x["r_omp64c"]), "| 64s", fm(x["r_omp64s"]), "| 32", fm(x["r_omp32"]))
