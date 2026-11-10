"""Abstract-level (cluster) bootstrap helpers. Sentences are resampled by abstract; the statistic is the mean over resampled sentences."""
import json, numpy as np
def ids_for(folder, pkl_name):
    return np.array(json.load(open(f"../results/abstract_ids/{folder}__{pkl_name}.json"))["abstract_id"])
def boot_se(d, ids=None, B=5000, rng=None):
    """d: per-sentence differences. ids None -> sentence bootstrap; else cluster bootstrap by abstract id."""
    rng = rng or np.random.default_rng(0)
    if ids is None:
        n = len(d); b = d[rng.integers(0, n, (B, n))].mean(1)
    else:
        u, inv = np.unique(ids, return_inverse=True)
        s = np.bincount(inv, weights=d); c = np.bincount(inv).astype(float)
        k = len(u); draw = rng.integers(0, k, (B, k)); b = s[draw].sum(1) / c[draw].sum(1)
    boot_se.last = (np.percentile(b, 2.5), np.percentile(b, 97.5))   # percentile interval of the last call
    return d.mean(), b.std()
