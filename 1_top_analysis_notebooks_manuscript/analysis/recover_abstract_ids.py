"""Step B: recover, for every evaluation sentence of a stored run, the abstract it came from and the new words it contains.
No model is loaded (tokenizer only). Needs: pip install transformers datasets; internet for PubMedQA and WikiText-2.
It rebuilds the sentence sampling exactly as in the notebooks (same abstracts, split, seed, sampling order), then CHECKS the rebuild
against the stored per-sentence token counts. If the check fails for a run, that run is NOT written (so no wrong IDs are used).
Usage (Colab/Kaggle):  python recover_abstract_ids.py  <pkl> [<pkl> ...]      -> writes abstract_ids/<pkl name>.json
"""
import sys, re, json, pickle, random, os
from collections import Counter
import numpy as np
from transformers import AutoTokenizer

N_ABSTRACTS, TRAIN_FRAC = 4000, 0.7
N_TRAIN_T, N_TRAIN_C, N_EVAL_T, N_EVAL_C, POOL_MAX = 300, 150, 600, 200, 2000
RECEIVER = {"GPT-2": "openai-community/gpt2", "GPT-2_large": "openai-community/gpt2-large", "OPT-125m": "facebook/opt-125m",
            "Pythia-1B": "EleutherAI/pythia-1b", "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B"}
DOMAIN_TXT = os.environ.get("DOMAIN_TXT")   # only for arXiv runs: the same one-abstract-per-line file used in the run

def split_sents(text):
    text = " ".join(text.split())
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if 8 <= len(s.split()) <= 60]
SEG = re.compile(r" ?[^\W\d_]+| ?\d+| ?_+| ?[^\s\w]+|\s+")
def segments(s): return SEG.findall(s)

def load_domain():
    if DOMAIN_TXT: return [l.strip() for l in open(DOMAIN_TXT, encoding="utf-8") if l.strip()][:N_ABSTRACTS]
    from datasets import load_dataset
    for cfg, stream in (("pqa_artificial", True), ("pqa_labeled", False)):
        try:
            ds, out = load_dataset("qiaojin/PubMedQA", cfg, split="train", streaming=stream), []
            for ex in ds:
                out.append(" ".join(ex["context"]["contexts"]))
                if len(out) >= N_ABSTRACTS: break
            if out: print(f"domain text: PubMedQA/{cfg}, {len(out)} abstracts"); return out
        except Exception as e: print("failed", cfg, e)
    raise RuntimeError("cannot load PubMedQA")

def load_wiki():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = " ".join(x for x in ds["text"][:30000] if x.strip() and not x.startswith(" ="))
    return split_sents(text)

def n_tokens(tok, segs, term_id, mode):
    n = 0
    for g in segs:
        n += 1 if (mode == "new" and g in term_id) else len(tok.encode(g, add_special_tokens=False))
    return n

def receiver_key(pair):               # e.g. "GPT-2 <- Pythia-160m", "GPT-2 large <- Qwen2.5-0.5B"
    left = pair.split("<-")[0].strip().replace(" ", "_")
    return left

def recover(pkl):
    r = pickle.load(open(pkl, "rb")); seed = r["seed"]; terms = r["terms"]
    tok = AutoTokenizer.from_pretrained(RECEIVER[receiver_key(r["pair"])])
    abstracts = load_domain(); n_tr = int(len(abstracts) * TRAIN_FRAC)
    tr = [(s, i) for i, a in enumerate(abstracts[:n_tr]) for s in split_sents(a)]
    ev = [(s, n_tr + i) for i, a in enumerate(abstracts[n_tr:]) for s in split_sents(a)]
    term_id = {g: j for j, g in enumerate(terms)}
    has = lambda s: any(g in term_id for g in segments(s))
    tr_t = [x for x in tr if has(x[0])]; tr_c = [x for x in tr if not has(x[0])]
    ev_t = [x for x in ev if has(x[0])]; ev_c = [x for x in ev if not has(x[0])]
    wiki = [s for s in load_wiki() if not has(s)]
    rng = random.Random(seed); smp = lambda L, n: rng.sample(L, min(n, len(L)))
    smp(tr_c, N_TRAIN_C // 2 if wiki else N_TRAIN_C)                       # calibration controls (drawn first in the notebook)
    if wiki: smp(wiki[:len(wiki) // 2], N_TRAIN_C // 2)
    smp(tr_t, N_TRAIN_T)                                                   # calibration sentences with new words
    smp(tr_t, POOL_MAX)                                                    # training pool
    t_eval = smp(ev_t, N_EVAL_T)                                           # evaluation sentences with new words
    segs = [segments(s) for s, _ in t_eval]
    o = np.array([n_tokens(tok, g, term_id, "orig") for g in segs]); n = np.array([n_tokens(tok, g, term_id, "new") for g in segs])
    st_o = np.asarray(r["curves"]["original"][0]["t_eval"]["ntok"]); st_n = np.asarray(r["curves"]["ridge[all]"][0]["t_eval"]["ntok"])
    ok = len(o) == len(st_o) and bool((o == st_o).all()) and bool((n == st_n).all())
    print(f"{os.path.basename(pkl)}: {len(t_eval)} sentences, token counts match stored: {ok}")
    if not ok: return None
    return dict(pair=r["pair"], seed=seed, abstract_id=[a for _, a in t_eval], sentence=[s for s, _ in t_eval],
                words=[sorted({g for g in g_ if g in term_id}) for g_ in segs])

if __name__ == "__main__":
    os.makedirs("abstract_ids", exist_ok=True)
    for p in sys.argv[1:]:
        out = recover(p)
        if out: json.dump(out, open(f"abstract_ids/{os.path.basename(os.path.dirname(os.path.abspath(p)))}__{os.path.basename(p)}.json", "w"))
