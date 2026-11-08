# Reproducibility guide

This guide lets a reader rerun every experiment and rebuild every table. Items that were not recorded at the time of the runs are marked **not recorded**.

## 1. Environment
- Notebooks were run on Kaggle (one GPU, Internet on). `notebooks/capture_environment.ipynb` records Python, PyTorch, Transformers, scikit-learn, mergekit, WECHSEL, FOCUS (deepfocus) versions, CUDA, driver and GPU, the Hugging Face checkpoint revisions in the local cache and the current hub heads, dataset revisions, and the SHA-256 of the fastText model used by WECHSEL. It writes `environment.json` and `environment_pip_freeze.txt`. The four v5 notebooks call the same recorder in their last cell.
- The experiment notebooks install packages **without version pins**. Versions in `environment.json` are those installed when the recorder ran. **Not recorded** for the runs already reported: the versions at the time. See section 6 for a rerun check.
- Official components: OMP-64 is mergekit's `batch_omp`, copied verbatim from mergekit 0.1.5 (`mergekit/tokensurgeon/omp.py`) into the notebooks; FOCUS is the `deepfocus` package; WECHSEL is `wechsel` 0.0.4 (same-language mode, `cc.en.300.bin` fastText vectors, `align_strategy=None`, 10 neighbours, temperature 0.1).
- All models run in float32. The ridge, least-squares, Procrustes and SALT-ours maps use float64 (CPU NumPy, SALT-ours in torch float64); official OMP-64 runs in float32 on the GPU.

### Recorded so far
`results/environment_gpu_session.json` (captured 2026-11-08, notebook session with two Tesla T4 GPUs of 15 GB, driver 580.178.04, CUDA 13.0, cuDNN 9.27; Python 3.13.16, PyTorch 2.11.0, Transformers 5.20.2, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1, datasets 4.8.5, mergekit 0.1.5, wechsel 0.0.4, deepfocus 1.0.1). It was captured **after** the reported runs, so it is the reference environment for reruns, not proof of the versions used then. The notebooks use the first GPU only.
Still **not captured**: the checkpoint revisions in the local cache (the cache was empty in that session; the hub heads at capture are listed, but are not necessarily the revisions the runs loaded) and the SHA-256 of `cc.en.300.bin`. The last cell of each v5 notebook records both automatically at the end of a run.

## 2. Checkpoints and data
| Role | Identifier (Hugging Face) |
|---|---|
| Receivers | `openai-community/gpt2`, `openai-community/gpt2-medium`, `openai-community/gpt2-large`, `facebook/opt-125m`, `EleutherAI/pythia-1b`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` |
| Donors | `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m`, `Qwen/Qwen2.5-0.5B`, `openai-community/gpt2` |
| Domain text | `qiaojin/PubMedQA`, configuration `pqa_artificial`, split `train`, streamed; the first 4000 abstracts (contexts joined) |
| General text | `Salesforce/wikitext`, `wikitext-2-raw-v1`, `train`, first 30000 lines, headings removed |
| Downstream | PubMed 20k RCT (`Franck-Dernoncourt/pubmed-rct`, `PubMed_20k_RCT`) |
Checkpoints and datasets are loaded from the `main` branch without a pinned revision; the revisions are in `environment.json`. **Not recorded** for the earlier runs.

## 3. Preprocessing (real-vocabulary protocol)
- Abstracts are split 70 % / 30 % by position into training and evaluation text. Text is whitespace-normalised and split into sentences at `.`, `!` or `?` followed by whitespace; sentences of 8 to 60 words are kept.
- Segmentation for word matching: regex ` ?[^\W\d_]+| ?\d+| ?_+| ?[^\s\w]+|\s+`.
- Eligible new words: ` ` + at least 4 letters, at least 5 occurrences in training and 3 in evaluation text (Pythia-1B: 3 and 2, at least 10 words), at least 2 receiver pieces, exactly 1 donor token; ranked by training plus evaluation count, ties alphabetical; top 300 kept.
- Anchors: tokens present in both vocabularies with the same string, leading "Ġ", alphabetic, at least 3 letters.
- Calibration sentences: 300 containing new words and 150 without (half domain text, half WikiText first half). Evaluation: up to 600 sentences with new words, 200 domain controls, 200 WikiText controls (second half). **If fewer sentences exist, all are used** (Pythia-1B: 129 evaluation sentences, so the evaluation set does not change with the seed).

## 4. Hyperparameters
| Item | Value |
|---|---|
| Ridge λ grid (relative to mean diagonal of XᵀX) | 1e-4, 1e-3, 1e-2, 1e-1, 1 |
| λ selection | mean cosine on held-out anchors: 10 % of fitting anchors, at least 50, at most 20 %; then refit on all fitting anchors |
| Anchor subsets | all anchors; 5000 most frequent anchors; chosen by the training objective |
| Length grid c | 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, chosen by the receiver NLL on the calibration sentences (half new-word sentences, half controls) |
| OMP-64 (official) | k = 64, raw coordinates ("as shipped") and mean-centred inputs |
| OMP-32 (ours) | scikit-learn `orthogonal_mp`, k = 32, unit-normalised centred vectors |
| SALT-ours | 2000 nearest anchors (cosine), ridge with intercept, λ = 1e-2 relative, dual form, float64 |
| Training of new rows | 200 steps, AdamW, lr 1e-3, weight decay 0, batch 8 sentences containing new words from a pool of at most 2000, model in eval mode (no dropout), only new rows updated (old rows restored exactly); Qwen2.5-1.5B uses micro-batches of 2 with the loss divided by the total tokens of the batch |
| Untied receivers | pipeline run for input and output matrices; one shared c; both sets of new rows trained |
| Statistics | paired percentile bootstrap over sentences, 5000 resamples, NumPy `default_rng(0)`; Holm as in the manuscript |

## 5. Seeds
- `random.Random(seed)` samples the calibration, pool, evaluation and control sentences and the order of the training batches.
- `numpy.random.default_rng(seed)` draws the random-direction baseline (`seed + 1` for the output side of an untied receiver).
- The held-out split used to choose λ is **fixed** (`default_rng(0)`) and does not depend on the run seed.
- No global seed is set for Python, NumPy or PyTorch. The only sources of non-determinism are GPU kernels and library versions.
- Seeds resample sentences and batches. Words, the 70/30 text split and the anchors do not change with the seed.

## 6. Checking that a rerun matches
1. Run `notebooks/real_vocab_transplant_v5_main.ipynb` for one pair (edit `MAIN_PAIRS` to keep, for example, GPT-2 ← Pythia-160m; about 14 minutes on one GPU).
2. `python analysis/check_rerun.py results/v5_small_and_scale/pubmed__GPT-2_from_Pythia-160m__seed1.pkl <your new pkl>`.
3. The words, the selected ridge variant and the calibrated scales should be identical; the mean sentence NLL should agree to within about 0.05 nats. Report the outcome next to `environment.json`.

## 7. Which script produces which table or figure
| Output | Source |
|---|---|
| Table with 16 runs (Table 2), Fig. real vocabulary | `analysis/make_realvocab_v3.py` and `manuscript_source/make_assets.py` on `results/runs16/` |
| WECHSEL table | `manuscript_source/make_assets.py` (from the WECHSEL notebook results) |
| Seven-pair table, intervals, bits per byte | `analysis/make_v5_tables.py` on `results/v5_small_and_scale/` |
| Holm over 32 and 98 comparisons, equivalence | `analysis/stats16.py`, `analysis/stats_family.py` |
| Scarce-anchor figure and table | `notebooks/ls_vs_ridge_*` and `results/other/ls_vs_ridge_anchor_sweep.json` |
| Mapping time | `notebooks/speed_benchmark_ridge_vs_omp.ipynb` |
| float32 / float64 check | `notebooks/ridge_fp32_vs_fp64_check.ipynb` and `analysis/ridge_fp32_vs_fp64_check.py` |
| Downstream tables | `notebooks/exp5b_downstream_pubmed_rct.ipynb`, `notebooks/exp5c_frozen_probe.ipynb` |
(Assigned from file names and run history; verify each row before release.)

## 8. Run times seen (one GPU)
GPT-2 or OPT-125m receivers with 82 or 300 words: 12 to 14 minutes per pair; GPT-2 large: 65 minutes; Pythia-1B: 71 minutes; Qwen2.5-1.5B: not recorded in the result file.
