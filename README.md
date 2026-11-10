# Calibrated ridge maps for token-embedding transplantation: code and results

Code, notebooks and result files for the manuscript "Calibrated Ridge Maps for Token-Embedding Transplantation Across Language Models".
Author: Shibdas Bhattacharya (23dp0094@iitism.ac.in), IIT (ISM) Dhanbad. Supervisor: Prof. Haider Banka.
Licence: MIT (see `LICENSE`). Citation: see `CITATION.cff`.

## What the method is
A new token is given an embedding by a linear map fitted on anchor tokens shared by the two vocabularies: ridge regression on mean-centred anchors (direction only), plus one length `c` chosen from a fixed grid by the receiver's loss on calibration sentences. Two anchor sets are tried (all anchors, the 5000 most frequent) and the training objective picks one. The least-squares baseline in the paper shares the centring, direction fitting and calibrated length, so ridge differs only by the penalty and, for the frequency-selected variant, the anchor subset.

## Layout
| Folder | Content |
|---|---|
| `notebooks/` | one notebook per experiment (Kaggle, one or two T4 GPUs; run top to bottom; each writes `.pkl`/`.json` results) |
| `results/` | the result files behind every table (see below) |
| `analysis/` | scripts that turn result files into tables, statistics and figures; defaults are repo-relative, outputs go to `analysis/out_*` or to `$OUT` |
| `manuscript_source/` | LaTeX source of the paper with tables, figures and the statistics JSON files (`latexmk -pdf main.tex`) |
| `REPRODUCE.md` | environment, checkpoints, preprocessing, hyper-parameters, seeds, statistics, table-to-script map |

## Results folders
| Folder | Content | Used for |
|---|---|---|
| `results/v6_trainonly/` | 17 PubMed runs, word rule uses training text only (12 small-pair runs, seeds 0 to 2; GPT-2 large, Pythia-1B, Qwen2.5-1.5B) | main real-vocabulary tables, scale tables |
| `results/abstract_ids/` | abstract of every evaluation sentence (v6 and arXiv) | abstract-level bootstrap |
| `results/v9_learned/` | AweDist (official) and TokDist (ours) with ridge, five tied pairs, seed 1 | learned-baseline table |
| `results/control_ls_freq5000/` | least squares on the same 5000 anchors, four small pairs, seed 0 | control table, ridge-vs-LS figure |
| `results/arxiv/` | arXiv second domain, four small pairs, seed 0, and the 3621 abstracts | arXiv table |
| `results/v10_7b/` | Qwen2.5-7B zero-shot, seed 1 | 7B table |
| `results/downstream_med_abstracts/` | medical-abstract classification after tokenizer swap | downstream table |
| `results/runs16/`, `results/v5_small_and_scale/`, `results/other/` | earlier runs (sentence-selected words), scarce-anchor sweep | earlier tables, appendix |
| `results/awedist_25snippets/` | error log of the AweDist default (25 snippets) attempt | explains why 5 snippets are used |
| `results/environment*.json`, `environment_pip_freeze_kaggle.txt` | recorded environments | reproducibility |

## Notebook map
| Notebook | Produces |
|---|---|
| `real_vocab_transplant_v6_trainonly` | main PubMed runs (training-only word rule) |
| `recover_abstract_ids_v6`, `recover_abstract_ids` | abstract ids of the evaluation sentences |
| `real_vocab_transplant_v5_main`, `_v5_scale`, `_v5_scale_pythia1b`, `_v5_scale_qwen15b` | earlier seven-pair and larger-receiver runs |
| `real_vocab_transplant_domain_v3_official_omp64`, `_v4_wechsel` | earlier 16-run set; official WECHSEL |
| `real_vocab_transplant_v7_tokdist`, `_v8_awedist` | precursors of v9 (TokDist; AweDist official, 5 snippets) |
| `real_vocab_transplant_v8b_awedist_defaults` | AweDist with its default 25 snippets: fails on our 2800 training abstracts (not enough snippets; log in `results/awedist_25snippets/`) |
| `real_vocab_transplant_v9_learned` | reported learned-baseline results |
| `control_ls_freq5000` | least-squares control on 5000 anchors |
| `real_vocab_transplant_v6_arxiv`, `recover_abstract_ids_arxiv` | arXiv domain |
| `real_vocab_transplant_v10_7b_zeroshot` | Qwen2.5-7B, fp16 over two T4 GPUs |
| `exp4_v2_fair_official_baselines` | zero-shot tokenizer swap with official baselines |
| `exp5b_downstream_pubmed_rct`, `exp5c_frozen_probe`, `downstream_med_abstracts` | downstream tasks |
| `ablation_ridge_v1`, `_gpt2` | ablations (earlier word rule) |
| `ls_vs_ridge_anchor_sweep`, `ls_vs_ridge_nll_sweep_v2` | scarce-anchor sweep |
| `speed_benchmark_ridge_vs_omp`, `ridge_fp32_vs_fp64_check`, `capture_environment` | timing, precision check, environment record |

## Rebuilding the tables
```
cd analysis
python make_v6_tables.py        # main, scale, learned-baseline and downstream tables + stats_v6.json
python make_ctl_arxiv_tables.py # control and arXiv tables
python make_7b_table.py         # 7B table
python make_fig_ls_decomp.py    # ridge-vs-least-squares figure
```
The output of these four scripts matches the tables in `manuscript_source/tables/` byte for byte. Intervals are 95 % paired percentile bootstraps over evaluation abstracts (all sentences of an abstract are resampled together); Holm correction uses a normal approximation to the bootstrap distribution. Earlier tables (sentence bootstrap, `stats16.py`, `stats_family.py`) are kept for the earlier runs.

## Known limits
- Receivers of 125M to 1.5B parameters, plus one 7B receiver tested zero-shot only; English text; PubMed (three seeds for small pairs) and arXiv (four pairs, one seed).
- At 1B parameters and above no initialisation, ridge included, beat leaving the words split.
- Seeds resample sentences and training batches; words, anchors and the text split are fixed.
- SALT-ours, LS, Procrustes, OMP-32 and TokDist are our implementations. FOCUS, WECHSEL, OMP-64 (mergekit 0.1.5 `batch_omp`, copied verbatim) and AweDist (authors' script) are official code.
- Checkpoints are loaded from the `main` branch of the Hugging Face hub without pinned revisions.
