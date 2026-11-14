# Revision notebooks (Neurocomputing version)

All four were run on Kaggle (GPU, internet on); each runs all its seeds in one go and is resumable. Their result files are in `results/revision/`.

| Notebook | What it runs | Results |
|---|---|---|
| `v11_revision_main.ipynb` | Same protocol, words and seeds as `real_vocab_transplant_v6_trainonly` (seeds 0 to 2); adds least squares and official OMP-64 on the 5000 frequent anchors, fixed length `c=1`, median-norm rescaling; stores abstract ids | `results/revision/v11_main/` |
| `v11_revision_scale.ipynb` | Same for GPT-2 large, Pythia-1B (untied) and Qwen2.5-1.5B, seed 0 | `results/revision/v11_scale/` |
| `real_vocab_transplant_v8c_awedist_25_moretext.ipynb` | Official AweDist at its default 25 snippets with extra PubMedQA abstracts (never evaluation abstracts); TokDist; OMP-64; seed 1 | `results/revision/awedist_25snippets_v8c/` |
| `ablation_ridge_v2_trainonly.ipynb` | Ablations with training-only word choice, four small pairs, seeds 1 and 2 | `results/revision/ablation_v2/` |
| `ls_vs_ridge_nll_sweep_v3_trainonly.ipynb` | Scarce-anchor NLL sweep, training-only word choice, two GPT-2 pairs, seed 1 | `results/revision/anchor_sweep_v3/` |

Statistics and tables: see the README of the repository (`analysis/make_v11_stats.py`, `make_revision_stats.py`, `make_revision_tables.py`).
