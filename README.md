# Calibrated ridge maps for token-embedding transplantation: code and results

Code, notebooks and result files for the manuscript "Calibrated Ridge Maps for Token-Embedding Transplantation Across Language Models".
Corresponding author: Shibdas Bhattacharya (23dp0094@iitism.ac.in).

> Status: draft package prepared for upload to a public repository. Licence, repository URL and DOI are to be filled in by the authors.

## Layout
- `notebooks/` one notebook per experiment (Kaggle or Colab, one GPU; run top to bottom; each writes `.pkl` result files).
- `results/` the result files behind the tables (`runs16/` the 16 real-vocabulary runs; `v5_small_and_scale/` the seven PubMed seed-1 pairs; `other/` least squares versus ridge anchor sweep). Qwen2.5-1.5B exists only as its printed report (the per-sentence file was not kept).
- `REPRODUCE.md` environment, checkpoints, preprocessing, hyperparameters, seeds, rerun check, table-to-script map.
- `analysis/` scripts that turn result files into tables and statistics (run from inside `analysis/`).
- `manuscript_source/` LaTeX source, tables and figures (`main.tex`, build with `latexmk -pdf main.tex`).

## Which notebook gives which table
(Assigned from the notebook names and our run history; check each row before release.)

| Manuscript part | Notebook |
|---|---|
| Real vocabulary expansion, 16 runs; OMP-64 official and centred | `real_vocab_transplant_domain_v3_official_omp64.ipynb` |
| Official WECHSEL (Table 3) | `real_vocab_transplant_domain_v4_wechsel.ipynb` |
| Seven-pair comparison (Table 4): four small pairs | `real_vocab_transplant_v5_main.ipynb` |
| GPT-2 large | `real_vocab_transplant_v5_scale.ipynb` |
| Pythia-1B (untied) | `real_vocab_transplant_v5_scale_pythia1b.ipynb` |
| Qwen2.5-1.5B | `real_vocab_transplant_v5_scale_qwen15b.ipynb` |
| Zero-shot tokenizer swap | `exp4_v2_fair_official_baselines.ipynb` |
| Ablations | `ablation_ridge_v1.ipynb`, `ablation_ridge_v1_gpt2.ipynb` |
| Downstream fine-tuning / frozen probe | `exp5b_downstream_pubmed_rct.ipynb`, `exp5c_frozen_probe.ipynb` |
| Scarce anchors (least squares versus ridge) | `ls_vs_ridge_anchor_sweep.ipynb`, `ls_vs_ridge_nll_sweep_v2.ipynb` |
| Mapping time (same device) | `speed_benchmark_ridge_vs_omp.ipynb` |
| float32 versus float64 directions | `ridge_fp32_vs_fp64_check.ipynb` |

## Reproducing the statistics
```
cd analysis
python stats16.py          # Holm over the 32 comparisons with centred OMP-64 (16 runs)
python stats_family.py     # Holm over the 98 comparisons of Table 4, equivalence check
python make_v5_tables.py   # tables/scale_*.tex from the seven-pair results
python make_realvocab_v3.py  # 16-run summary (writes to /tmp/rv and tables/)
```
Intervals are paired percentile bootstraps over evaluation sentences (5000 resamples). Sentences are not grouped by abstract, because the result files do not store abstract identifiers.

## Notes and known limits
- Package versions of the reported runs were not recorded; `notebooks/capture_environment.ipynb` records the current environment (see `REPRODUCE.md`).
- SALT-ours, LS, Procrustes and OMP-32 are our own implementations, not official code. FOCUS, WECHSEL and OMP-64 (mergekit `batch_omp`) are the official implementations.
- Results are single seeds for the three larger receivers; seeds resample sentences and training batches, not words or abstracts.
- The mapping-time lines printed inside the real-vocabulary notebooks compare CPU float64 ridge with GPU OMP and must not be used for speed claims; use `speed_benchmark_ridge_vs_omp.ipynb`.
