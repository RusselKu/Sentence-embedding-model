# QA on CPU — Bianca

**Environment:** CPU only, no GPU. Python 3.13.15, torch 2.14.1 (CPU), transformers 4.48.3, sentence-transformers 3.4.1, datasets 4.3.0.

## What passed

| Check | Result |
|---|---|
| `python -m pytest tests` (offline, tiny BERT) | **11 passed**: 7 existing + 4 new in `tests/test_metrics_scale.py` |
| `scripts/validate_dataset.py` | PASS: 100,000 records; 165,529 unique (165,528 stripped); 33,351 pairs; 9,488 hard negatives (28.45 %) |
| `scripts/validate_run_history.py` | PASS: all 13 runs have config, seed, hardware and results |
| Baselines match the PDF reference values | raw BERT 59.31 / 47.29 and SBERT 80.77 / 76.98: exact (team logs) |
| Hub model verified | test 67.32 reloaded = 67.32 in the table (Δ 0.00) |
| Test-set discipline | training scores dev only unless `--eval_test`; dev-only runs log `test: null` |
| Hub model loads and runs on a CPU laptop (Windows, Python 3.11) | PASS, after re-pinning sentence-transformers (finding #11) |

## Bugs and inconsistencies found

| # | Finding | Impact | Status |
|---|---|---|---|
| 1 | `sentence-transformers/stsb` scores are in [0, 1]. Runs made before the rescaling in `load_stsb_dataset` had an empty "score ≥ 4" mask, so **alignment silently used all pairs**. Proof: raw BERT dev alignment is 0.3678 in `baseline_raw_bert.json` vs. 0.1948 in `jonav_sanity_raw_bert_dev.json`, with identical Spearman and uniformity. | Alignment in `REPORT_DRAFT.md`, `unsup_simcse_default.json`, `ablation_unsup_same_mask.json` and both baseline JSONs was not comparable with the supervised runs. | **Fixed in the report**: every Part 6 number now uses the ≥ 4 definition (values from Rivaldo's notebook and `eval_hub_unsup.json`). Regression test added. The same-mask ablation keeps its old values, labeled as such, until its checkpoint is re-scored. |
| 2 | Part 7 of the draft said the reloaded Hub model had alignment 0.6090; `eval_hub_unsup.json` says 0.3501. | Wrong number in the report (a consequence of #1, not a weight mismatch). | Fixed |
| 3 | `src/analysis.py` built supervised checkpoints without `inference_mlp`, so plots and retrievals would use embeddings **without the trained MLP**, unlike `evaluate.py`. | Supervised figures would not match the evaluated model. | Fixed: uses `SimCSEModel.from_checkpoint` |
| 4 | `reports/figures/similarity_distribution_bert.png` shows only 2 score brackets, and `retrieval_analysis_bert.json` has human scores in [0, 1]. | Figure produced before the scale fix (#1). | **Done**: regenerated on Bianca's laptop, 5/5 brackets |
| 5 | The `discrepancy_explanation` heuristic in `analysis.py` labels over-predicted unrelated pairs as "low lexical overlap with synonymous meaning". | Misleading text in the JSON. | Not used; the report writes its own analysis (§4.2) |
| 6 | The assignment asks for nearest-neighbor retrievals; `analysis.py` only scores pairs. | Missing report element. | **Done**: `scripts/bianca_cpu_qa.py` run on CPU; unsup R@1 76.6 %, R@5 94.7 % (`reports/figures/nn_retrieval_unsup.md`, report §4.3) |
| 7 | `requirements.txt` used `>=` ranges, but the assignment asks for pinned dependencies. | Reproducibility | Pinned to the executed Colab versions |
| 8 | The model card template described pooling as "CLS / Mean", its Transformers example used **mean** pooling for a CLS model, it had no dev alignment or uniformity, its limitations were minimal, and it contained an off-topic example sentence. | Wrong usage instructions on the Hub | New template in `src/publish.py`; card ready in `reports/MODEL_CARD_unsup.md` |
| 9 | The unsupervised model was trained on 165,528 sentences (whitespace-stripped); the assignment reference is 165,529. | Negligible (one sentence). | Documented in the report (Part 2) |
| 11 | First pin (sentence-transformers 3.4.1, from Jonav's Colab) **could not load the Hub model**, saved with sentence-transformers 6.1.0 (`No module named 'sentence_transformers.base'`). | Anyone installing `requirements.txt` could not reproduce Part 7. | Fixed: re-pinned to the versions verified on Bianca's CPU laptop (transformers 5.18.0, sentence-transformers 6.1.0, tokenizers 0.23.2, huggingface_hub 1.33.0, torch 2.11.0) |
| 10 | The supervised exporter drops the MLP (needs a `Dense(768, 768, tanh)` module). | The supervised model cannot be published as-is. | Documented; Russel's task if the team wants it on the Hub |

## Still to do

- Upload `reports/MODEL_CARD_unsup.md` as `README.md` of the Hub repository (Russel has write access).
- Optional, if Jonav shares the Drive checkpoint: `python scripts/bianca_cpu_qa.py --skip_plot --sup_checkpoint "<path>/best_checkpoint"` for supervised retrieval and distribution figures.
- Optional, if Russel shares the same-mask checkpoint: `--ablation_checkpoint` to re-score its alignment with the corrected definition.
