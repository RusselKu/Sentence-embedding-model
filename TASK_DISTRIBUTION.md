# Team Activity Distribution & Execution Plan

**Project:** SimCSE Sentence Embedding Model (U2T02)  
**Team Members (5):**
- **Russel (Russ)** — *GPU Available* 🟢
- **Joni** — *GPU Available* 🟢
- **Rivaldo** — *GPU Available* 🟢
- **Damian** — *No GPU (CPU)* ⚪
- **Bianca** — *No GPU (CPU)* ⚪

---

## 1. Workload & Hardware-Aware Allocation Matrix

| Member | Hardware | Core Role & Responsibilities | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Russel (Russ)** | GPU 🟢 | **Unsupervised SimCSE Lead & Publication**<br>• Model architecture & base pipeline setup<br>• Unsupervised SimCSE training runs & tuning<br>• Unsupervised Ablation (same dropout mask)<br>• Hugging Face export, publishing & verification | • `src/models.py`, `src/train.py`<br>• Best Unsup checkpoint & run logs<br>• Unsupervised ablation results<br>• Hugging Face Hub published model |
| **Joni** | GPU 🟢 | **Supervised SimCSE Lead & Hard Negatives**<br>• Supervised SimCSE training with hard negatives<br>• Supervised Ablation (Hard negatives ON vs. OFF)<br>• Hyperparameter sweep (learning rate, temperature $\tau$, epochs)<br>• Checkpoint selection based on STS-B Dev | • Best Supervised checkpoint & run logs<br>• Supervised ablation results<br>• Hyperparameter sweep logs (`runs/*.json`) |
| **Rivaldo** | GPU 🟢 | **Evaluation, Alignment & Uniformity Lead**<br>• STS-B evaluation harness & baseline sanity checks (`bert-base-uncased`, `SBERT-2019`)<br>• Alignment and Uniformity calculation implementation (Wang & Isola, 2020)<br>• Failure cases & nearest-neighbor retrieval analysis<br>• Benchmark comparison table generation | • `src/metrics.py`, `src/evaluate.py`<br>• Baseline verification JSONs<br>• Alignment/Uniformity scores for all runs<br>• Retrieval failure cases report section |
| **Damian** | CPU ⚪ | **Data Engineering, Logging & Visualizations**<br>• SNLI 100k data parser (`data/snli_train_100k.jsonl`)<br>• Unsupervised unique sentence extraction (165k target)<br>• Supervised triplet extraction (premise, entailment, contradiction)<br>• Cosine similarity distribution plots grouped by human score brackets | • `src/data.py`<br>• Dataset validation script<br>• `runs/run_history.json` logger tracker<br>• Figures: `similarity_distribution.png` |
| **Bianca** | CPU ⚪ | **Theory, Documentation, Model Cards & QA**<br>• Part 1 theoretical questions synthesis & math review<br>• Hugging Face Model Card (`README.md` for Hub) drafting<br>• Environment validation & QA testing on CPU<br>• Final report compilation & proofreading | • Part 1 solutions in `reports/REPORT_DRAFT.md`<br>• Standardized Hugging Face Model Card template<br>• Complete final report document |

---

## 2. Step-by-Step Execution Plan by Member

### 🟢 Russel (Russ) — GPU
1. **Unsupervised Training:**
   ```bash
   python scripts/train_unsupervised.py --data_path data/snli_train_100k.jsonl --lr 3e-5 --batch_size 64 --epochs 1 --temperature 0.05
   ```
2. **Unsupervised Ablation (Same Dropout Mask):**
   ```bash
   python scripts/run_ablations.py --ablation unsup --data_path data/snli_train_100k.jsonl
   ```
3. **Hugging Face Publication & Verification:**
   ```bash
   python src/publish.py --checkpoint_dir checkpoints/unsup_simcse_default/best_checkpoint --repo_id "<your-username>/simcse-bert-snli" --run_json runs/unsup_simcse_default.json
   ```

### 🟢 Joni — GPU
1. **Supervised Training (with Hard Negatives):**
   ```bash
   python scripts/train_supervised.py --data_path data/snli_train_100k.jsonl --lr 5e-5 --batch_size 64 --epochs 3 --temperature 0.05
   ```
2. **Supervised Ablation (Hard Negatives OFF):**
   ```bash
   python scripts/run_ablations.py --ablation sup --data_path data/snli_train_100k.jsonl
   ```
3. **Hyperparameter Tuning:**
   - Experiment with $\tau \in \{0.03, 0.05, 0.1\}$ and learning rate $\in \{3\text{e-}5, 5\text{e-}5\}$.
   - Record all resulting JSON files in `runs/`.

### 🟢 Rivaldo — GPU
1. **Baseline Sanity Checks:**
   ```bash
   python scripts/run_sanity_checks.py
   ```
   *Verify against target reference values:*
   - Raw BERT (mean pooling): Dev $\approx 59.31$, Test $\approx 47.29$
   - SBERT-2019: Dev $\approx 80.77$, Test $\approx 76.98$
2. **Benchmark Table Compilation:**
   - Run `src/evaluate.py` across all trained checkpoints to compute Spearman, Alignment ($\alpha=2$), and Uniformity ($t=2$).
3. **Failure Case Retrieval Analysis:**
   ```bash
   python src/analysis.py --model_path checkpoints/sup_simcse_default/best_checkpoint
   ```
   - Identify at least 1 high-overlap failure case and 1 semantic discrepancy for the report.

### ⚪ Damian — CPU
1. **Data Verification:**
   - Place `snli_train_100k.jsonl` in `data/`.
   - Validate that unique sentences total $\approx 165,529$ and supervised pairs total $\approx 33,351$ (with $\approx 28\%$ having contradiction hypotheses).
2. **Similarity Distribution Plots:**
   - Run `src/analysis.py` to produce density plots across human score brackets $[0-1], [1-2], [2-3], [3-4], [4-5]$.
3. **Run Log Aggregation:**
   - Verify all runs are properly stored in `runs/run_history.json`.

### ⚪ Bianca — CPU
1. **Part 1 Theoretical Solutions:**
   - Review and finalize Part 1 answers in [REPORT_DRAFT.md](file:///c:/Users/russe/Documents/github_repo/Sentence-embedding-model/reports/REPORT_DRAFT.md) (Equations 1 & 5, $\tau$ temperature role, SBERT vs SimCSE loss differences, Alignment & Uniformity theory).
2. **Hugging Face Model Card:**
   - Ensure the model card includes training data details, hyperparameters, usage instructions, and limitations.
3. **Final Report Assembly:**
   - Merge benchmark tables, ablation deltas, figures, and gap discussion into the final submission report.
   - **Crucial:** Confirm all 5 team member names are clearly listed on the report.

---

## 3. Mandatory Deliverables Checklist

- [x] **Code Repository:** Clean, structured codebase with pinned `requirements.txt`.
- [x] **Data Pipeline:** SNLI 100k parsed for both Unsupervised (unique sentences) and Supervised (entailment + hard negatives).
- [x] **Baselines Verified:** Raw BERT ($59.31 / 47.29$) and SBERT-2019 ($80.77 / 76.98$) sanity check passed.
- [x] **Trained Models (Unsupervised Lead):** Unsupervised SimCSE trained (Dev: $76.32$, Test: $67.32$) with best checkpoints saved.
- [ ] **Trained Models (Supervised Lead):** Supervised SimCSE model training (Joni).
- [x] **Ablations Executed (Unsupervised):** Independent dropout mask vs. Same dropout mask ($\Delta = +19.54$ Test points).
- [ ] **Ablations Executed (Supervised):** Hard negatives ON vs. OFF (Joni).
- [x] **Hugging Face Hub Model:** Published [RusselKuAguilar/simcse-bert-uncased-unsup](https://huggingface.co/RusselKuAguilar/simcse-bert-uncased-unsup) with rich Model Card, verified by reloading and re-evaluating on STS-B Test ($67.32$, $\Delta = 0.00$).
- [x] **Run Logs:** `runs/unsup_simcse_default.json` and `runs/ablation_unsup_same_mask.json` generated and tracked.
- [ ] **Final Report:** Complete PDF/Markdown covering Part 1–8 with team names (Bianca/Team).
