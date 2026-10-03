# SimCSE: Simple Contrastive Learning of Sentence Embeddings

An end-to-end, modular implementation of **SimCSE (Gao et al., EMNLP 2021)** for training state-of-the-art sentence embedding models using `bert-base-uncased` on an SNLI 100k subset, evaluating on STS-Benchmark (STS-B), conducting ablation studies, and publishing the final model to the Hugging Face Hub.

---

## 🌟 Repository Architecture

```text
Sentence-embedding-model/
├── data/
│   ├── .gitkeep
│   └── snli_train_100k.jsonl       # 100k records of premise, hypothesis, and label
├── src/
│   ├── __init__.py
│   ├── data.py                      # SNLI 100k & STS-B dataset loaders and collators
│   ├── models.py                    # SimCSE PyTorch model with InfoNCE loss & MLP head
│   ├── metrics.py                   # Spearman correlation, Alignment & Uniformity metrics
│   ├── train.py                     # Unified training pipeline with dev checkpointing
│   ├── evaluate.py                  # Standalone STS-B benchmark evaluation tool
│   ├── analysis.py                  # Similarity distribution plots & failure case analysis
│   └── publish.py                   # Hugging Face Hub export, publishing & verification
├── scripts/
│   ├── run_sanity_checks.py         # Baseline verification (Raw BERT & SBERT-2019)
│   ├── train_unsupervised.py        # Helper script for Unsupervised SimCSE
│   ├── train_supervised.py          # Helper script for Supervised SimCSE
│   └── run_ablations.py             # Automation runner for Unsup & Sup ablations
├── reports/
│   ├── figures/                     # Output plots and retrieval analysis
│   └── REPORT_DRAFT.md              # Template & solutions for assignment report
├── runs/                            # JSON logs for all runs (config, seed, hardware, results)
├── TASK_DISTRIBUTION.md             # Role & task assignment for all 5 team members
├── requirements.txt                 # Pinned dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/RusselKu/Sentence-embedding-model.git
cd Sentence-embedding-model

# Create virtual environment (Python 3.10+)
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Place `snli_train_100k.jsonl` in the `data/` directory:
```text
data/snli_train_100k.jsonl
```

---

## 📊 Step-by-Step Workflow

### Step 1: Baseline Sanity Checks
Before running any training, verify that your evaluation code reproduces the expected reference values on STS-B:
- **Raw `bert-base-uncased` (Mean Pooling)**: `59.31` Dev / `47.29` Test
- **SBERT-2019 (`bert-base-nli-mean-tokens`)**: `80.77` Dev / `76.98` Test

Run the sanity check:
```bash
python scripts/run_sanity_checks.py
```

---

### Step 2: Train Unsupervised SimCSE
Uses standard dropout as minimal data augmentation. The encoder takes two forward passes of the same batch with independent dropout masks.

```bash
python scripts/train_unsupervised.py \
  --data_path data/snli_train_100k.jsonl \
  --lr 3e-5 \
  --batch_size 64 \
  --epochs 1 \
  --temperature 0.05 \
  --pooling cls \
  --run_name unsup_simcse_default
```

---

### Step 3: Train Supervised SimCSE (with Hard Negatives)
Uses `(premise, entailment, contradiction)` triplets with InfoNCE contrastive loss:

Jonav's completed Colab run selected LR `3e-5`, temperature `0.05`, batch 64 and 3 epochs using dev only: **81.67 Dev / 79.02 Test**. Hard-negative ablation improved test by **1.91 points**. See [the supervised report](reports/JONAV_RESULTADOS.md), [results CSV](reports/jonav/supervised_benchmark.csv), and [portable Colab notebook](notebooks/jonav_simcse_supervisado.ipynb). The notebook uses `requirements-colab.txt` and saves checkpoints in Drive. Training now evaluates dev by default; `--eval_test` explicitly enables final test scoring. Development-only logs have `test: null`.

Supervised checkpoints retain the trained MLP for inference and are loaded with `SimCSEModel.from_checkpoint`. Their weights remain in Drive, outside Git. The existing Hub exporter needs adaptation before publishing these supervised checkpoints; publishing the already-exported unsupervised model remains a separate workflow.

```bash
python scripts/train_supervised.py \
  --data_path data/snli_train_100k.jsonl \
  --lr 5e-5 \
  --batch_size 64 \
  --epochs 3 \
  --temperature 0.05 \
  --pooling cls \
  --run_name sup_simcse_default
```

---

### Step 4: Run Ablation Studies
Run both required ablations:
1. **Unsupervised Ablation**: Same dropout mask for both views.
2. **Supervised Ablation**: Hard negatives turned OFF (only entailment positive pairs).

```bash
python scripts/run_ablations.py --ablation all --data_path data/snli_train_100k.jsonl
```

---

### Step 5: Evaluate Checkpoints & Analysis

#### 1. Evaluate on STS-B (Spearman, Alignment, Uniformity)
```bash
python src/evaluate.py \
  --model_path checkpoints/sup_simcse_default/best_checkpoint \
  --model_type simcse \
  --pooling cls \
  --output_json runs/eval_sup_best.json
```

#### 2. Plot Cosine Similarity Distribution & Analyze Failures
```bash
python src/analysis.py \
  --model_path checkpoints/sup_simcse_default/best_checkpoint \
  --pooling cls \
  --output_plot reports/figures/similarity_distribution.png \
  --output_json reports/figures/retrieval_analysis.json
```

---

### Step 6: Publish to Hugging Face Hub & Verification
Export the best checkpoint as a standard `SentenceTransformer` model, upload it to the Hugging Face Hub, and verify by reloading and scoring the STS-B test split:

```bash
# Set your Hugging Face Token (or run huggingface-cli login)
python src/publish.py \
  --checkpoint_dir checkpoints/sup_simcse_default/best_checkpoint \
  --repo_id "<your-username>/simcse-bert-snli" \
  --run_json runs/sup_simcse_default.json \
  --pooling cls
```

The script automatically:
1. Converts the checkpoint into `sentence-transformers` architecture.
2. Generates a comprehensive **Model Card** (`README.md`).
3. Pushes the model repository to the Hugging Face Hub.
4. Reloads the model from the Hub (`SentenceTransformer('<your-username>/simcse-bert-snli')`).
5. Re-evaluates on STS-B test split and confirms the result matches the benchmark table.

---

## 👥 Team & Task Distribution

See [TASK_DISTRIBUTION.md](file:///c:/Users/russe/Documents/github_repo/Sentence-embedding-model/TASK_DISTRIBUTION.md) for detailed responsibilities and checklists.

| Member | Hardware | Primary Responsibilities |
| :--- | :--- | :--- |
| **Russel (Russ)** | GPU 🟢 | Unsupervised SimCSE pipeline, Unsup ablation, HF Hub publishing & verification lead |
| **Joni** | GPU 🟢 | Supervised SimCSE pipeline, Supervised ablation (hard negatives), hyperparam sweeps |
| **Rivaldo** | GPU 🟢 | Baseline verification, Alignment & Uniformity metrics, retrieval failure analysis |
| **Damian** | CPU ⚪ | SNLI 100k dataset parser, run history logging, similarity distribution plotting |
| **Bianca** | CPU ⚪ | Part 1 theoretical answers synthesis, HF Model Card documentation, report compilation |

---

## 📚 References
- **SimCSE**: Gao, Yao & Chen, *SimCSE: Simple Contrastive Learning of Sentence Embeddings*, EMNLP 2021 ([arXiv:2104.08821](https://arxiv.org/abs/2104.08821))
- **Sentence-BERT**: Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP 2019 ([arXiv:1908.10084](https://arxiv.org/abs/1908.10084))
- **Alignment and Uniformity**: Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML 2020 ([arXiv:2005.10242](https://arxiv.org/abs/2005.10242))
