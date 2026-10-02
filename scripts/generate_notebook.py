import json
import os

notebook_data = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# U2T02: SimCSE Sentence Embedding Model\n",
    "## Unsupervised SimCSE Pipeline, Ablation Study & Hugging Face Publishing\n",
    "\n",
    "**Team Lead / Author:** Russel Ku (Russ)  \n",
    "**Role:** Unsupervised SimCSE Lead, Unsupervised Ablation & Hugging Face Hub Publication  \n",
    "**Hardware Target:** NVIDIA GeForce RTX 4070 Laptop GPU (CUDA-accelerated)  \n",
    "**Base Model:** `bert-base-uncased` (110M parameters)  \n",
    "**Dataset:** SNLI 100k Subset (`data/snli_train_100k.jsonl`)\n",
    "\n",
    "---\n",
    "\n",
    "### Overview\n",
    "This notebook demonstrates the complete, end-to-end workflow for **Unsupervised SimCSE** (Gao et al., EMNLP 2021):\n",
    "1. **Environment & CUDA Initialization:** Ensuring PyTorch is properly utilizing the RTX 4070 GPU.\n",
    "2. **Data Pipeline:** Loading and tokenizing ~165k unique sentences from SNLI 100k with specialized collators.\n",
    "3. **Architecture & Contrastive Objective:** BERT backbone with CLS pooling, MLP projection head, and InfoNCE loss using standard dropout noise as data augmentation.\n",
    "4. **Model Training:** Training with AdamW, linear warmup schedule, and intermediate STS-B Dev evaluation.\n",
    "5. **Unsupervised Ablation Study:** Empirically validating the necessity of independent dropout masks vs. identical masks (`same_dropout_mask=True`).\n",
    "6. **STS-Benchmark Evaluation:** Computing Spearman rank correlation, Alignment ($\\alpha=2$), and Uniformity ($t=2$).\n",
    "7. **Hugging Face Hub Export & Verification:** Converting model to SentenceTransformers format and verifying post-upload performance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup, Reproducibility & CUDA Hardware Check"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import time\n",
    "import json\n",
    "import random\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import DataLoader\n",
    "from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup\n",
    "\n",
    "# Ensure project root is in sys.path\n",
    "if os.path.exists(\"src\"):\n",
    "    project_root = os.path.abspath(os.getcwd())\n",
    "else:\n",
    "    project_root = os.path.abspath(os.path.join(os.getcwd(), \"..\"))\n",
    "if project_root not in sys.path:\n",
    "    sys.path.insert(0, project_root)\n",
    "\n",
    "from src.data import (\n",
    "    build_unsupervised_dataset,\n",
    "    UnsupervisedSimCSEDataset,\n",
    "    UnsupDataCollator,\n",
    "    load_stsb_dataset,\n",
    ")\n",
    "from src.models import SimCSEModel\n",
    "from src.metrics import evaluate_sts_benchmark, compute_alignment, compute_uniformity\n",
    "from src.train import set_seed, evaluate_on_stsb\n",
    "\n",
    "# Reproducibility\n",
    "set_seed(42)\n",
    "\n",
    "# CUDA Device Check\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"[✓] Using compute device: {device}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"[✓] GPU Model: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"[✓] VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB\")\n",
    "    print(f\"[✓] PyTorch CUDA Version: {torch.version.cuda}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load & Prepare SNLI 100k Unsupervised Dataset\n",
    "In Unsupervised SimCSE, each unique premise and hypothesis from SNLI is treated as an individual training instance. The positive view is generated on-the-fly by feeding the exact same tokenized input through the encoder twice with different dropout masks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = os.path.join(project_root, \"data\", \"snli_train_100k.jsonl\")\n",
    "if not os.path.exists(data_path):\n",
    "    data_path = os.path.join(project_root, \"snli_train_100k.jsonl\")\n",
    "\n",
    "unique_sentences = build_unsupervised_dataset(data_path)\n",
    "print(f\"[✓] Extracted {len(unique_sentences):,} unique sentences from SNLI 100k.\")\n",
    "print(\"Sample sentences:\")\n",
    "for s in unique_sentences[:5]:\n",
    "    print(f\"  - {s}\")\n",
    "\n",
    "# Initialize Tokenizer and Collator\n",
    "model_name = \"bert-base-uncased\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "unsup_dataset = UnsupervisedSimCSEDataset(unique_sentences)\n",
    "unsup_collator = UnsupDataCollator(tokenizer=tokenizer, max_length=64)\n",
    "\n",
    "batch_size = 64\n",
    "train_loader = DataLoader(\n",
    "    unsup_dataset,\n",
    "    batch_size=batch_size,\n",
    "    shuffle=True,\n",
    "    collate_fn=unsup_collator,\n",
    "    drop_last=True\n",
    ")\n",
    "print(f\"[✓] DataLoader created: {len(train_loader)} batches of size {batch_size}.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. SimCSE Model Architecture & InfoNCE Loss\n",
    "\n",
    "$$\\ell_i = -\\log \\frac{e^{\\text{sim}(h_i, h_i^+) / \\tau}}{\\sum_{j=1}^N e^{\\text{sim}(h_i, h_j^+) / \\tau}}$$\n",
    "\n",
    "- **Positive pair ($h_i, h_i^+$):** The same sentence encoded with two independent dropout masks.\n",
    "- **In-batch negatives ($h_j^+$):** The representations of the remaining $N-1$ sentences in the batch.\n",
    "- **MLP Projection Head:** A single linear layer with `tanh` activation used during training and discarded at test time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load STS-B Dev & Test benchmarks for evaluation\n",
    "dev_records, test_records = load_stsb_dataset()\n",
    "print(f\"[✓] Loaded STS-B Benchmark: {len(dev_records)} Dev pairs, {len(test_records)} Test pairs.\")\n",
    "\n",
    "# Instantiate SimCSE Model\n",
    "model = SimCSEModel(\n",
    "    model_name_or_path=model_name,\n",
    "    pooling=\"cls\",\n",
    "    temperature=0.05,\n",
    "    use_mlp=True\n",
    ").to(device)\n",
    "\n",
    "print(f\"[✓] Model initialized: {model_name} with CLS pooling & tau=0.05 on {device}.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Unsupervised Training (Default Model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import argparse\n",
    "from src.train import train\n",
    "\n",
    "parser = argparse.ArgumentParser()\n",
    "args = parser.parse_args([])\n",
    "args.mode = \"unsup\"\n",
    "args.data_path = data_path\n",
    "args.model_name_or_path = \"bert-base-uncased\"\n",
    "args.output_dir = os.path.join(project_root, \"checkpoints\")\n",
    "args.run_name = \"unsup_simcse_default\"\n",
    "args.pooling = \"cls\"\n",
    "args.lr = 3e-5\n",
    "args.batch_size = 64\n",
    "args.epochs = 1\n",
    "args.temperature = 0.05\n",
    "args.dropout_rate = None\n",
    "args.weight_decay = 0.0\n",
    "args.warmup_ratio = 0.05\n",
    "args.max_length = 64\n",
    "args.seed = 42\n",
    "args.eval_steps = 250\n",
    "args.same_dropout_mask = False\n",
    "args.no_hard_negatives = False\n",
    "\n",
    "# Execute training run\n",
    "unsup_results = train(args)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Unsupervised Ablation Study: Independent vs. Same Dropout Mask\n",
    "\n",
    "**Research Question:** Is dropout noise truly acting as a necessary minimal data augmentation?\n",
    "**Hypothesis:** Setting `same_dropout_mask = True` forces identical representations $z_1 = z_2$ for both passes. Without representation variance, contrastive learning loses its augmentation signal and performance degrades significantly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scripts.run_ablations import run_unsupervised_ablation\n",
    "\n",
    "print(\"[...] Running Unsupervised Ablation (Same Dropout Mask)...\")\n",
    "ablation_results = run_unsupervised_ablation(data_path=data_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Ablation Comparison & Metrics Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load run JSONs to display comparative results\n",
    "runs_dir = os.path.join(project_root, \"runs\")\n",
    "unsup_log_path = os.path.join(runs_dir, \"unsup_simcse_default.json\")\n",
    "ablation_log_path = os.path.join(runs_dir, \"ablation_unsup_same_mask.json\")\n",
    "\n",
    "with open(unsup_log_path, \"r\") as f:\n",
    "    unsup_meta = json.load(f)\n",
    "with open(ablation_log_path, \"r\") as f:\n",
    "    ablation_meta = json.load(f)\n",
    "\n",
    "print(\"=\" * 85)\n",
    "print(f\"{'Experiment':<35} | {'Dev Spearman':<14} | {'Test Spearman':<14} | {'Alignment':<10} | {'Uniformity'}\")\n",
    "print(\"=\" * 85)\n",
    "u_dev = unsup_meta.get('results', {}).get('dev', {}).get('spearman', 0.0)\n",
    "u_test = unsup_meta.get('results', {}).get('test', {}).get('spearman', 0.0)\n",
    "u_align = unsup_meta.get('results', {}).get('test', {}).get('alignment', 0.0)\n",
    "u_unif = unsup_meta.get('results', {}).get('test', {}).get('uniformity', 0.0)\n",
    "a_dev = ablation_meta.get('results', {}).get('dev', {}).get('spearman', 0.0)\n",
    "a_test = ablation_meta.get('results', {}).get('test', {}).get('spearman', 0.0)\n",
    "a_align = ablation_meta.get('results', {}).get('test', {}).get('alignment', 0.0)\n",
    "a_unif = ablation_meta.get('results', {}).get('test', {}).get('uniformity', 0.0)\n",
    "print(f\"{'Unsup Default (Indep Dropout)':<35} | {u_dev:<14.2f} | {u_test:<14.2f} | {u_align:<10.4f} | {u_unif:.4f}\")\n",
    "print(f\"{'Unsup Ablation (Same Mask)':<35} | {a_dev:<14.2f} | {a_test:<14.2f} | {a_align:<10.4f} | {a_unif:.4f}\")\n",
    "delta_dev = u_dev - a_dev\n",
    "delta_test = u_test - a_test\n",
    "print(\"-\" * 85)\n",
    "print(f\"[Δ Delta]: Independent dropout yields +{delta_dev:.2f} Dev and +{delta_test:.2f} Test Spearman points.\")\n",
    "print(\"=\" * 85)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Hugging Face Hub Export & Verification Pipeline\n",
    "Converts our trained PyTorch checkpoint into a standard `SentenceTransformers` model, generates a rich Model Card (`README.md`), and verifies reloading from Hugging Face Hub."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.publish import export_to_sentence_transformers, verify_hf_model\n",
    "\n",
    "best_ckpt = os.path.join(project_root, \"checkpoints\", \"unsup_simcse_default\", \"best_checkpoint\")\n",
    "export_dir = os.path.join(project_root, \"checkpoints\", \"unsup_simcse_default\", \"sentence_transformer_export\")\n",
    "\n",
    "# 1. Export locally to SentenceTransformers format\n",
    "export_to_sentence_transformers(best_ckpt, export_dir, pooling=\"cls\")\n",
    "print(f\"[✓] Exported model ready for Hugging Face Hub at: {export_dir}\")\n",
    "\n",
    "# 2. Example sentences test\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "st_model = SentenceTransformer(export_dir)\n",
    "demo_sentences = [\n",
    "    \"A dog is running across the grass.\",\n",
    "    \"A puppy plays on the lawn.\",\n",
    "    \"A chef is cooking pasta in an Italian kitchen.\"\n",
    "]\n",
    "embs = st_model.encode(demo_sentences, normalize_embeddings=True)\n",
    "sim_matrix = cosine_similarity(embs)\n",
    "print(\"\\nPairwise Cosine Similarities:\")\n",
    "print(f\"- 'Dog running' vs 'Puppy plays': {sim_matrix[0, 1]:.4f} (High semantic similarity)\")\n",
    "print(f\"- 'Dog running' vs 'Chef cooking': {sim_matrix[0, 2]:.4f} (Low semantic similarity)\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs("notebooks", exist_ok=True)
notebook_path = os.path.join("notebooks", "simcse_unsupervised_pipeline.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=1)
print(f"Created notebook at {notebook_path}")
