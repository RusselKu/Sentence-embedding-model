"""Sanity check script for environment and baseline scores.
Validates raw bert-base-uncased and SBERT-2019 reference scores against PDF expectations:
- raw bert-base-uncased (mean pooling): ~59.31 dev / ~47.29 test
- SBERT-2019 (bert-base-nli-mean-tokens): ~80.77 dev / ~76.98 test
"""

import os
import sys
import torch

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import run_evaluation


def main():
    print("=" * 60)
    print("SIMCSE ENVIRONMENT & BASELINE SANITY CHECK")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    os.makedirs("runs", exist_ok=True)

    # 1. Evaluate Raw BERT-base-uncased (Mean Pooling)
    print("\n>>> 1. Evaluating Raw bert-base-uncased Baseline (Mean Pooling)...")
    raw_bert_results = run_evaluation(
        model_path_or_name="bert-base-uncased",
        model_type="raw_bert",
        pooling="mean",
        output_json="runs/baseline_raw_bert.json",
    )

    # 2. Evaluate SBERT-2019 Baseline (bert-base-nli-mean-tokens)
    print("\n>>> 2. Evaluating SBERT-2019 Baseline (bert-base-nli-mean-tokens)...")
    sbert_results = run_evaluation(
        model_path_or_name="sentence-transformers/bert-base-nli-mean-tokens",
        model_type="sbert",
        pooling="mean",
        output_json="runs/baseline_sbert_2019.json",
    )

    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON AGAINST PDF REFERENCE VALUES")
    print("=" * 60)
    print(f"{'Model':<30} | {'Dev (Target)':<14} | {'Dev (Actual)':<14} | {'Test (Target)':<14} | {'Test (Actual)':<14}")
    print("-" * 94)
    print(f"{'Raw BERT (mean pooling)':<30} | {'59.31':<14} | {raw_bert_results['dev']['spearman']:<14.2f} | {'47.29':<14} | {raw_bert_results['test']['spearman']:<14.2f}")
    print(f"{'SBERT-2019':<30} | {'80.77':<14} | {sbert_results['dev']['spearman']:<14.2f} | {'76.98':<14} | {sbert_results['test']['spearman']:<14.2f}")
    print("=" * 60)
    print("Sanity checks finished! Evaluation pipeline verified.")


if __name__ == "__main__":
    main()
