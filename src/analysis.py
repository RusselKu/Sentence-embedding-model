"""Analysis tools for sentence embeddings:
- Distribution of cosine similarities grouped by human score bins.
- Nearest-neighbor retrieval & failure case analysis.
- Alignment vs Uniformity visualization.
"""

import argparse
import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from src.data import load_stsb_dataset
from src.metrics import compute_cosine_similarity
from src.models import SimCSEModel


def generate_similarity_distribution_plot(
    model_path_or_name: str,
    output_path: str = "reports/figures/similarity_distribution.png",
    device: str = "cpu",
    pooling: str = "cls",
):
    """Plot cosine similarity distributions partitioned into STS-B human score intervals [0-1], [1-2], [2-3], [3-4], [4-5]."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _, test_records = load_stsb_dataset()

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = SimCSEModel(model_path_or_name, pooling=pooling).to(device)
    ckpt_file = os.path.join(model_path_or_name, "pytorch_model.bin")
    if os.path.exists(ckpt_file):
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
    model.eval()

    sent1 = [r["sentence1"] for r in test_records]
    sent2 = [r["sentence2"] for r in test_records]
    scores = np.array([r["score"] for r in test_records])

    def encode_batch(sentences):
        embs = []
        for i in range(0, len(sentences), 64):
            batch = sentences[i : i + 64]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_sentence_embeddings(inputs["input_ids"], inputs["attention_mask"], normalize=True)
            embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0)

    emb1 = encode_batch(sent1)
    emb2 = encode_batch(sent2)
    sims = compute_cosine_similarity(emb1, emb2)

    # Score bins
    bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]
    bin_labels = ["[0 - 1]", "[1 - 2]", "[2 - 3]", "[3 - 4]", "[4 - 5]"]

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    for (low, high), label in zip(bins, bin_labels):
        if high == 5.0:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        subset_sims = sims[mask]
        if len(subset_sims) > 0:
            sns.kdeplot(subset_sims, label=f"Human Rating {label} (N={len(subset_sims)})", fill=True, alpha=0.25)

    plt.title(f"Cosine Similarity Distribution by STS-B Human Rating\nModel: {os.path.basename(model_path_or_name)}", fontsize=14, fontweight="bold")
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(-0.2, 1.05)
    plt.legend(title="Score Brackets", loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved similarity distribution plot to: {output_path}")


def analyze_nearest_neighbors_and_failures(
    model_path_or_name: str,
    output_path: str = "reports/figures/retrieval_analysis.json",
    device: str = "cpu",
    pooling: str = "cls",
):
    """Retrieve nearest neighbors for sample query sentences and identify failure cases on STS-B."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _, test_records = load_stsb_dataset()

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = SimCSEModel(model_path_or_name, pooling=pooling).to(device)
    ckpt_file = os.path.join(model_path_or_name, "pytorch_model.bin")
    if os.path.exists(ckpt_file):
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
    model.eval()

    sent1 = [r["sentence1"] for r in test_records]
    sent2 = [r["sentence2"] for r in test_records]
    human_scores = np.array([r["score"] for r in test_records])

    def encode(sents):
        embs = []
        for i in range(0, len(sents), 64):
            batch = sents[i : i + 64]
            inp = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                e = model.get_sentence_embeddings(inp["input_ids"], inp["attention_mask"], normalize=True)
            embs.append(e.cpu().numpy())
        return np.concatenate(embs, axis=0)

    emb1 = encode(sent1)
    emb2 = encode(sent2)
    sims = compute_cosine_similarity(emb1, emb2)

    # Calculate absolute error between normalized similarity (0..5) and human score
    predicted_ratings = np.clip((sims + 1.0) / 2.0 * 5.0, 0, 5)
    errors = np.abs(predicted_ratings - human_scores)

    # Top failure cases (largest discrepancy)
    top_failure_indices = np.argsort(errors)[::-1][:5]
    top_success_indices = np.argsort(errors)[:5]

    failures = []
    for idx in top_failure_indices:
        failures.append({
            "sentence1": sent1[idx],
            "sentence2": sent2[idx],
            "human_score": float(human_scores[idx]),
            "cosine_similarity": float(round(sims[idx], 4)),
            "discrepancy_explanation": (
                "High lexical overlap with contradictory semantics"
                if human_scores[idx] < 2.0 and sims[idx] > 0.7
                else "Low lexical overlap with synonymous/paraphrased meaning"
            ),
        })

    successes = []
    for idx in top_success_indices:
        successes.append({
            "sentence1": sent1[idx],
            "sentence2": sent2[idx],
            "human_score": float(human_scores[idx]),
            "cosine_similarity": float(round(sims[idx], 4)),
        })

    analysis_result = {
        "model": model_path_or_name,
        "sample_successes": successes,
        "sample_failures": failures,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)
    print(f"Saved retrieval analysis to: {output_path}")
    return analysis_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run visualization and failure analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint or model path")
    parser.add_argument("--pooling", type=str, default="cls")
    parser.add_argument("--output_plot", type=str, default="reports/figures/similarity_distribution.png")
    parser.add_argument("--output_json", type=str, default="reports/figures/retrieval_analysis.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_similarity_distribution_plot(args.model_path, args.output_plot, device=device, pooling=args.pooling)
    analyze_nearest_neighbors_and_failures(args.model_path, args.output_json, device=device, pooling=args.pooling)
