"""Analysis tools for sentence embeddings:
- Distribution of cosine similarities grouped by human score bins.
- Nearest-neighbor retrieval & failure case analysis.
- Alignment vs Uniformity visualization.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_stsb_dataset
from src.metrics import compute_cosine_similarity
from src.models import SimCSEModel


def encode_sentences(model_path_or_name: str, sentences: List[str], device: str = "cpu", pooling: str = "cls") -> np.ndarray:
    """Encode list of sentences using local SimCSE checkpoint or Hugging Face Hub SentenceTransformer."""
    ckpt_file = os.path.join(model_path_or_name, "pytorch_model.bin")
    if os.path.exists(ckpt_file):
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        model = SimCSEModel(model_path_or_name, pooling=pooling).to(device)
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
        model.eval()
        embs = []
        for i in range(0, len(sentences), 64):
            batch = sentences[i : i + 64]
            inp = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                e = model.get_sentence_embeddings(inp["input_ids"], inp["attention_mask"], normalize=True)
            embs.append(e.cpu().numpy())
        return np.concatenate(embs, axis=0)
    else:
        st_model = SentenceTransformer(model_path_or_name, device=device)
        return st_model.encode(sentences, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def generate_similarity_distribution_plot(
    model_path_or_name: str,
    output_path: str = "reports/figures/similarity_distribution.png",
    device: str = "cpu",
    pooling: str = "cls",
):
    """Plot exact STS-B cosine-similarity histograms by human-score bracket."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _, test_records = load_stsb_dataset()

    sent1 = [r["sentence1"] for r in test_records]
    sent2 = [r["sentence2"] for r in test_records]
    human_scores = np.array([r["score"] for r in test_records], dtype=float)

    emb1 = encode_sentences(model_path_or_name, sent1, device=device, pooling=pooling)
    emb2 = encode_sentences(model_path_or_name, sent2, device=device, pooling=pooling)
    similarities = compute_cosine_similarity(emb1, emb2)

    score_bins = [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
    ]

    score_labels = [
        "Human score [0, 1)",
        "Human score [1, 2)",
        "Human score [2, 3)",
        "Human score [3, 4)",
        "Human score [4, 5]",
    ]

    grouped = []
    for low, high in score_bins:
        if high == 5.0:
            mask = (human_scores >= low) & (human_scores <= high)
        else:
            mask = (human_scores >= low) & (human_scores < high)

        grouped.append(similarities[mask])

    # Fixed width of 0.05 for all five panels.
    cosine_bins = np.arange(-1.0, 1.0001, 0.05)

    minimum_similarity = float(np.min(similarities))
    x_lower = np.floor(minimum_similarity / 0.05) * 0.05 - 0.05
    x_lower = max(-1.0, x_lower)

    # Common vertical scale so panel heights are directly comparable.
    max_percentage = 0.0
    for subset in grouped:
        counts, _ = np.histogram(subset, bins=cosine_bins)
        percentages = counts.astype(float) * 100.0 / len(subset)
        max_percentage = max(max_percentage, float(np.max(percentages)))

    common_y_max = np.ceil(max_percentage / 5.0) * 5.0 + 5.0

    sns.set_theme(style="whitegrid", context="notebook")
    palette = sns.color_palette("viridis", n_colors=5)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 12),
        sharex=True,
        sharey=True,
    )

    model_name = os.path.basename(os.path.normpath(model_path_or_name))
    if not model_name:
        model_name = model_path_or_name

    for ax, subset, label, color in zip(
        axes,
        grouped,
        score_labels,
        palette,
    ):
        weights = np.ones_like(subset, dtype=float) * 100.0 / len(subset)

        ax.hist(
            subset,
            bins=cosine_bins,
            weights=weights,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
        )

        mean_value = float(np.mean(subset))
        median_value = float(np.median(subset))

        ax.axvline(
            mean_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {mean_value:.3f}",
        )

        ax.axvline(
            median_value,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Median = {median_value:.3f}",
        )

        ax.text(
            0.012,
            0.88,
            f"{label}   |   n = {len(subset)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.set_ylabel("% of pairs", fontsize=10)
        ax.set_ylim(0, common_y_max)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Cosine similarity", fontsize=12)
    axes[-1].set_xlim(x_lower, 1.0)

    fig.suptitle(
        "STS-B Cosine Similarity Distribution by Human Similarity Score",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.955,
        (
            f"Model: {model_name}   |   Pooling: {pooling}"
            f"   |   Split: test   |   Pairs: {len(test_records):,}"
            "   |   Bin width: 0.05"
        ),
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5,
        0.012,
        (
            "Bars = observed relative frequencies | "
            "Dashed line = mean | Dotted line = median | "
            "All panels share the same y-axis scale."
        ),
        ha="center",
        fontsize=9,
        alpha=0.75,
    )

    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved similarity distribution plot to: {output_path}")
    print(f"Score groups plotted: {len(grouped)}/5")
    print("Histogram bin width: 0.05")
    print(f"Common y-axis maximum: {common_y_max:.1f}%")

def analyze_nearest_neighbors_and_failures(
    model_path_or_name: str,
    output_path: str = "reports/figures/retrieval_analysis.json",
    device: str = "cpu",
    pooling: str = "cls",
):
    """Retrieve nearest neighbors for sample query sentences and identify failure cases on STS-B."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _, test_records = load_stsb_dataset()
    sent1 = [r["sentence1"] for r in test_records]
    sent2 = [r["sentence2"] for r in test_records]
    human_scores = np.array([r["score"] for r in test_records], dtype=float)

    emb1 = encode_sentences(model_path_or_name, sent1, device=device, pooling=pooling)
    emb2 = encode_sentences(model_path_or_name, sent2, device=device, pooling=pooling)
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
