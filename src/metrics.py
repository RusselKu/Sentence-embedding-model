"""Evaluation metrics for sentence embeddings:
- Spearman rank correlation (STS-B)
- Alignment and Uniformity (Wang & Isola, ICML 2020)
- Cosine similarity utilities
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Union


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of row vectors.
    
    Args:
        emb1: Shape (N, D)
        emb2: Shape (N, D)
    Returns:
        1D array of cosine similarities of length N.
    """
    # Normalize vectors
    norm1 = emb1 / np.clip(np.linalg.norm(emb1, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    norm2 = emb2 / np.clip(np.linalg.norm(emb2, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    similarities = np.sum(norm1 * norm2, axis=1)
    return similarities


def compute_spearman_correlation(predicted_scores: Union[np.ndarray, List[float]],
                                 ground_truth_scores: Union[np.ndarray, List[float]]) -> float:
    """Compute Spearman rank correlation (scaled x100).
    
    Args:
        predicted_scores: Predicted cosine similarities.
        ground_truth_scores: Human ratings from STS-B.
    Returns:
        Spearman correlation multiplied by 100.
    """
    corr, _ = spearmanr(predicted_scores, ground_truth_scores)
    return float(corr * 100.0)


def compute_alignment(x: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> float:
    """Compute Alignment metric from Wang & Isola (ICML 2020).
    
    alignment = E_{(x, y) ~ p_pos} [ ||f(x) - f(y)||_2^alpha ]
    Vectors should be L2-normalized on the unit hypersphere.
    
    Args:
        x: Tensor of shape (N, D), normalized representations of positive view 1.
        y: Tensor of shape (N, D), normalized representations of positive view 2.
        alpha: Exponent power (default: 2).
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return torch.norm(x - y, p=2, dim=-1).pow(alpha).mean().item()


def compute_uniformity(x: torch.Tensor, t: float = 2.0) -> float:
    """Compute Uniformity metric from Wang & Isola (ICML 2020).
    
    uniformity = log E_{x, y ~ p_data} [ exp(-t * ||f(x) - f(y)||_2^2) ]
    Vectors should be L2-normalized on the unit hypersphere.
    
    Args:
        x: Tensor of shape (N, D), normalized representations.
        t: Gaussian kernel parameter (default: 2).
    """
    x = F.normalize(x, p=2, dim=-1)
    # Pairwise squared Euclidean distance: ||x_i - x_j||^2 = 2 - 2 * <x_i, x_j>
    # pdist computes pairwise euclidean distance
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_pdist).mean()).item()


def evaluate_sts_benchmark(embeddings1: np.ndarray,
                           embeddings2: np.ndarray,
                           labels: List[float]) -> Dict[str, float]:
    """Complete evaluation on STS dataset split.
    
    Returns:
        Dict with 'spearman', 'alignment', 'uniformity'.
    """
    sims = compute_cosine_similarity(embeddings1, embeddings2)
    spearman = compute_spearman_correlation(sims, labels)

    # Convert to torch for hypersphere metrics
    t_emb1 = torch.tensor(embeddings1, dtype=torch.float32)
    t_emb2 = torch.tensor(embeddings2, dtype=torch.float32)
    all_emb = torch.cat([t_emb1, t_emb2], dim=0)

    # For alignment, pairs with high human similarity (e.g. score >= 4.0 out of 5.0) or all paired sentences
    high_sim_mask = [i for i, s in enumerate(labels) if s >= 4.0]
    if len(high_sim_mask) > 0:
        align_score = compute_alignment(t_emb1[high_sim_mask], t_emb2[high_sim_mask])
    else:
        align_score = compute_alignment(t_emb1, t_emb2)

    unif_score = compute_uniformity(all_emb)

    return {
        "spearman": round(spearman, 2),
        "alignment": round(align_score, 4),
        "uniformity": round(unif_score, 4)
    }
