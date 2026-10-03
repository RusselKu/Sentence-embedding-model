"""Alignment-uniformity plot (Wang & Isola, 2020; SimCSE Fig. 2 style).

Uses only logged values computed with the CURRENT metric definition:
alignment on STS-B pairs with human score >= 4 (on the 0-5 scale), uniformity
on all sentences of the split. Lower is better on both axes.

Sources (all already in the repo, nothing is re-run):
- raw BERT / SBERT / unsup: evaluation_and_analysis.ipynb (Rivaldo), runs/eval_hub_unsup.json
- supervised ON / OFF:      reports/jonav/supervised_benchmark.csv
The same-dropout-mask ablation is NOT plotted: its logged alignment was computed
with the old (all-pairs) definition and its checkpoint is needed to recompute it.

Usage:  python scripts/plot_alignment_uniformity.py
"""

import os

import matplotlib.pyplot as plt

# (alignment, uniformity, STS-B Spearman) per split
MODELS = {
    "Raw BERT (mean)": {"dev": (0.1948, -1.6348, 59.31), "test": (0.2155, -1.6186, 47.29)},
    "SBERT-2019": {"dev": (0.1929, -3.0557, 80.77), "test": (0.1798, -3.0493, 76.98)},
    "Unsup SimCSE (ours)": {"dev": (0.3150, -2.8887, 76.32), "test": (0.3501, -2.8900, 67.32)},
    "Sup SimCSE, hard neg ON (ours)": {"dev": (0.1857, -3.1374, 81.67), "test": (0.1903, -3.0460, 79.02)},
    "Sup SimCSE, hard neg OFF (ours)": {"dev": (0.1799, -3.0665, 81.01), "test": (0.1914, -2.9656, 77.11)},
}
MARKERS = ["o", "s", "^", "D", "v"]
# label offsets (points) so the clustered supervised/SBERT labels do not overlap
OFFSETS = {"dev": [(8, 6), (-8, -20), (8, 6), (10, -14), (-38, 2)],
           "test": [(8, 6), (-38, 10), (8, 6), (10, -14), (-38, -4)]}


def main(output="reports/figures/alignment_uniformity.png"):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, split in zip(axes, ["dev", "test"]):
        for (name, vals), marker, off in zip(MODELS.items(), MARKERS, OFFSETS[split]):
            align, unif, rho = vals[split]
            ax.scatter(unif, align, s=110, marker=marker, label=name, zorder=3)
            ax.annotate(f"{rho:.1f}", (unif, align), textcoords="offset points",
                        xytext=off, fontsize=9)
        ax.set_title(f"STS-B {split} (labels = Spearman x100)")
        ax.set_xlabel("Uniformity  (lower = more uniform)")
        ax.grid(alpha=0.3)
        ax.invert_xaxis()   # better (more negative) uniformity to the right
    axes[0].set_ylabel("Alignment, pairs with score >= 4  (lower = better)")
    axes[0].invert_yaxis()  # better alignment upward
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Alignment vs. uniformity (better = up and to the right)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
