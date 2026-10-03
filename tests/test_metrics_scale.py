"""QA regression tests (Bianca, CPU): STS-B score scale and the alignment subset.

Bug found during QA: early runs loaded STS-B scores on the [0, 1] scale, so the
"score >= 4" positive-pair mask in evaluate_sts_benchmark was empty and alignment
silently fell back to ALL pairs. Spearman and uniformity were unaffected, but the
alignment values were not comparable with later runs (e.g. raw BERT dev 0.3678
vs. 0.1948 for the same model). These tests pin the intended behaviour.
"""
import unittest
from unittest.mock import patch

import numpy as np

from src.data import load_stsb_dataset
from src.metrics import compute_alignment, evaluate_sts_benchmark
import torch


class _FakeSplit(list):
    pass


def _fake_dataset(scores):
    rows = [{"sentence1": f"a{i}", "sentence2": f"b{i}", "score": s} for i, s in enumerate(scores)]
    return {"validation": _FakeSplit(rows), "test": _FakeSplit(rows)}


class STSBScaleTests(unittest.TestCase):
    def test_unit_interval_scores_are_rescaled_to_zero_five(self):
        with patch("src.data.load_dataset", return_value=_fake_dataset([0.0, 0.5, 0.84, 1.0])):
            dev, test = load_stsb_dataset()
        self.assertEqual([r["score"] for r in dev], [0.0, 2.5, 4.2, 5.0])
        self.assertEqual(max(r["score"] for r in test), 5.0)

    def test_zero_five_scores_are_left_untouched(self):
        with patch("src.data.load_dataset", return_value=_fake_dataset([0.0, 2.4, 5.0])):
            dev, _ = load_stsb_dataset()
        self.assertEqual([r["score"] for r in dev], [0.0, 2.4, 5.0])


class AlignmentSubsetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.e1 = rng.normal(size=(6, 8)).astype(np.float32)
        self.e2 = rng.normal(size=(6, 8)).astype(np.float32)
        self.e2[:2] = self.e1[:2] + 0.01  # the two high-score pairs are near-identical

    def test_alignment_uses_only_pairs_scored_at_least_four(self):
        labels = [4.5, 4.0, 1.0, 0.5, 2.0, 3.9]
        result = evaluate_sts_benchmark(self.e1, self.e2, labels)
        expected = compute_alignment(torch.tensor(self.e1[:2]), torch.tensor(self.e2[:2]))
        self.assertAlmostEqual(result["alignment"], round(expected, 4), places=4)

    def test_unit_scale_labels_would_fall_back_to_all_pairs(self):
        """Documents the old bug: [0,1] labels make the >=4 mask empty."""
        labels_unit = [0.9, 0.8, 0.2, 0.1, 0.4, 0.78]
        labels_five = [s * 5 for s in labels_unit]
        wrong = evaluate_sts_benchmark(self.e1, self.e2, labels_unit)["alignment"]
        right = evaluate_sts_benchmark(self.e1, self.e2, labels_five)["alignment"]
        self.assertGreater(wrong, right)  # all-pairs alignment looks much worse


if __name__ == "__main__":
    unittest.main()
