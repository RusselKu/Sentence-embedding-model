"""Offline regression tests with a tiny BERT; no training data downloads."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import BertConfig

from src.data import SupDataCollator
from src.models import SimCSEModel
from src.train import train
from src.evaluate import run_evaluation


class TokenizerStub:
    def __init__(self):
        self.calls = []

    def __call__(self, sentences, **kwargs):
        self.calls.append(sentences)
        ids = torch.tensor([[1, 2 + len(s) % 20, 3] for s in sentences])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def save_pretrained(self, directory):
        Path(directory, "tokenizer_stub.json").write_text("{}", encoding="utf-8")


def tiny_model(directory, **kwargs):
    kwargs.pop("model_name_or_path", None)
    BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
               num_attention_heads=2, intermediate_size=32,
               hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0).save_pretrained(directory)
    return SimCSEModel(str(directory), config_only=True, **kwargs)


class SupervisedPipelineTests(unittest.TestCase):
    def test_partial_negatives_never_substitute_positives(self):
        tokenizer = TokenizerStub()
        batch = SupDataCollator(tokenizer)([
            {"premise": "anchor1", "positive": "positive1", "negative": None},
            {"premise": "anchor2", "positive": "positive2", "negative": "contradiction"},
        ])
        self.assertEqual(tokenizer.calls[-1], ["contradiction"])
        self.assertEqual(batch["negatives"]["input_ids"].shape[0], 1)

    def test_no_available_negatives_and_ablation(self):
        example = {"premise": "anchor", "positive": "positive", "negative": None}
        self.assertIsNone(SupDataCollator(TokenizerStub())([example])["negatives"])
        example["negative"] = "contradiction"
        self.assertIsNone(SupDataCollator(TokenizerStub(), use_hard_negatives=False)([example])["negatives"])

    def test_variable_negative_count_loss_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model(directory, inference_mlp=True)
            tokenizer = TokenizerStub()
            premises = tokenizer(["anchor1", "anchor long"])
            positives = tokenizer(["positive", "another positive"])
            negatives = tokenizer(["negative example"])
            loss, logits = model.forward_supervised(premises, positives, negatives)
            self.assertEqual(tuple(logits.shape), (2, 3))
            z1 = F.normalize(model.encode(**premises, apply_mlp=True), dim=-1)
            zp = F.normalize(model.encode(**positives, apply_mlp=True), dim=-1)
            zn = F.normalize(model.encode(**negatives, apply_mlp=True), dim=-1)
            expected = torch.cat([z1 @ zp.T, z1 @ zn.T], dim=1) / model.temperature
            torch.testing.assert_close(logits, expected)
            torch.testing.assert_close(loss, F.cross_entropy(expected, torch.arange(2)))
            loss.backward()
            self.assertTrue(torch.isfinite(model.mlp.dense.weight.grad).all())
            self.assertGreater(model.mlp.dense.weight.grad.abs().sum().item(), 0)
            _, off_logits = model.forward_supervised(premises, positives, negatives, use_hard_negatives=False)
            self.assertEqual(tuple(off_logits.shape), (2, 2))

    def test_checkpoint_roundtrip_keeps_supervised_mlp(self):
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model(directory, inference_mlp=True)
            inputs = TokenizerStub()(["roundtrip"])
            before = model.get_sentence_embeddings(**inputs)
            model.save_checkpoint(directory, TokenizerStub())
            restored = SimCSEModel.from_checkpoint(directory)
            self.assertTrue(restored.inference_mlp)
            torch.testing.assert_close(before, restored.get_sentence_embeddings(**inputs), rtol=0, atol=0)
            expected = F.normalize(restored.encode(**inputs, apply_mlp=True), dim=-1)
            torch.testing.assert_close(before, expected)

    def test_training_does_not_evaluate_test_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            tokenizer = TokenizerStub()
            dev, test = [{"split": "dev"}], [{"split": "test"}]
            examples = [{"premise": "anchor", "positive": "positive", "negative": None},
                        {"premise": "long anchor", "positive": "entailment", "negative": "negative"}]
            args = argparse.Namespace(seed=42, mode="sup", run_name="offline_test",
                output_dir=str(root / "checkpoints"), runs_dir=str(root / "runs"),
                model_name_or_path=str(base), pooling="cls", temperature=0.05,
                dropout_rate=None, data_path="unused", no_hard_negatives=False,
                max_length=64, batch_size=2, lr=5e-5, weight_decay=0., epochs=1,
                warmup_ratio=0.05, eval_steps=0, same_dropout_mask=False)
            metrics = {"spearman": 50., "alignment": 0.5, "uniformity": -2.}
            with patch("src.train.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch("src.train.SimCSEModel", side_effect=lambda *a, **k: tiny_model(base, **k)), \
                 patch("src.train.load_stsb_dataset", return_value=(dev, test)), \
                 patch("src.train.build_supervised_dataset", return_value=examples), \
                 patch("src.train.evaluate_on_stsb", return_value=metrics) as evaluate:
                result = train(args)
            self.assertIsNone(result["results"]["test"])
            self.assertTrue(all(call.args[2] is dev for call in evaluate.call_args_list))
            self.assertTrue(Path(result["checkpoint_path"], "config.json").exists())
            recorded = json.loads((root / "runs" / "offline_test.json").read_text())
            self.assertTrue(recorded["hyperparameters"]["inference_mlp"])

    def test_evaluation_dev_only_never_scores_test(self):
        dev, test = [{"split": "dev"}], [{"split": "test"}]
        with patch("src.evaluate.load_stsb_dataset", return_value=(dev, test)), \
             patch("src.evaluate.SentenceTransformer"), \
             patch("src.evaluate.evaluate_split", return_value={"spearman": 50., "alignment": .5, "uniformity": -2.}) as evaluate:
            result = run_evaluation("unused", model_type="sbert", device="cpu", split="dev")
        self.assertEqual(evaluate.call_count, 1)
        self.assertIs(evaluate.call_args.args[0], dev)
        self.assertIsNone(result["test"])

    def test_evaluation_test_only_never_scores_dev(self):
        dev, test = [{"split": "dev"}], [{"split": "test"}]
        with patch("src.evaluate.load_stsb_dataset", return_value=(dev, test)), \
             patch("src.evaluate.SentenceTransformer"), \
             patch("src.evaluate.evaluate_split", return_value={"spearman": 50., "alignment": .5, "uniformity": -2.}) as evaluate:
            result = run_evaluation("unused", model_type="sbert", device="cpu", split="test")
        self.assertEqual(evaluate.call_count, 1)
        self.assertIs(evaluate.call_args.args[0], test)
        self.assertIsNone(result["dev"])


if __name__ == "__main__":
    unittest.main()
