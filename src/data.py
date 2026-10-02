"""Data loading and preprocessing utilities for SimCSE.

Handles:
- SNLI 100k subset.
- Unsupervised SimCSE sentence extraction.
- Supervised SimCSE entailment/hard-negative extraction.
- STS-B benchmark loading.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_snli_raw(file_path: str) -> List[Dict]:
    """Load SNLI JSONL records from disk.

    The function attempts several possible locations so the dataset can
    be stored either in the repository root or inside the data/ folder.
    """
    candidate_paths = [
        file_path,
        os.path.join("data", os.path.basename(file_path)),
        os.path.basename(file_path),
        os.path.join("data", "snli_train_100k.jsonl"),
        "snli_train_100k.jsonl",
    ]

    resolved_path = None

    for path in candidate_paths:
        if os.path.exists(path):
            resolved_path = path
            break

    if resolved_path is None:
        raise FileNotFoundError(
            f"SNLI dataset file not found. Tried paths: {candidate_paths}. "
            "Please ensure 'snli_train_100k.jsonl' is located in the "
            "repository root or inside the 'data/' directory."
        )

    records = []

    with open(resolved_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def build_unsupervised_dataset(file_path: str) -> List[str]:
    """Extract unique SNLI sentences for unsupervised SimCSE.

    Uniqueness is calculated using the exact sentence strings contained
    in the provided SNLI dataset.

    Leading/trailing whitespace is preserved when storing sentences
    because the assignment reference count of 165,529 unique sentences
    corresponds to the raw strings in the provided file.

    Whitespace-only strings are ignored.

    Expected count:
        165,529 unique sentences.
    """
    records = load_snli_raw(file_path)

    unique_sentences = set()

    for record in records:
        premise = record.get("premise", "")
        hypothesis = record.get("hypothesis", "")

        if isinstance(premise, str) and premise.strip():
            unique_sentences.add(premise)

        if isinstance(hypothesis, str) and hypothesis.strip():
            unique_sentences.add(hypothesis)

    return sorted(unique_sentences)


def build_supervised_dataset(
    file_path: str,
) -> List[Dict[str, Optional[str]]]:
    """Build supervised SimCSE examples from the SNLI subset.

    SNLI labels:
        0 = entailment
        1 = neutral
        2 = contradiction

    Each supervised example contains:
        premise
        positive = entailment hypothesis
        negative = contradiction hypothesis when available

    The contradiction is used as a hard negative.

    Expected values from the provided dataset:
        33,351 supervised entailment pairs
        9,488 pairs with hard negatives
        ~28.45% with hard negatives
    """
    records = load_snli_raw(file_path)

    premise_map: Dict[str, Dict[int, List[str]]] = {}

    for record in records:
        premise = record.get("premise", "")
        hypothesis = record.get("hypothesis", "")
        label = record.get("label", -1)

        if not isinstance(premise, str):
            continue

        if not isinstance(hypothesis, str):
            continue

        premise = premise.strip()
        hypothesis = hypothesis.strip()

        if not premise or not hypothesis:
            continue

        if label not in (0, 1, 2):
            continue

        if premise not in premise_map:
            premise_map[premise] = {
                0: [],
                1: [],
                2: [],
            }

        premise_map[premise][label].append(hypothesis)

    triplets = []

    for premise, hypotheses in premise_map.items():
        entailments = hypotheses[0]
        contradictions = hypotheses[2]

        for entailment in entailments:
            hard_negative = (
                contradictions[0]
                if contradictions
                else None
            )

            triplets.append(
                {
                    "premise": premise,
                    "positive": entailment,
                    "negative": hard_negative,
                }
            )

    return triplets


class UnsupervisedSimCSEDataset(Dataset):
    """PyTorch dataset for unsupervised SimCSE."""

    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class SupervisedSimCSEDataset(Dataset):
    """PyTorch dataset for supervised SimCSE."""

    def __init__(
        self,
        triplets: List[Dict[str, Optional[str]]],
        use_hard_negatives: bool = True,
    ):
        self.triplets = triplets
        self.use_hard_negatives = use_hard_negatives

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Optional[str]]:
        item = self.triplets[idx]

        if not self.use_hard_negatives:
            return {
                "premise": item["premise"],
                "positive": item["positive"],
                "negative": None,
            }

        return item


class UnsupDataCollator:
    """Tokenize batches for unsupervised SimCSE.

    The model receives a single batch of sentences and performs two
    forward passes so independent dropout masks create the two
    contrastive views.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        batch: List[str],
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class SupDataCollator:
    """Tokenize premise, positive and hard-negative sentences."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 64,
        use_hard_negatives: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_hard_negatives = use_hard_negatives

    def __call__(
        self,
        batch: List[Dict[str, Optional[str]]],
    ) -> Dict:
        premises = [
            item["premise"]
            for item in batch
        ]

        positives = [
            item["positive"]
            for item in batch
        ]

        tokenized_premises = self.tokenizer(
            premises,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokenized_positives = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        has_negatives = (
            self.use_hard_negatives
            and any(
                item["negative"] is not None
                for item in batch
            )
        )

        tokenized_negatives = None

        if has_negatives:
            negatives = [
                (
                    item["negative"]
                    if item["negative"] is not None
                    else item["positive"]
                )
                for item in batch
            ]

            tokenized_negatives = self.tokenizer(
                negatives,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        return {
            "premises": tokenized_premises,
            "positives": tokenized_positives,
            "negatives": tokenized_negatives,
            "has_negatives": has_negatives,
        }


def load_stsb_dataset() -> Tuple[List[Dict], List[Dict]]:
    """Load STS-B development and test datasets.

    The project uses the original STS-B human-score scale from 0 to 5.

    Some Hugging Face versions of sentence-transformers/stsb expose
    scores normalized to the interval [0, 1], while GLUE STS-B uses
    the original [0, 5] scale.

    This function standardizes both sources to the [0, 5] scale.

    Expected sizes:
        Dev:  1,500 sentence pairs
        Test: 1,379 sentence pairs

    Returns:
        Tuple containing:
            (dev_records, test_records)
    """

    def ensure_zero_to_five_scale(
        records: List[Dict],
    ) -> List[Dict]:
        """Convert normalized STS-B scores from [0, 1] to [0, 5]."""

        if not records:
            return records

        scores = [
            record["score"]
            for record in records
        ]

        min_score = min(scores)
        max_score = max(scores)

        if min_score >= 0.0 and max_score <= 1.0:
            for record in records:
                record["score"] = record["score"] * 5.0

        return records

    try:
        dataset = load_dataset(
            "sentence-transformers/stsb"
        )

        dev_records = [
            {
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "score": float(item["score"]),
            }
            for item in dataset["validation"]
        ]

        test_records = [
            {
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "score": float(item["score"]),
            }
            for item in dataset["test"]
        ]

    except Exception:
        dataset = load_dataset(
            "glue",
            "stsb",
        )

        dev_records = [
            {
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "score": float(item["label"]),
            }
            for item in dataset["validation"]
        ]

        test_records = [
            {
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "score": float(item["label"]),
            }
            for item in dataset["test"]
        ]

    dev_records = ensure_zero_to_five_scale(
        dev_records
    )

    test_records = ensure_zero_to_five_scale(
        test_records
    )

    return dev_records, test_records
