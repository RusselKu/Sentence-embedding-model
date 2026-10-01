"""Data loading and preprocessing utilities for SimCSE.
Handles SNLI 100k subset and STS-B benchmark dataset.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


def load_snli_raw(file_path: str) -> List[Dict]:
    """Load SNLI JSONL records from disk, resolving path in data/ or root if needed."""
    candidate_paths = [
        file_path,
        os.path.join("data", os.path.basename(file_path)),
        os.path.basename(file_path),
        os.path.join("data", "snli_train_100k.jsonl"),
        "snli_train_100k.jsonl",
    ]
    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        raise FileNotFoundError(
            f"SNLI dataset file not found. Tried paths: {candidate_paths}. "
            "Please ensure 'snli_train_100k.jsonl' is located in the root or 'data/' directory."
        )

    records = []
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_unsupervised_dataset(file_path: str) -> List[str]:
    """Extract all unique sentences from SNLI records for unsupervised SimCSE.
    
    Expected count from 100k subset: ~165,529 unique sentences.
    """
    records = load_snli_raw(file_path)
    unique_sentences = set()
    for rec in records:
        premise = rec.get("premise", "").strip()
        hypothesis = rec.get("hypothesis", "").strip()
        if premise:
            unique_sentences.add(premise)
        if hypothesis:
            unique_sentences.add(hypothesis)
    sentence_list = sorted(list(unique_sentences))
    return sentence_list


def build_supervised_dataset(file_path: str) -> List[Dict[str, Optional[str]]]:
    """Extract (premise, entailment, optional contradiction) triplets for supervised SimCSE.
    
    Label mapping in SNLI subset:
    0 = entailment
    1 = neutral
    2 = contradiction
    
    Returns a list of dicts: {'premise': str, 'positive': str, 'negative': Optional[str]}
    """
    records = load_snli_raw(file_path)
    # Group hypotheses by premise and label
    premise_map: Dict[str, Dict[int, List[str]]] = {}
    for rec in records:
        premise = rec.get("premise", "").strip()
        hypothesis = rec.get("hypothesis", "").strip()
        label = rec.get("label", -1)
        if not premise or not hypothesis or label not in (0, 1, 2):
            continue
        if premise not in premise_map:
            premise_map[premise] = {0: [], 1: [], 2: []}
        premise_map[premise][label].append(hypothesis)

    triplets = []
    for premise, hyps in premise_map.items():
        entailments = hyps[0]
        contradictions = hyps[2]
        for ent in entailments:
            # Pair premise with entailment
            neg = contradictions[0] if contradictions else None
            triplets.append({
                "premise": premise,
                "positive": ent,
                "negative": neg
            })
    return triplets


class UnsupervisedSimCSEDataset(Dataset):
    """Dataset for Unsupervised SimCSE (single sentences)."""

    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class SupervisedSimCSEDataset(Dataset):
    """Dataset for Supervised SimCSE (premise, positive/entailment, optional negative/contradiction)."""

    def __init__(self, triplets: List[Dict[str, Optional[str]]], use_hard_negatives: bool = True):
        self.triplets = triplets
        self.use_hard_negatives = use_hard_negatives

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, Optional[str]]:
        item = self.triplets[idx]
        if not self.use_hard_negatives:
            return {
                "premise": item["premise"],
                "positive": item["positive"],
                "negative": None
            }
        return item


class UnsupDataCollator:
    """Collator for Unsupervised SimCSE.
    Produces tokenized batches. For unsupervised training, the model does two forward passes
    or duplicates batch items to apply independent dropout masks.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


class SupDataCollator:
    """Collator for Supervised SimCSE.
    Batches premise, positive, and negative sentences into tokenized inputs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 64, use_hard_negatives: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_hard_negatives = use_hard_negatives

    def __call__(self, batch: List[Dict[str, Optional[str]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        premises = [b["premise"] for b in batch]
        positives = [b["positive"] for b in batch]

        tokenized_premises = self.tokenizer(
            premises, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        tokenized_positives = self.tokenizer(
            positives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        has_negatives = self.use_hard_negatives and any(b["negative"] is not None for b in batch)
        tokenized_negatives = None
        if has_negatives:
            # Fallback to positive or premise if negative is missing for a specific instance
            negatives = [b["negative"] if b["negative"] is not None else b["positive"] for b in batch]
            tokenized_negatives = self.tokenizer(
                negatives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )

        return {
            "premises": tokenized_premises,
            "positives": tokenized_positives,
            "negatives": tokenized_negatives,
            "has_negatives": has_negatives
        }


def load_stsb_dataset() -> Tuple[List[Dict], List[Dict]]:
    """Load STS-B dev (1500 pairs) and test (1379 pairs) splits.
    
    Returns:
        (dev_records, test_records)
    """
    try:
        ds = load_dataset("sentence-transformers/stsb")
        dev_records = [
            {"sentence1": x["sentence1"], "sentence2": x["sentence2"], "score": float(x["score"])}
            for x in ds["validation"]
        ]
        test_records = [
            {"sentence1": x["sentence1"], "sentence2": x["sentence2"], "score": float(x["score"])}
            for x in ds["test"]
        ]
        return dev_records, test_records
    except Exception as e:
        # Fallback to GLUE STS-B if sentence-transformers/stsb has network issues
        ds = load_dataset("glue", "stsb")
        dev_records = [
            {"sentence1": x["sentence1"], "sentence2": x["sentence2"], "score": float(x["label"])}
            for x in ds["validation"]
        ]
        test_records = [
            {"sentence1": x["sentence1"], "sentence2": x["sentence2"], "score": float(x["label"])}
            for x in ds["test"]
        ]
        return dev_records, test_records
