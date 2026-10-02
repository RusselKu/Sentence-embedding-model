"""Evaluation harness for Sentence Embeddings on STS-B benchmark.
Computes Spearman correlation, Alignment, and Uniformity.
Supports raw BERT, SBERT-2019 baseline, trained SimCSE checkpoints, and HF Hub models.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from src.data import load_stsb_dataset
from src.metrics import evaluate_sts_benchmark, compute_alignment, compute_uniformity
from src.models import SimCSEModel


def encode_sentences_hf(
    sentences: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch_size: int = 64,
    device: str = "cpu",
    pooling: str = "cls",
) -> np.ndarray:
    """Encode list of sentences using PyTorch / Transformers model."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch_text = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch_text, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            if isinstance(model, SimCSEModel):
                emb = model.get_sentence_embeddings(
                    inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids"), normalize=True
                )
            else:
                outputs = model(**inputs)
                if pooling == "cls":
                    emb = outputs.last_hidden_state[:, 0]
                elif pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    emb = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                else:
                    emb = outputs.last_hidden_state[:, 0]
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def encode_sentences_st(sentences: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Encode list of sentences using SentenceTransformers."""
    embeddings = model.encode(sentences, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return embeddings


def evaluate_split(
    records: List[Dict],
    model_or_fn,
    is_st: bool = False,
    tokenizer: Optional[AutoTokenizer] = None,
    device: str = "cpu",
    pooling: str = "cls",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate sentence pairs on STS-B."""
    sent1 = [r["sentence1"] for r in records]
    sent2 = [r["sentence2"] for r in records]
    scores = [r["score"] for r in records]

    if is_st:
        emb1 = encode_sentences_st(sent1, model_or_fn, batch_size=batch_size)
        emb2 = encode_sentences_st(sent2, model_or_fn, batch_size=batch_size)
    else:
        emb1 = encode_sentences_hf(sent1, model_or_fn, tokenizer, batch_size=batch_size, device=device, pooling=pooling)
        emb2 = encode_sentences_hf(sent2, model_or_fn, tokenizer, batch_size=batch_size, device=device, pooling=pooling)

    return evaluate_sts_benchmark(emb1, emb2, scores)


def run_evaluation(
    model_path_or_name: str,
    model_type: str = "simcse",  # "simcse", "raw_bert", "sbert", or "st_hub"
    pooling: str = "cls",
    device: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict:
    """Run full STS-B evaluation across dev and test splits."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- Loading STS-B Dataset ---")
    dev_records, test_records = load_stsb_dataset()
    print(f"Loaded {len(dev_records)} dev pairs and {len(test_records)} test pairs.")

    print(f"Evaluating Model: {model_path_or_name} (Type: {model_type}, Pooling: {pooling}, Device: {device})")

    if model_type == "sbert":
        st_model = SentenceTransformer(model_path_or_name, device=device)
        dev_res = evaluate_split(dev_records, st_model, is_st=True)
        test_res = evaluate_split(test_records, st_model, is_st=True)
    elif model_type == "raw_bert":
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        model = AutoModel.from_pretrained(model_path_or_name).to(device)
        dev_res = evaluate_split(dev_records, model, is_st=False, tokenizer=tokenizer, device=device, pooling=pooling)
        test_res = evaluate_split(test_records, model, is_st=False, tokenizer=tokenizer, device=device, pooling=pooling)
    elif model_type == "st_hub":
        st_model = SentenceTransformer(model_path_or_name, device=device)
        dev_res = evaluate_split(dev_records, st_model, is_st=True)
        test_res = evaluate_split(test_records, st_model, is_st=True)
    else:  # simcse checkpoint or dir
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        model = SimCSEModel(model_path_or_name, pooling=pooling).to(device)
        # If checkpoint has state_dict
        ckpt_file = os.path.join(model_path_or_name, "pytorch_model.bin")
        if os.path.exists(ckpt_file):
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
        dev_res = evaluate_split(dev_records, model, is_st=False, tokenizer=tokenizer, device=device, pooling=pooling)
        test_res = evaluate_split(test_records, model, is_st=False, tokenizer=tokenizer, device=device, pooling=pooling)

    results = {
        "model": model_path_or_name,
        "model_type": model_type,
        "pooling": pooling,
        "device": device,
        "dev": dev_res,
        "test": test_res,
    }

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS FOR {model_path_or_name}")
    print("=" * 50)
    print(f"Dev  Spearman (x100): {dev_res['spearman']:.2f} | Alignment: {dev_res['alignment']:.4f} | Uniformity: {dev_res['uniformity']:.4f}")
    print(f"Test Spearman (x100): {test_res['spearman']:.2f} | Alignment: {test_res['alignment']:.4f} | Uniformity: {test_res['uniformity']:.4f}")
    print("=" * 50 + "\n")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentence embedding models on STS-B")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased", help="Model path or HuggingFace ID")
    parser.add_argument("--model_type", type=str, choices=["simcse", "raw_bert", "sbert", "st_hub"], default="raw_bert")
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="mean")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save evaluation output JSON")
    args = parser.parse_args()

    run_evaluation(
        model_path_or_name=args.model_path,
        model_type=args.model_type,
        pooling=args.pooling,
        output_json=args.output_json,
    )
