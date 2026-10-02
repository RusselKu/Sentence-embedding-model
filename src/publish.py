"""Export SimCSE model to SentenceTransformers format, publish to Hugging Face Hub,
and perform post-upload verification against STS-B test split.
"""

import argparse
import os
import sys
import json
from typing import Optional, Dict, List
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from src.data import load_stsb_dataset
from src.metrics import evaluate_sts_benchmark
from src.models import SimCSEModel


def generate_model_card(
    repo_id: str,
    base_model: str,
    mode: str,
    dev_spearman: float,
    test_spearman: float,
    alignment: float,
    uniformity: float,
    hyperparameters: dict,
) -> str:
    """Generate Hugging Face Model Card README.md."""
    card = f"""---
language:
- en
license: apache-2.0
library_name: sentence-transformers
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- simcse
- contrastive-learning
pipeline_tag: sentence-similarity
metrics:
- spearman_cosine
---

# {repo_id}

This is a **SimCSE ({mode.capitalize()})** sentence embedding model trained on a 100k subset of SNLI from `{base_model}`. It maps sentences into a dense 768-dimensional vector space for semantic similarity, clustering, and retrieval tasks.

## Model Details
- **Architecture**: `{base_model}` with {mode} contrastive learning objective (InfoNCE).
- **Pooling**: CLS / Mean pooling on token embeddings.
- **Training Dataset**: SNLI (100k subset, `snli_train_100k.jsonl`).
- **Evaluation Dataset**: STS-Benchmark (STS-B).

## Evaluation Results (STS-B)
| Metric | Dev Split | Test Split |
| :--- | :--- | :--- |
| **Spearman Correlation ($\times 100$)** | **{dev_spearman:.2f}** | **{test_spearman:.2f}** |
| **Alignment ($\alpha=2$)** | - | {alignment:.4f} |
| **Uniformity ($t=2$)** | - | {uniformity:.4f} |

## Training Recipe & Hyperparameters
```json
{json.dumps(hyperparameters, indent=2)}
```

## Usage (Sentence-Transformers)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model from Hugging Face Hub
model = SentenceTransformer("{repo_id}")

sentences = [
    "A man is playing a guitar in the room.",
    "A musician is playing music indoors.",
    "A dog is chasing a ball in the park."
]

embeddings = model.encode(sentences, normalize_embeddings=True)
sims = cosine_similarity(embeddings)
print("Similarity matrix:", sims)
```

## Usage (HuggingFace Transformers)

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModel.from_pretrained("{repo_id}")

sentences = ["Antigravity AI is writing code.", "AI agents are generating software."]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # Mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    embeddings = F.normalize(embeddings, p=2, dim=1)

cos_sim = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
print("Cosine similarity:", cos_sim.item())
```

## Limitations & Biases
- Trained exclusively on English data.
- Evaluated primarily on sentence similarity (STS-B); may require domain adaptation for specialized retrieval (e.g. legal, biomedical).
"""
    return card


def export_to_sentence_transformers(
    checkpoint_dir: str,
    export_dir: str,
    base_model_name: str = "bert-base-uncased",
    pooling_mode: str = "cls",
) -> SentenceTransformer:
    """Convert SimCSE checkpoint into a valid SentenceTransformers model folder."""
    os.makedirs(export_dir, exist_ok=True)
    print(f"Converting SimCSE checkpoint from {checkpoint_dir} -> {export_dir}")

    # Load encoder weights
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")) else base_model_name)
    model = SimCSEModel(base_model_name, pooling=pooling_mode)
    ckpt_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(ckpt_file):
        model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
    
    # Save base transformer
    transformer_subfolder = os.path.join(export_dir, "transformer")
    model.encoder.save_pretrained(transformer_subfolder)
    tokenizer.save_pretrained(transformer_subfolder)

    # Build SentenceTransformers modules
    word_embedding_model = models.Transformer(transformer_subfolder, max_seq_length=64)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=(pooling_mode == "cls"),
        pooling_mode_mean_tokens=(pooling_mode == "mean"),
        pooling_mode_max_tokens=False,
    )
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    st_model.save(export_dir)
    print(f"SentenceTransformers export saved to: {export_dir}")
    return st_model


def publish_and_verify(
    checkpoint_dir: str,
    repo_id: str,
    run_json_path: Optional[str] = None,
    pooling_mode: str = "cls",
    hf_token: Optional[str] = None,
):
    """Export, push to Hugging Face Hub, and verify by reloading and scoring test set."""
    export_dir = "exported_model"
    st_model = export_to_sentence_transformers(checkpoint_dir, export_dir, pooling_mode=pooling_mode)

    # Load run metadata if available
    run_meta = {}
    if run_json_path and os.path.exists(run_json_path):
        with open(run_json_path, "r", encoding="utf-8") as f:
            run_meta = json.load(f)

    mode = run_meta.get("mode", "supervised")
    base_model = run_meta.get("model_name_or_path", "bert-base-uncased")
    dev_res = run_meta.get("results", {}).get("dev", {"spearman": 0.0, "alignment": 0.0, "uniformity": 0.0})
    test_res = run_meta.get("results", {}).get("test", {"spearman": 0.0, "alignment": 0.0, "uniformity": 0.0})
    hyperparams = run_meta.get("hyperparameters", {})

    # Generate Model Card
    readme_content = generate_model_card(
        repo_id=repo_id,
        base_model=base_model,
        mode=mode,
        dev_spearman=dev_res.get("spearman", 0.0),
        test_spearman=test_res.get("spearman", 0.0),
        alignment=test_res.get("alignment", 0.0),
        uniformity=test_res.get("uniformity", 0.0),
        hyperparameters=hyperparams,
    )

    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"\n--- Uploading Model to Hugging Face Hub: {repo_id} ---")
    st_model.push_to_hub(
        repo_id=repo_id,
        token=hf_token or os.environ.get("HF_TOKEN"),
        private=False,
    )
    print("Upload completed successfully!")

    # Reload and Verify
    print(f"\n--- Verification Step: Reloading from Hugging Face Hub ({repo_id}) ---")
    reloaded_model = SentenceTransformer(repo_id)
    _, test_records = load_stsb_dataset()

    sent1 = [r["sentence1"] for r in test_records]
    sent2 = [r["sentence2"] for r in test_records]
    scores = [r["score"] for r in test_records]

    emb1 = reloaded_model.encode(sent1, batch_size=64, normalize_embeddings=True)
    emb2 = reloaded_model.encode(sent2, batch_size=64, normalize_embeddings=True)

    verified_metrics = evaluate_sts_benchmark(emb1, emb2, scores)
    print("=" * 50)
    print("VERIFICATION BENCHMARK ON STS-B TEST")
    print(f"Reloaded Test Spearman (x100): {verified_metrics['spearman']:.2f}")
    print(f"Expected Test Spearman       : {test_res.get('spearman', 'N/A')}")
    print(f"Alignment: {verified_metrics['alignment']:.4f} | Uniformity: {verified_metrics['uniformity']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and Publish SimCSE to Hugging Face Hub")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo id, e.g., 'username/simcse-bert-snli'")
    parser.add_argument("--run_json", type=str, default=None, help="Path to run JSON file with metrics")
    parser.add_argument("--pooling", type=str, default="cls", help="Pooling mode (cls or mean)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (optional if logged in)")
    args = parser.parse_args()

    publish_and_verify(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        run_json_path=args.run_json,
        pooling_mode=args.pooling,
        hf_token=args.token,
    )
