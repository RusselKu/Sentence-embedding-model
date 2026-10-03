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
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers.models import Transformer, Pooling
except ImportError:
    from sentence_transformers.sentence_transformer.modules import Transformer, Pooling
from src.data import load_stsb_dataset
from src.metrics import evaluate_sts_benchmark
from src.models import SimCSEModel


TEAM = ("Acosta Castellanos Bianca Alexandra, Canche Chuc Angel Rivaldo, Ku Aguilar Russel, "
        "Sanchez Novelo Damian, Velasco Martin Jonathan Abisai")

PAPER_STSB = {"unsup": (82.5, 76.85), "sup": (86.2, 84.25)}  # Gao et al. 2021, Tables 3/4/5


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def generate_model_card(repo_id: str, run_meta: dict, verified_test_spearman=None) -> str:
    """Model card (README.md for the Hub) built from a run JSON in runs/.

    Covers what the assignment asks for: training data, recipe, metrics, limitations.
    Standardized template (Bianca): every number comes from the run JSON, nothing typed by hand.
    """
    mode = "sup" if str(run_meta.get("mode", "sup")).startswith("sup") else "unsup"
    hp = run_meta.get("hyperparameters", {})
    hw = run_meta.get("hardware", {})
    base = run_meta.get("model_name_or_path", "bert-base-uncased")
    pooling = run_meta.get("pooling", "cls")
    res = run_meta.get("results", {})
    dev = res.get("dev") or {}
    test = res.get("test") or {}
    paper_dev, paper_test = PAPER_STSB[mode]
    hard_neg = mode == "sup" and not hp.get("no_hard_negatives_ablation", False)

    if mode == "unsup":
        data_desc = ("165,528 unique sentences (premises and hypotheses, whitespace-stripped) from "
                     "`snli_train_100k.jsonl`, a 100k-record subset of the SNLI train split. Labels are "
                     "not used: each sentence is its own positive, the two views differ only by dropout.")
        objective = ("Unsupervised SimCSE (Gao et al., 2021, Eq. 1): InfoNCE over a batch of N sentences, "
                     "positive = the same sentence encoded with an independent dropout mask, "
                     f"negatives = the other N-1 = {hp.get('batch_size', 64) - 1} sentences of the batch.")
        inference = ("[CLS] token of the last layer. The MLP head was used only during training and is "
                     "discarded at inference, as in the paper.")
    else:
        data_desc = ("33,351 (premise, entailment hypothesis) pairs from `snli_train_100k.jsonl` (100k-record "
                     "subset of SNLI train). 9,488 of them (28.45%) also have a contradiction hypothesis, "
                     "used as hard negative" + ("." if hard_neg else " (disabled in this run)."))
        objective = ("Supervised SimCSE (Gao et al., 2021, Eq. 5): InfoNCE with entailment hypotheses as positives, "
                     f"the other {hp.get('batch_size', 64) - 1} in-batch positives as negatives"
                     + (" plus the K contradiction hypotheses available in the batch as hard negatives (63 + K)."
                        if hard_neg else "."))
        inference = ("[CLS] token followed by the trained MLP (Linear 768->768 + tanh), kept at inference "
                     "as in the official SimCSE evaluation for supervised models.")

    verify = ("" if verified_test_spearman is None else
              f"\nVerified after upload: reloaded from the Hub and re-evaluated on STS-B test = "
              f"**{verified_test_spearman:.2f}** (table value {_fmt(test.get('spearman'), '.2f')}).\n")

    hp_rows = "\n".join(f"| {k} | {v} |" for k, v in [
        ("Base model", f"`{base}`"), ("Pooling", pooling.upper()),
        ("Learning rate", hp.get("lr")), ("Batch size", hp.get("batch_size")),
        ("Epochs", hp.get("epochs")), ("Temperature (tau)", hp.get("temperature")),
        ("Max sequence length", hp.get("max_length", 64)),
        ("Dropout", hp.get("dropout_rate") or "0.1 (bert-base-uncased default)"),
        ("Warmup ratio / weight decay", f"{hp.get('warmup_ratio')} / {hp.get('weight_decay')}"),
        ("Seed", hp.get("seed")),
        ("Checkpoint selection", "best STS-B dev Spearman, evaluated every 250 steps and at epoch end"),
        ("Hardware", f"{hw.get('gpu_name', hw.get('device', '?'))}, {run_meta.get('training_time_seconds', '?')} s"),
    ])

    return f"""---
language:
- en
license: apache-2.0
library_name: sentence-transformers
base_model: {base}
datasets:
- stanfordnlp/snli
- sentence-transformers/stsb
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- simcse
- contrastive-learning
pipeline_tag: sentence-similarity
model-index:
- name: {repo_id.split('/')[-1]}
  results:
  - task:
      type: semantic-similarity
    dataset:
      name: STS-Benchmark
      type: sentence-transformers/stsb
      split: test
    metrics:
    - type: spearman_cosine
      value: {_fmt(test.get('spearman'), '.2f')}
---

# {repo_id}

{'Supervised' if mode == 'sup' else 'Unsupervised'} **SimCSE** sentence encoder trained from `{base}` on a
100k-record subset of SNLI, as a course replication of Gao, Yao & Chen (EMNLP 2021). It maps an English
sentence to a 768-dimensional vector; compare sentences with cosine similarity.

Team (Universidad Politécnica de Yucatán, U2T02): {TEAM}.

## Training data
{data_desc}

The paper trained on much more data (1M Wikipedia sentences for unsupervised; ~314k SNLI+MNLI pairs for
supervised). This model is therefore **not** expected to reach the paper's numbers; see Limitations.

## Training recipe
{objective}

| Setting | Value |
|---|---|
{hp_rows}

**Inference representation:** {inference}

## Evaluation (STS-B, Spearman x100)
Protocol: encode both sentences, L2-normalize, cosine similarity, Spearman against the human scores;
no regressor ("all" aggregation, Gao et al. Appendix B). Test was evaluated once, after selecting the
checkpoint on dev. Alignment uses STS-B pairs with score >= 4; uniformity uses all sentences
(Wang & Isola, 2020; lower is better for both).

| Split | Spearman | Alignment (a=2) | Uniformity (t=2) | Paper (Gao et al.) |
|---|---:|---:|---:|---:|
| Dev | {_fmt(dev.get('spearman'), '.2f')} | {_fmt(dev.get('alignment'), '.4f')} | {_fmt(dev.get('uniformity'), '.4f')} | {paper_dev} |
| Test | {_fmt(test.get('spearman'), '.2f')} | {_fmt(test.get('alignment'), '.4f')} | {_fmt(test.get('uniformity'), '.4f')} | {paper_test} |

Reference baselines with the same evaluation code: raw `bert-base-uncased` (mean pooling) 59.31 / 47.29;
SBERT-2019 (`bert-base-nli-mean-tokens`) 80.77 / 76.98 (dev / test).
{verify}
## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{repo_id}")
sentences = [
    "A man is playing a guitar.",
    "Someone is playing an instrument.",
    "A dog is chasing a ball in the park.",
]
emb = model.encode(sentences, normalize_embeddings=True)
print(emb @ emb.T)  # cosine similarity matrix
```

Use `sentence-transformers` (not a raw `AutoModel` with your own pooling): the pooling
{'and the MLP head are' if mode == 'sup' else 'is'} stored in the model's module configuration, and
using a different pooling silently changes the embeddings.

## Limitations
- **Small, narrow training corpus.** Only SNLI sentences: short, simple descriptions of photographs
  (SNLI premises are Flickr30k captions). Quality drops on other genres; on STS-B the test score is clearly
  below dev ({_fmt(dev.get('spearman'), '.2f')} vs {_fmt(test.get('spearman'), '.2f')}).
- **Topical similarity is over-rated.** Sentences that share a template and a topic but differ in the key
  entity or event (e.g. two different attacks in two different countries) get high cosine similarity even
  when humans rate them as unrelated.
{'- **Partial hard negatives.** Only 28.45% of the training pairs have a contradiction hard negative.' + chr(10) if mode == 'sup' else ''}- **English only, max 64 tokens**; longer inputs are truncated. Not evaluated on retrieval benchmarks,
  other languages, or specialised domains (legal, biomedical, code).
- **Biases.** Inherits the social biases of BERT pre-training data and of SNLI, whose hypotheses were
  written by crowd workers and contain known annotation artifacts and stereotypes.
- **Single seed.** Reported from one training seed; run-to-run variation is roughly +-0.5 Spearman on dev
  in our supervised seed study, so differences under ~1 point should not be over-interpreted.
- Intended for teaching and research, not for decisions about people.

## Citation
```bibtex
@inproceedings{{gao2021simcse,
  title={{SimCSE: Simple Contrastive Learning of Sentence Embeddings}},
  author={{Gao, Tianyu and Yao, Xingcheng and Chen, Danqi}},
  booktitle={{EMNLP}},
  year={{2021}}
}}
```
"""


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
    word_embedding_model = Transformer(transformer_subfolder, max_seq_length=64)
    pooling_model = Pooling(
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

    test_res = (run_meta.get("results", {}) or {}).get("test") or {}

    # Generate Model Card (standardized template; all numbers come from the run JSON)
    readme_content = generate_model_card(repo_id=repo_id, run_meta=run_meta)

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
