---
language:
- en
license: apache-2.0
library_name: sentence-transformers
base_model: bert-base-uncased
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
- name: simcse-bert-uncased-unsup
  results:
  - task:
      type: semantic-similarity
    dataset:
      name: STS-Benchmark
      type: sentence-transformers/stsb
      split: test
    metrics:
    - type: spearman_cosine
      value: 67.32
---

# RusselKuAguilar/simcse-bert-uncased-unsup

Unsupervised **SimCSE** sentence encoder trained from `bert-base-uncased` on a
100k-record subset of SNLI, as a course replication of Gao, Yao & Chen (EMNLP 2021). It maps an English
sentence to a 768-dimensional vector; compare sentences with cosine similarity.

Team (Universidad Politécnica de Yucatán, U2T02): Acosta Castellanos Bianca Alexandra, Canche Chuc Angel Rivaldo, Ku Aguilar Russel, Sanchez Novelo Damian, Velasco Martin Jonathan Abisai.

## Training data
165,528 unique sentences (premises and hypotheses, whitespace-stripped) from `snli_train_100k.jsonl`, a 100k-record subset of the SNLI train split. Labels are not used: each sentence is its own positive, the two views differ only by dropout.

The paper trained on much more data (1M Wikipedia sentences for unsupervised; ~314k SNLI+MNLI pairs for
supervised). This model is therefore **not** expected to reach the paper's numbers; see Limitations.

## Training recipe
Unsupervised SimCSE (Gao et al., 2021, Eq. 1): InfoNCE over a batch of N sentences, positive = the same sentence encoded with an independent dropout mask, negatives = the other N-1 = 63 sentences of the batch.

| Setting | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Pooling | CLS |
| Learning rate | 3e-05 |
| Batch size | 64 |
| Epochs | 1 |
| Temperature (tau) | 0.05 |
| Max sequence length | 64 |
| Dropout | 0.1 (bert-base-uncased default) |
| Warmup ratio / weight decay | 0.05 / 0.0 |
| Seed | 42 |
| Checkpoint selection | best STS-B dev Spearman, evaluated every 250 steps and at epoch end |
| Hardware | NVIDIA L4, 835.2 s |

**Inference representation:** [CLS] token of the last layer. The MLP head was used only during training and is discarded at inference, as in the paper.

## Evaluation (STS-B, Spearman x100)
Protocol: encode both sentences, L2-normalize, cosine similarity, Spearman against the human scores;
no regressor ("all" aggregation, Gao et al. Appendix B). Test was evaluated once, after selecting the
checkpoint on dev. Alignment uses STS-B pairs with score >= 4; uniformity uses all sentences
(Wang & Isola, 2020; lower is better for both).

| Split | Spearman | Alignment (a=2) | Uniformity (t=2) | Paper (Gao et al.) |
|---|---:|---:|---:|---:|
| Dev | 76.32 | 0.3150 | -2.8887 | 82.5 |
| Test | 67.32 | 0.3501 | -2.8900 | 76.85 |

Reference baselines with the same evaluation code: raw `bert-base-uncased` (mean pooling) 59.31 / 47.29;
SBERT-2019 (`bert-base-nli-mean-tokens`) 80.77 / 76.98 (dev / test).

Verified after upload: reloaded from the Hub and re-evaluated on STS-B test = **67.32** (table value 67.32).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("RusselKuAguilar/simcse-bert-uncased-unsup")
sentences = [
    "A man is playing a guitar.",
    "Someone is playing an instrument.",
    "A dog is chasing a ball in the park.",
]
emb = model.encode(sentences, normalize_embeddings=True)
print(emb @ emb.T)  # cosine similarity matrix
```

Use `sentence-transformers` (not a raw `AutoModel` with your own pooling): the pooling
is stored in the model's module configuration, and
using a different pooling silently changes the embeddings.

## Limitations
- **Small, narrow training corpus.** Only SNLI sentences: short, simple descriptions of photographs
  (SNLI premises are Flickr30k captions). Quality drops on other genres; on STS-B the test score is clearly
  below dev (76.32 vs 67.32).
- **Topical similarity is over-rated.** Sentences that share a template and a topic but differ in the key
  entity or event (e.g. two different attacks in two different countries) get high cosine similarity even
  when humans rate them as unrelated.
- **English only, max 64 tokens**; longer inputs are truncated. Not evaluated on retrieval benchmarks,
  other languages, or specialised domains (legal, biomedical, code).
- **Biases.** Inherits the social biases of BERT pre-training data and of SNLI, whose hypotheses were
  written by crowd workers and contain known annotation artifacts and stereotypes.
- **Single seed.** Reported from one training seed; run-to-run variation is roughly +-0.5 Spearman on dev
  in our supervised seed study, so differences under ~1 point should not be over-interpreted.
- Intended for teaching and research, not for decisions about people.

## Citation
```bibtex
@inproceedings{gao2021simcse,
  title={SimCSE: Simple Contrastive Learning of Sentence Embeddings},
  author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle={EMNLP},
  year={2021}
}
```
