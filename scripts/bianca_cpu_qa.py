"""CPU QA + report figures (Bianca). Runs on a laptop without GPU in ~5-10 min.

What it does (all evaluation-only, no training, never touches the test number):
  1. Regenerates the raw-BERT similarity-distribution figure with the current
     (0-5 scale) code. The old figure only had 2 score brackets because it was
     produced while STS-B scores were still on the [0, 1] scale.
  2. Nearest-neighbour retrieval for the published unsupervised model (and, if
     you pass --sup_checkpoint, for Jonav's supervised checkpoint): the corpus is
     every unique sentence of STS-B test; for a few query sentences it lists the
     top-5 neighbours, and it reports the high-score pairs (human >= 4) whose
     partner is ranked worst = retrieval failure cases for the report.
  3. (optional) --ablation_checkpoint: recomputes the same-dropout-mask
     ablation with the current alignment definition (needs Russel's checkpoint).

Usage (from the repo root, inside the venv):
    python scripts/bianca_cpu_qa.py
    python scripts/bianca_cpu_qa.py --sup_checkpoint "path/to/jonav_sup_hnon_.../best_checkpoint"
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis import encode_sentences, generate_similarity_distribution_plot  # noqa: E402
from src.data import load_stsb_dataset  # noqa: E402

HUB_UNSUP = "RusselKuAguilar/simcse-bert-uncased-unsup"
QUERIES = [
    "A man is playing a guitar.",
    "A woman is slicing an onion.",
    "Gunmen kill nine people in northwest Pakistan",
    "The stock market fell sharply on Monday.",
]


def nn_retrieval(model_path, label, out_dir, k=5):
    _, test = load_stsb_dataset()
    corpus = sorted({r["sentence1"] for r in test} | {r["sentence2"] for r in test})
    index = {s: i for i, s in enumerate(corpus)}
    emb = encode_sentences(model_path, corpus, device="cpu", pooling="cls")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    q_emb = encode_sentences(model_path, QUERIES, device="cpu", pooling="cls")
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    queries = []
    for q, v in zip(QUERIES, q_emb):
        sims = emb @ v
        if q in index:
            sims[index[q]] = -np.inf  # a query that is in the corpus must not retrieve itself
        top = np.argsort(-sims)[:k]
        queries.append({"query": q, "neighbors": [
            {"rank": r + 1, "sentence": corpus[j], "cosine": round(float(sims[j]), 4)}
            for r, j in enumerate(top)]})

    # Failure cases: human score >= 4 but the partner is ranked far down.
    fails = []
    for r in test:
        if r["score"] < 4.0:
            continue
        i, j = index[r["sentence1"]], index[r["sentence2"]]
        sims = emb @ emb[i]
        sims[i] = -np.inf  # exclude the query itself
        rank = int((sims > sims[j]).sum()) + 1
        top1 = int(np.argmax(sims))
        fails.append({"sentence1": r["sentence1"], "sentence2": r["sentence2"],
                      "human_score": round(r["score"], 2), "partner_rank": rank,
                      "partner_cosine": round(float(sims[j]), 4),
                      "retrieved_top1": corpus[top1], "top1_cosine": round(float(sims[top1]), 4)})
    fails.sort(key=lambda x: -x["partner_rank"])
    ranks = np.array([f["partner_rank"] for f in fails])
    result = {"model": model_path, "corpus_size": len(corpus),
              "high_score_pairs": len(fails),
              "recall_at_1": round(float((ranks == 1).mean()), 4),
              "recall_at_5": round(float((ranks <= 5).mean()), 4),
              "median_partner_rank": int(np.median(ranks)),
              "queries": queries, "worst_failures": fails[:5]}

    path_json = os.path.join(out_dir, f"nn_retrieval_{label}.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [f"### Nearest-neighbour retrieval: {label}", "",
             f"Corpus: {len(corpus)} unique STS-B test sentences. Over the {len(fails)} pairs "
             f"with human score >= 4: Recall@1 = {result['recall_at_1']:.2%}, "
             f"Recall@5 = {result['recall_at_5']:.2%}, median partner rank = {result['median_partner_rank']}.", ""]
    for qd in queries:
        lines += [f"**Query:** {qd['query']}", "", "| Rank | Neighbour | Cosine |", "|---:|---|---:|"]
        lines += [f"| {n['rank']} | {n['sentence']} | {n['cosine']:.3f} |" for n in qd["neighbors"]] + [""]
    lines += ["**Worst failures (high human score, partner ranked low):**", "",
              "| Human | Sentence 1 | Partner (rank, cos) | Retrieved top-1 instead (cos) |", "|---:|---|---|---|"]
    lines += [f"| {f['human_score']} | {f['sentence1']} | {f['sentence2']} ({f['partner_rank']}, {f['partner_cosine']:.3f}) "
              f"| {f['retrieved_top1']} ({f['top1_cosine']:.3f}) |" for f in fails[:5]]
    path_md = os.path.join(out_dir, f"nn_retrieval_{label}.md")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path_json} and {path_md} (R@1={result['recall_at_1']:.2%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sup_checkpoint", default=None, help="Jonav's supervised best_checkpoint folder")
    p.add_argument("--ablation_checkpoint", default=None, help="Russel's same-mask ablation best_checkpoint")
    p.add_argument("--out_dir", default="reports/figures")
    p.add_argument("--skip_plot", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if not a.skip_plot:
        generate_similarity_distribution_plot("bert-base-uncased",
            os.path.join(a.out_dir, "similarity_distribution_bert.png"), device="cpu", pooling="mean")
    nn_retrieval(HUB_UNSUP, "unsup", a.out_dir)
    if a.sup_checkpoint:
        if not a.skip_plot:
            generate_similarity_distribution_plot(a.sup_checkpoint,
                os.path.join(a.out_dir, "similarity_distribution_sup.png"), device="cpu", pooling="cls")
        nn_retrieval(a.sup_checkpoint, "sup", a.out_dir)
    if a.ablation_checkpoint:
        from src.evaluate import run_evaluation
        run_evaluation(a.ablation_checkpoint, model_type="simcse", pooling="cls", device="cpu",
                       output_json="runs/ablation_unsup_same_mask_realigned.json")


if __name__ == "__main__":
    main()
