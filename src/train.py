"""Training script for SimCSE (Unsupervised and Supervised).
Includes STS-B Dev validation, checkpointing, and comprehensive JSON run logging.
"""

import argparse
import datetime
import json
import os
import platform
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import (
    build_unsupervised_dataset,
    build_supervised_dataset,
    UnsupervisedSimCSEDataset,
    SupervisedSimCSEDataset,
    UnsupDataCollator,
    SupDataCollator,
    load_stsb_dataset,
)
from src.metrics import evaluate_sts_benchmark
from src.models import SimCSEModel


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hardware_info() -> Dict:
    """Collect hardware specifications for run metadata logging."""
    info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "N/A (CPU)"
    return info


def evaluate_on_stsb(
    model: SimCSEModel,
    tokenizer: AutoTokenizer,
    records: list,
    device: str,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate model on STS-B records (dev or test split)."""
    model.eval()
    sent1 = [r["sentence1"] for r in records]
    sent2 = [r["sentence2"] for r in records]
    scores = [r["score"] for r in records]

    def encode(sentences):
        all_emb = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_sentence_embeddings(inputs["input_ids"], inputs["attention_mask"], normalize=True)
            all_emb.append(emb.cpu().numpy())
        return np.concatenate(all_emb, axis=0)

    emb1 = encode(sent1)
    emb2 = encode(sent2)
    return evaluate_sts_benchmark(emb1, emb2, scores)


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Create output directory
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.mode}_{args.pooling}_{run_timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = SimCSEModel(
        model_name_or_path=args.model_name_or_path,
        pooling=args.pooling,
        temperature=args.temperature,
        use_mlp=True,
        dropout_rate=args.dropout_rate,
    ).to(device)

    # Load STS-B for validation
    dev_records, test_records = load_stsb_dataset()
    print(f"Loaded STS-B: {len(dev_records)} Dev, {len(test_records)} Test pairs.")

    # Load and build training dataset
    print(f"Loading training data from {args.data_path} for mode: {args.mode}")
    if args.mode == "unsup":
        sentences = build_unsupervised_dataset(args.data_path)
        print(f"Built Unsupervised dataset with {len(sentences)} unique sentences.")
        dataset = UnsupervisedSimCSEDataset(sentences)
        collator = UnsupDataCollator(tokenizer, max_length=args.max_length)
    else:  # supervised
        triplets = build_supervised_dataset(args.data_path)
        has_neg_count = sum(1 for t in triplets if t["negative"] is not None)
        print(f"Built Supervised dataset with {len(triplets)} pairs ({has_neg_count} have hard negatives).")
        dataset = SupervisedSimCSEDataset(triplets, use_hard_negatives=not args.no_hard_negatives)
        collator = SupDataCollator(tokenizer, max_length=args.max_length, use_hard_negatives=not args.no_hard_negatives)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\n--- Starting Training: {run_name} ---")
    print(f"Total Steps: {total_steps} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")

    best_dev_spearman = -1.0
    best_checkpoint_path = os.path.join(output_dir, "best_checkpoint")
    start_time = time.time()
    step_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            if args.mode == "unsup":
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _ = model.forward_unsupervised(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    same_dropout_mask=args.same_dropout_mask,
                )
            else:  # supervised
                premises = {k: v.to(device) for k, v in batch["premises"].items()}
                positives = {k: v.to(device) for k, v in batch["positives"].items()}
                negatives = {k: v.to(device) for k, v in batch["negatives"].items()} if batch["negatives"] is not None else None
                loss, _ = model.forward_supervised(
                    premise_inputs=premises,
                    positive_inputs=positives,
                    negative_inputs=negatives,
                    use_hard_negatives=not args.no_hard_negatives,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            step_losses.append(loss_val)
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            # Intermediate dev evaluation if requested
            global_step = (epoch - 1) * len(dataloader) + step + 1
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                dev_res = evaluate_on_stsb(model, tokenizer, dev_records, device)
                print(f"\n[Step {global_step}] Dev Spearman: {dev_res['spearman']:.2f} (Best: {best_dev_spearman:.2f})")
                if dev_res["spearman"] > best_dev_spearman:
                    best_dev_spearman = dev_res["spearman"]
                    os.makedirs(best_checkpoint_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(best_checkpoint_path, "pytorch_model.bin"))
                    tokenizer.save_pretrained(best_checkpoint_path)
                model.train()

        # End of epoch evaluation
        dev_res = evaluate_on_stsb(model, tokenizer, dev_records, device)
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n[Epoch {epoch} Done] Avg Loss: {avg_loss:.4f} | Dev Spearman: {dev_res['spearman']:.2f}")

        if dev_res["spearman"] > best_dev_spearman:
            best_dev_spearman = dev_res["spearman"]
            os.makedirs(best_checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_checkpoint_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(best_checkpoint_path)

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes.")

    # Load best checkpoint for final evaluation
    print("\n--- Final Evaluation on Best Checkpoint ---")
    model.load_state_dict(torch.load(os.path.join(best_checkpoint_path, "pytorch_model.bin"), map_location=device))
    dev_final = evaluate_on_stsb(model, tokenizer, dev_records, device)
    test_final = evaluate_on_stsb(model, tokenizer, test_records, device)

    print(f"Final Best Dev  -> Spearman: {dev_final['spearman']:.2f} | Alignment: {dev_final['alignment']:.4f} | Uniformity: {dev_final['uniformity']:.4f}")
    print(f"Final Best Test -> Spearman: {test_final['spearman']:.2f} | Alignment: {test_final['alignment']:.4f} | Uniformity: {test_final['uniformity']:.4f}")

    # Build Run Metadata
    run_metadata = {
        "run_name": run_name,
        "timestamp": run_timestamp,
        "mode": args.mode,
        "model_name_or_path": args.model_name_or_path,
        "pooling": args.pooling,
        "hyperparameters": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "temperature": args.temperature,
            "dropout_rate": args.dropout_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "same_dropout_mask_ablation": args.same_dropout_mask,
            "no_hard_negatives_ablation": args.no_hard_negatives,
        },
        "hardware": get_hardware_info(),
        "training_time_seconds": round(elapsed_time, 2),
        "results": {
            "dev": dev_final,
            "test": test_final,
        },
        "checkpoint_path": best_checkpoint_path,
    }

    # Save run json
    run_file = os.path.join("runs", f"{run_name}.json")
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved run metadata to: {run_file}")

    # Append to run history
    history_file = os.path.join("runs", "run_history.json")
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(run_metadata)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return run_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimCSE Model")
    parser.add_argument("--mode", type=str, choices=["unsup", "sup"], required=True, help="Training mode")
    parser.add_argument("--data_path", type=str, default="data/snli_train_100k.jsonl", help="Path to SNLI 100k JSONL")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Base model")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run identifier")
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls", help="Pooling strategy")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (e.g. 3e-5 for unsup, 5e-5 for sup)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--temperature", type=float, default=0.05, help="Softmax temperature tau")
    parser.add_argument("--dropout_rate", type=float, default=None, help="Encoder dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=64, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluate dev set every N steps (0 to disable)")

    # Ablation Flags
    parser.add_argument("--same_dropout_mask", action="store_true", help="Ablation: use identical dropout mask (unsup)")
    parser.add_argument("--no_hard_negatives", action="store_true", help="Ablation: disable hard negatives (sup)")

    args = parser.parse_args()
    train(args)
