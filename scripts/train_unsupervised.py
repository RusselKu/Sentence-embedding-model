"""Helper script to train Unsupervised SimCSE."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import train, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unsupervised SimCSE")
    parser.add_argument("--data_path", type=str, default="data/snli_train_100k.jsonl")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--pooling", type=str, default="cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="unsup_simcse_default")
    args = parser.parse_args()

    args.mode = "unsup"
    args.model_name_or_path = "bert-base-uncased"
    args.output_dir = "checkpoints"
    args.dropout_rate = None
    args.weight_decay = 0.0
    args.warmup_ratio = 0.05
    args.max_length = 64
    args.eval_steps = 250
    args.same_dropout_mask = False
    args.no_hard_negatives = False

    train(args)
