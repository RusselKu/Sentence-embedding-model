"""Script to run both required SimCSE ablations:
1. Unsupervised: Same dropout mask for both views.
2. Supervised: Hard negatives ON vs. OFF.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import train, argparse


def run_unsupervised_ablation(data_path: str = "data/snli_train_100k.jsonl"):
    print("\n" + "=" * 60)
    print("RUNNING ABLATION 1: Unsupervised SimCSE (Same Dropout Mask)")
    print("=" * 60)
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.mode = "unsup"
    args.data_path = data_path
    args.model_name_or_path = "bert-base-uncased"
    args.output_dir = "checkpoints"
    args.run_name = "ablation_unsup_same_mask"
    args.pooling = "cls"
    args.lr = 3e-5
    args.batch_size = 64
    args.epochs = 1
    args.temperature = 0.05
    args.dropout_rate = None
    args.weight_decay = 0.0
    args.warmup_ratio = 0.05
    args.max_length = 64
    args.seed = 42
    args.eval_steps = 250
    args.same_dropout_mask = True
    args.no_hard_negatives = False

    return train(args)


def run_supervised_ablation(data_path: str = "data/snli_train_100k.jsonl"):
    print("\n" + "=" * 60)
    print("RUNNING ABLATION 2: Supervised SimCSE (Hard Negatives OFF)")
    print("=" * 60)
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.mode = "sup"
    args.data_path = data_path
    args.model_name_or_path = "bert-base-uncased"
    args.output_dir = "checkpoints"
    args.run_name = "ablation_sup_no_hard_negatives"
    args.pooling = "cls"
    args.lr = 5e-5
    args.batch_size = 64
    args.epochs = 3
    args.temperature = 0.05
    args.dropout_rate = None
    args.weight_decay = 0.0
    args.warmup_ratio = 0.05
    args.max_length = 64
    args.seed = 42
    args.eval_steps = 250
    args.same_dropout_mask = False
    args.no_hard_negatives = True

    return train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SimCSE Ablation Experiments")
    parser.add_argument("--data_path", type=str, default="data/snli_train_100k.jsonl")
    parser.add_argument("--ablation", type=str, choices=["all", "unsup", "sup"], default="all")
    args = parser.parse_args()

    if args.ablation in ("all", "unsup"):
        run_unsupervised_ablation(args.data_path)
    if args.ablation in ("all", "sup"):
        run_supervised_ablation(args.data_path)
