"""Validate the SNLI 100k subset used by the SimCSE project.

This script checks:
- Total number of JSONL records.
- Label distribution.
- Number of unique sentences.
- Number of supervised entailment pairs.
- Number and percentage of pairs with hard negatives.
- Whitespace normalization collisions.

Expected values from the assignment:
- 100,000 records
- 165,529 unique sentences
- 33,351 supervised entailment pairs
- ~28% with contradiction hard negatives
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_RECORDS = 100_000
EXPECTED_UNIQUE_SENTENCES = 165_529
EXPECTED_SUPERVISED_PAIRS = 33_351


def load_records(path: Path):
    """Load JSONL records and perform basic schema validation."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            required_fields = {"premise", "hypothesis", "label"}

            missing = required_fields - record.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {sorted(missing)}"
                )

            records.append(record)

    return records


def analyze_dataset(records):
    """Calculate the main statistics required by the assignment."""

    label_counts = Counter()

    # Exact strings as provided by the assignment dataset.
    exact_sentences = set()

    # Used only to detect whitespace collisions.
    stripped_sentences = set()

    premise_map = defaultdict(lambda: {0: [], 1: [], 2: []})

    for record in records:
        premise = record["premise"]
        hypothesis = record["hypothesis"]
        label = record["label"]

        label_counts[label] += 1

        if premise:
            exact_sentences.add(premise)
            stripped_sentences.add(premise.strip())

        if hypothesis:
            exact_sentences.add(hypothesis)
            stripped_sentences.add(hypothesis.strip())

        clean_premise = premise.strip()
        clean_hypothesis = hypothesis.strip()

        if (
            clean_premise
            and clean_hypothesis
            and label in (0, 1, 2)
        ):
            premise_map[clean_premise][label].append(clean_hypothesis)

    supervised_pairs = 0
    hard_negative_pairs = 0

    for labels in premise_map.values():
        entailments = labels[0]
        contradictions = labels[2]

        supervised_pairs += len(entailments)

        if contradictions:
            hard_negative_pairs += len(entailments)

    hard_negative_percentage = (
        hard_negative_pairs / supervised_pairs * 100
        if supervised_pairs
        else 0.0
    )

    return {
        "records": len(records),
        "label_counts": label_counts,
        "exact_unique_sentences": len(exact_sentences),
        "stripped_unique_sentences": len(stripped_sentences),
        "whitespace_collisions": (
            len(exact_sentences) - len(stripped_sentences)
        ),
        "supervised_pairs": supervised_pairs,
        "hard_negative_pairs": hard_negative_pairs,
        "hard_negative_percentage": hard_negative_percentage,
    }


def print_report(stats):
    """Print a human-readable validation report."""

    print("\n" + "=" * 60)
    print("SNLI 100K DATASET VALIDATION")
    print("=" * 60)

    print(f"\nTotal records:               {stats['records']:,}")

    print("\nLabel distribution:")
    print(
        f"  Entailment     (0):        "
        f"{stats['label_counts'][0]:,}"
    )
    print(
        f"  Neutral        (1):        "
        f"{stats['label_counts'][1]:,}"
    )
    print(
        f"  Contradiction  (2):        "
        f"{stats['label_counts'][2]:,}"
    )

    print("\nUnsupervised dataset:")
    print(
        f"  Exact unique sentences:    "
        f"{stats['exact_unique_sentences']:,}"
    )
    print(
        f"  After .strip():             "
        f"{stats['stripped_unique_sentences']:,}"
    )
    print(
        f"  Whitespace collisions:      "
        f"{stats['whitespace_collisions']:,}"
    )

    print("\nSupervised dataset:")
    print(
        f"  Entailment pairs:           "
        f"{stats['supervised_pairs']:,}"
    )
    print(
        f"  With hard negatives:        "
        f"{stats['hard_negative_pairs']:,}"
    )
    print(
        f"  Hard-negative percentage:   "
        f"{stats['hard_negative_percentage']:.2f}%"
    )

    print("\nExpected assignment values:")
    print(f"  Records:                    {EXPECTED_RECORDS:,}")
    print(
        f"  Unique sentences:           "
        f"{EXPECTED_UNIQUE_SENTENCES:,}"
    )
    print(
        f"  Supervised pairs:           "
        f"{EXPECTED_SUPERVISED_PAIRS:,}"
    )

    print("\nValidation:")

    checks = {
        "Record count": (
            stats["records"] == EXPECTED_RECORDS
        ),
        "Unique sentence count": (
            stats["exact_unique_sentences"]
            == EXPECTED_UNIQUE_SENTENCES
        ),
        "Supervised pair count": (
            stats["supervised_pairs"]
            == EXPECTED_SUPERVISED_PAIRS
        ),
        "Hard negatives close to 28%": (
            27.0
            <= stats["hard_negative_percentage"]
            <= 30.0
        ),
    }

    for name, passed in checks.items():
        symbol = "PASS" if passed else "FAIL"
        print(f"  [{symbol}] {name}")

    print("=" * 60)

    return all(checks.values())


def main():
    parser = argparse.ArgumentParser(
        description="Validate the SNLI 100k dataset."
    )

    parser.add_argument(
        "--data",
        default="snli_train_100k.jsonl",
        help="Path to the SNLI JSONL file.",
    )

    args = parser.parse_args()

    path = Path(args.data)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    print(f"Reading dataset: {path.resolve()}")

    records = load_records(path)

    stats = analyze_dataset(records)

    valid = print_report(stats)

    if not valid:
        raise SystemExit(
            "\nDataset validation failed."
        )

    print("\nDataset validation completed successfully.")


if __name__ == "__main__":
    main()
