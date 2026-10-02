"""Validate SimCSE training run history.

The SimCSE training pipeline stores experiment metadata in:

    runs/run_history.json

Each training run should contain enough information to reproduce
and analyze the experiment, including configuration, seed, hardware,
evaluation results, and checkpoint information.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


REQUIRED_TOP_LEVEL_FIELDS: Set[str] = {
    "run_name",
    "timestamp",
    "mode",
    "model_name_or_path",
    "pooling",
    "hyperparameters",
    "hardware",
    "training_time_seconds",
    "results",
    "checkpoint_path",
}

REQUIRED_HYPERPARAMETER_FIELDS: Set[str] = {
    "lr",
    "batch_size",
    "epochs",
    "temperature",
    "dropout_rate",
    "weight_decay",
    "warmup_ratio",
    "seed",
    "same_dropout_mask_ablation",
    "no_hard_negatives_ablation",
}

REQUIRED_HARDWARE_FIELDS: Set[str] = {
    "os",
    "python_version",
    "device",
    "gpu_name",
}

REQUIRED_RESULT_SPLITS: Set[str] = {
    "dev",
    "test",
}


def load_history(path: Path) -> List[Dict[str, Any]]:
    """Load run_history.json.

    An empty list is returned if no training runs exist yet.
    """
    if not path.exists():
        print(f"Run history not found: {path}")
        print(
            "This is expected before the first SimCSE training run. "
            "src/train.py creates the file automatically."
        )
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "run_history.json must contain a JSON list of run records."
        )

    return data


def find_missing_fields(
    data: Dict[str, Any],
    required: Set[str],
) -> List[str]:
    """Return required dictionary keys that are missing."""
    return sorted(required - set(data.keys()))


def validate_run(
    run: Dict[str, Any],
    index: int,
) -> List[str]:
    """Validate one SimCSE training-run record."""
    problems: List[str] = []

    if not isinstance(run, dict):
        return [
            f"Run #{index} must be a JSON object, "
            f"got {type(run).__name__}."
        ]

    missing = find_missing_fields(
        run,
        REQUIRED_TOP_LEVEL_FIELDS,
    )

    if missing:
        problems.append(
            f"Missing top-level fields: {missing}"
        )

    mode = run.get("mode")

    if mode not in {"unsup", "sup"}:
        problems.append(
            f"Invalid mode: {mode!r}. "
            "Expected 'unsup' or 'sup'."
        )

    pooling = run.get("pooling")

    if pooling not in {"cls", "mean"}:
        problems.append(
            f"Unexpected pooling strategy: {pooling!r}."
        )

    hyperparameters = run.get("hyperparameters")

    if not isinstance(hyperparameters, dict):
        problems.append(
            "'hyperparameters' must be a JSON object."
        )
    else:
        missing = find_missing_fields(
            hyperparameters,
            REQUIRED_HYPERPARAMETER_FIELDS,
        )

        if missing:
            problems.append(
                f"Missing hyperparameters: {missing}"
            )

        seed = hyperparameters.get("seed")

        if not isinstance(seed, int):
            problems.append(
                "Hyperparameter 'seed' must be an integer."
            )

        batch_size = hyperparameters.get("batch_size")

        if (
            batch_size is not None
            and (
                not isinstance(batch_size, int)
                or batch_size <= 0
            )
        ):
            problems.append(
                "'batch_size' must be a positive integer."
            )

        epochs = hyperparameters.get("epochs")

        if (
            epochs is not None
            and (
                not isinstance(epochs, int)
                or epochs <= 0
            )
        ):
            problems.append(
                "'epochs' must be a positive integer."
            )

    hardware = run.get("hardware")

    if not isinstance(hardware, dict):
        problems.append(
            "'hardware' must be a JSON object."
        )
    else:
        missing = find_missing_fields(
            hardware,
            REQUIRED_HARDWARE_FIELDS,
        )

        if missing:
            problems.append(
                f"Missing hardware fields: {missing}"
            )

        device = hardware.get("device")

        if device not in {"cpu", "cuda"}:
            problems.append(
                f"Unexpected hardware device: {device!r}."
            )

    results = run.get("results")

    if not isinstance(results, dict):
        problems.append(
            "'results' must be a JSON object."
        )
    else:
        missing = find_missing_fields(
            results,
            REQUIRED_RESULT_SPLITS,
        )

        if missing:
            problems.append(
                f"Missing evaluation splits: {missing}"
            )

        for split in REQUIRED_RESULT_SPLITS:
            split_results = results.get(split)

            if split_results is None:
                continue

            if not isinstance(split_results, dict):
                problems.append(
                    f"Results for '{split}' must be a JSON object."
                )

    training_time = run.get(
        "training_time_seconds"
    )

    if training_time is not None:
        if (
            not isinstance(training_time, (int, float))
            or training_time < 0
        ):
            problems.append(
                "'training_time_seconds' must be "
                "a non-negative number."
            )

    return problems


def validate_history(
    history: List[Dict[str, Any]],
) -> bool:
    """Validate every training run."""
    if not history:
        print("\nNo SimCSE training runs available yet.")
        print(
            "The validator itself is ready. "
            "Run it again after Russel or Joni produces "
            "the first training run."
        )
        return True

    print("\n" + "=" * 65)
    print("SIMCSE RUN HISTORY VALIDATION")
    print("=" * 65)

    print(f"\nRuns found: {len(history)}\n")

    all_valid = True

    run_names = set()

    for index, run in enumerate(
        history,
        start=1,
    ):
        run_name = (
            run.get("run_name", "<unnamed>")
            if isinstance(run, dict)
            else "<invalid>"
        )

        problems = validate_run(
            run,
            index,
        )

        if run_name in run_names:
            problems.append(
                f"Duplicate run_name: {run_name!r}"
            )

        run_names.add(run_name)

        if problems:
            all_valid = False

            print(
                f"[FAIL] Run #{index}: {run_name}"
            )

            for problem in problems:
                print(f"       - {problem}")

        else:
            print(
                f"[PASS] Run #{index}: {run_name}"
            )

    print("\n" + "-" * 65)

    if all_valid:
        print(
            "All run records contain the required "
            "reproducibility metadata."
        )
    else:
        print(
            "One or more run records require correction."
        )

    print("=" * 65)

    return all_valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SimCSE training "
            "run_history.json."
        )
    )

    parser.add_argument(
        "--history",
        default="runs/run_history.json",
        help="Path to run_history.json.",
    )

    args = parser.parse_args()

    path = Path(args.history)

    print(f"Checking run history: {path}")

    history = load_history(path)

    valid = validate_history(history)

    if not valid:
        raise SystemExit(
            "\nRun history validation failed."
        )

    print(
        "\nRun history validation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
