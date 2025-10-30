"""Utility for converting ML-2-QML log entries into a Hugging Face dataset.

The script reads the structured metadata stored in ``ML-2-QML/log.json`` and
aggregates it with the referenced classical and quantum source files.  It then
uploads the assembled dataset to the Hugging Face Hub.

Example
-------
python dataset/build_hf_dataset.py \
    --repo-id username/llms-qml \
    --token hf_xxx \
    --private
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import Dataset, DatasetDict, Features, Value


DEFAULT_LOG_PATH = Path("ML-2-QML/log.json")


def _find_relative_repo_root(path: Path, marker: str) -> Path:
    """Return the subpath of ``path`` starting from ``marker``.

    Parameters
    ----------
    path:
        Absolute path recorded in the log file.
    marker:
        Directory name that is guaranteed to exist in the repository.  For the
        ML/QML code this is ``"ML-2-QML"``.

    Returns
    -------
    Path
        Relative path from the repository root to the desired file.

    Raises
    ------
    ValueError
        If ``marker`` is not part of ``path``.
    """

    try:
        marker_index = path.parts.index(marker)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"'{marker}' not found in '{path}'") from exc

    relative_parts = path.parts[marker_index:]
    return Path(*relative_parts)


def _remap_path(original_path: str, repo_root: Path) -> Path:
    """Convert absolute paths in the log to valid paths in the checkout."""
    original = Path(original_path)
    relative = _find_relative_repo_root(original, "ML-2-QML")
    local_path = repo_root / relative

    if not local_path.exists():
        raise FileNotFoundError(f"Could not locate '{local_path}'")

    return local_path


def _build_records(raw_entries: Iterable[Dict], repo_root: Path) -> List[Dict]:
    """Assemble Hugging Face dataset records from raw log entries."""
    records: List[Dict] = []
    for entry in raw_entries:
        ml_path = _remap_path(entry["ML_code_path"], repo_root)
        qml_path = _remap_path(entry["QML_code_path"], repo_root)

        record = {
            "id": entry["id"],
            "number of references": entry["reference_number"],
            "average length": entry["length"]["average"],
            "scaling_paradigm": entry["scaling_paradigm"],
            "summary": entry["summary"],
            "CML": ml_path.read_text(encoding="utf-8"),
            "QML": qml_path.read_text(encoding="utf-8"),
        }
        records.append(record)

    return records


def build_dataset(log_path: Path, repo_root: Path) -> Dataset:
    """Create a ``datasets.Dataset`` containing the combined records."""
    with log_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    records = _build_records(raw_data.values(), repo_root)

    features = Features(
        {
            "id": Value("int32"),
            "number of references": Value("int32"),
            "average length": Value("float32"),
            "scaling_paradigm": Value("string"),
            "summary": Value("string"),
            "CML": Value("string"),
            "QML": Value("string"),
        }
    )

    return Dataset.from_list(records, features=features)


def split_dataset(dataset: Dataset, test_size: float = 0.1) -> DatasetDict:
    """Split a dataset into train and test partitions."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    return dataset.train_test_split(test_size=test_size)


def push_dataset(
    dataset: Dataset | DatasetDict,
    repo_id: str,
    token: str | None = None,
    branch: str | None = None,
) -> None:
    """Upload the dataset to the Hugging Face Hub."""

    dataset.push_to_hub(repo_id, token=token, branch=branch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the ML-2-QML log.json file",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Root of the repository; used to resolve local file paths",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target dataset repository on the Hugging Face Hub",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token. If omitted, relies on cached login.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help=(
            "Branch name to use when pushing to the Hub. When omitted, the main "
            "branch is updated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.log_path, args.repo_root)
    dataset_dict = split_dataset(dataset, test_size=0.1)
    push_dataset(dataset_dict, args.repo_id, token=args.token, branch=args.branch)


if __name__ == "__main__":
    main()