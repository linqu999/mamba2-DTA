"""Dataset splitting helpers with leakage checks."""

from __future__ import annotations

import json
import random
import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


SplitName = Literal["random", "cold-drug", "cold-target"]


@dataclass(frozen=True)
class SplitMetadata:
    split: str
    seed: int
    train_rows: int
    valid_rows: int
    test_rows: int
    train_drugs: int
    valid_drugs: int
    test_drugs: int
    train_targets: int
    valid_targets: int
    test_targets: int
    leakage_checked: bool


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for split generation") from exc
    return pd


def _validate_ratios(valid_ratio: float, test_ratio: float) -> None:
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1)")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1)")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be less than 1")


def _shuffle(values: list[str], seed: int) -> list[str]:
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _cut_counts(total: int, valid_ratio: float, test_ratio: float) -> tuple[int, int]:
    test_count = int(round(total * test_ratio))
    valid_count = int(round(total * valid_ratio))
    if total >= 3:
        test_count = max(1, test_count)
        valid_count = max(1, valid_count)
        if test_count + valid_count >= total:
            valid_count = max(1, total - test_count - 1)
    return valid_count, test_count


def _assign_random(table, seed: int, valid_ratio: float, test_ratio: float):
    pd = _require_pandas()
    indices = _shuffle(list(table.index), seed)
    valid_count, test_count = _cut_counts(len(indices), valid_ratio, test_ratio)
    test_indices = set(indices[:test_count])
    valid_indices = set(indices[test_count : test_count + valid_count])

    split = pd.Series("train", index=table.index, dtype="object")
    split.loc[list(valid_indices)] = "valid"
    split.loc[list(test_indices)] = "test"
    return split


def _assign_cold_entity(table, entity_column: str, seed: int, valid_ratio: float, test_ratio: float):
    pd = _require_pandas()
    entities = _shuffle([str(value) for value in table[entity_column].unique()], seed)
    valid_count, test_count = _cut_counts(len(entities), valid_ratio, test_ratio)
    test_entities = set(entities[:test_count])
    valid_entities = set(entities[test_count : test_count + valid_count])

    split = pd.Series("train", index=table.index, dtype="object")
    entity_values = table[entity_column].astype(str)
    split.loc[entity_values.isin(valid_entities)] = "valid"
    split.loc[entity_values.isin(test_entities)] = "test"
    return split


def make_split_table(
    table,
    split: SplitName,
    *,
    seed: int = 42,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """Return a copy of `table` with a `split` column."""
    _validate_ratios(valid_ratio, test_ratio)
    result = table.copy()

    if split == "random":
        result["split"] = _assign_random(result, seed, valid_ratio, test_ratio)
    elif split == "cold-drug":
        result["split"] = _assign_cold_entity(result, "drug_id", seed, valid_ratio, test_ratio)
    elif split == "cold-target":
        result["split"] = _assign_cold_entity(result, "target_id", seed, valid_ratio, test_ratio)
    else:
        raise ValueError(f"Unsupported split: {split}")

    assert_no_leakage(result, split)
    return result


def assert_no_leakage(split_table, split: str) -> None:
    """Raise when cold split entities overlap across train/valid/test."""
    if split not in {"cold-drug", "cold-target"}:
        return

    entity_column = "drug_id" if split == "cold-drug" else "target_id"
    entity_sets = {
        name: set(split_table.loc[split_table["split"] == name, entity_column].astype(str))
        for name in ["train", "valid", "test"]
    }
    overlaps = {
        "train_valid": entity_sets["train"] & entity_sets["valid"],
        "train_test": entity_sets["train"] & entity_sets["test"],
        "valid_test": entity_sets["valid"] & entity_sets["test"],
    }
    leaked = {name: sorted(values) for name, values in overlaps.items() if values}
    if leaked:
        raise ValueError(f"{split} leakage detected for {entity_column}: {leaked}")


def summarize_split(split_table, split: str, seed: int) -> SplitMetadata:
    counts = split_table["split"].value_counts()
    subset = {name: split_table.loc[split_table["split"] == name] for name in ["train", "valid", "test"]}
    return SplitMetadata(
        split=split,
        seed=seed,
        train_rows=int(counts.get("train", 0)),
        valid_rows=int(counts.get("valid", 0)),
        test_rows=int(counts.get("test", 0)),
        train_drugs=int(subset["train"]["drug_id"].nunique()),
        valid_drugs=int(subset["valid"]["drug_id"].nunique()),
        test_drugs=int(subset["test"]["drug_id"].nunique()),
        train_targets=int(subset["train"]["target_id"].nunique()),
        valid_targets=int(subset["valid"]["target_id"].nunique()),
        test_targets=int(subset["test"]["target_id"].nunique()),
        leakage_checked=split in {"cold-drug", "cold-target"},
    )


def write_split(
    table,
    output_csv: str | Path,
    metadata_json: str | Path,
    split: SplitName,
    *,
    seed: int = 42,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitMetadata:
    split_table = make_split_table(
        table,
        split,
        seed=seed,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    metadata = summarize_split(split_table, split, seed)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_table.to_csv(output_path, index=False)

    metadata_path = Path(metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _flatten_train_folds(value) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("Official train folds must be a list")
    if not value:
        return []
    if all(isinstance(item, list) for item in value):
        return [int(index) for fold in value for index in fold]
    return [int(index) for index in value]


def load_official_deepdta_indices(folds_dir: str | Path) -> tuple[list[int], list[int]]:
    folds_path = Path(folds_dir)
    train_raw = ast.literal_eval((folds_path / "train_fold_setting1.txt").read_text(encoding="utf-8"))
    test_raw = ast.literal_eval((folds_path / "test_fold_setting1.txt").read_text(encoding="utf-8"))
    train_indices = _flatten_train_folds(train_raw)
    test_indices = [int(index) for index in test_raw]
    return train_indices, test_indices


def make_official_deepdta_split_table(table, folds_dir: str | Path, *, valid_fold: int = 0):
    """Create the DeepDTA setting1 split from official fold files.

    The official train file contains five train/validation folds. We reserve
    `valid_fold` as validation and use the remaining folds as training.
    """
    pd = _require_pandas()
    folds_path = Path(folds_dir)
    train_raw = ast.literal_eval((folds_path / "train_fold_setting1.txt").read_text(encoding="utf-8"))
    test_raw = ast.literal_eval((folds_path / "test_fold_setting1.txt").read_text(encoding="utf-8"))
    if not isinstance(train_raw, list) or not all(isinstance(item, list) for item in train_raw):
        raise ValueError("Expected train_fold_setting1.txt to contain a list of folds")
    if valid_fold < 0 or valid_fold >= len(train_raw):
        raise ValueError(f"valid_fold must be in [0, {len(train_raw) - 1}]")

    valid_indices = set(int(index) for index in train_raw[valid_fold])
    train_indices = set(
        int(index)
        for fold_idx, fold in enumerate(train_raw)
        if fold_idx != valid_fold
        for index in fold
    )
    test_indices = set(int(index) for index in test_raw)

    all_indices = train_indices | valid_indices | test_indices
    if len(all_indices) != len(train_indices) + len(valid_indices) + len(test_indices):
        raise ValueError("Official split indices overlap")
    if max(all_indices) >= len(table):
        raise ValueError("Official split index exceeds table length")

    result = table.copy()
    result["split"] = pd.Series("unused", index=result.index, dtype="object")
    result.loc[list(train_indices), "split"] = "train"
    result.loc[list(valid_indices), "split"] = "valid"
    result.loc[list(test_indices), "split"] = "test"
    return result


def recommended_mambatransdta_valid_fold(dataset: str) -> int:
    """Return the DeepDTA validation fold that matches MambaTransDTA Table 1 counts."""
    normalized = dataset.lower()
    if normalized == "davis":
        return 1
    if normalized == "kiba":
        return 0
    raise ValueError(f"No MambaTransDTA Table 1 fold recommendation for dataset={dataset!r}")
