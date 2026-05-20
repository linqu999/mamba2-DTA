"""Normalize and validate DTA dataset tables."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ["drug_id", "target_id", "smiles", "protein_sequence", "affinity", "dataset"]


@dataclass(frozen=True)
class DatasetStats:
    dataset: str
    num_rows: int
    num_drugs: int
    num_targets: int
    affinity_min: float
    affinity_max: float
    smiles_length_min: int
    smiles_length_median: float
    smiles_length_max: int
    protein_length_min: int
    protein_length_median: float
    protein_length_max: int


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for dataset preprocessing") from exc
    return pd


def validate_normalized_columns(columns: Iterable[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in set(columns)]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_normalized_table(path: str | Path):
    pd = _require_pandas()
    table = pd.read_csv(path)
    validate_normalized_columns(table.columns)
    return table[REQUIRED_COLUMNS].copy()


def validate_normalized_table(table) -> None:
    validate_normalized_columns(table.columns)
    if table.empty:
        raise ValueError("Dataset table is empty")

    null_columns = [column for column in REQUIRED_COLUMNS if table[column].isna().any()]
    if null_columns:
        raise ValueError(f"Columns contain missing values: {', '.join(null_columns)}")

    for column in ["drug_id", "target_id", "smiles", "protein_sequence", "dataset"]:
        empty_mask = table[column].astype(str).str.len() == 0
        if empty_mask.any():
            raise ValueError(f"Column {column} contains empty strings")

    try:
        table["affinity"].astype(float)
    except ValueError as exc:
        raise ValueError("Column affinity must be numeric") from exc


def compute_dataset_stats(table) -> DatasetStats:
    validate_normalized_table(table)
    dataset_values = sorted(str(value) for value in table["dataset"].unique())
    dataset_name = dataset_values[0] if len(dataset_values) == 1 else "+".join(dataset_values)
    smiles_lengths = table["smiles"].astype(str).str.len()
    protein_lengths = table["protein_sequence"].astype(str).str.len()
    affinity = table["affinity"].astype(float)

    return DatasetStats(
        dataset=dataset_name,
        num_rows=int(len(table)),
        num_drugs=int(table["drug_id"].nunique()),
        num_targets=int(table["target_id"].nunique()),
        affinity_min=float(affinity.min()),
        affinity_max=float(affinity.max()),
        smiles_length_min=int(smiles_lengths.min()),
        smiles_length_median=float(smiles_lengths.median()),
        smiles_length_max=int(smiles_lengths.max()),
        protein_length_min=int(protein_lengths.min()),
        protein_length_median=float(protein_lengths.median()),
        protein_length_max=int(protein_lengths.max()),
    )


def normalize_csv(input_path: str | Path, output_path: str | Path, metadata_path: str | Path | None = None) -> DatasetStats:
    """Validate a CSV already matching the project schema and write a clean copy."""
    table = load_normalized_table(input_path)
    validate_normalized_table(table)
    stats = compute_dataset_stats(table)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    if metadata_path is not None:
        metadata = Path(metadata_path)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return stats


def _load_json_dict(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(key): str(value) for key, value in data.items()}


def _load_numpy_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _davis_kd_to_pkd(value: float) -> float:
    return -math.log10(value / 1e9)


def load_deepdta_table(raw_dataset_dir: str | Path, dataset: str):
    """Load DeepDTA raw files into the project normalized schema.

    Davis labels are converted from Kd nM to pKd, matching common DeepDTA
    preprocessing. KIBA labels are kept as KIBA scores.
    """
    pd = _require_pandas()
    raw_dir = Path(raw_dataset_dir)
    ligands = _load_json_dict(raw_dir / "ligands_can.txt")
    proteins = _load_json_dict(raw_dir / "proteins.txt")
    y = _load_numpy_pickle(raw_dir / "Y")

    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())
    expected_shape = (len(drug_ids), len(target_ids))
    if tuple(y.shape) != expected_shape:
        raise ValueError(f"Y shape mismatch for {dataset}: expected {expected_shape}, got {tuple(y.shape)}")

    rows = []
    for drug_idx, drug_id in enumerate(drug_ids):
        for target_idx, target_id in enumerate(target_ids):
            value = float(y[drug_idx, target_idx])
            if math.isnan(value):
                continue
            affinity = _davis_kd_to_pkd(value) if dataset == "davis" else value
            rows.append(
                {
                    "drug_id": str(drug_id),
                    "target_id": str(target_id),
                    "smiles": ligands[drug_id],
                    "protein_sequence": proteins[target_id],
                    "affinity": affinity,
                    "dataset": dataset,
                }
            )

    table = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    validate_normalized_table(table)
    return table


def prepare_deepdta_dataset(
    raw_dataset_dir: str | Path,
    dataset: str,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
) -> DatasetStats:
    table = load_deepdta_table(raw_dataset_dir, dataset)
    stats = compute_dataset_stats(table)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    if metadata_path is not None:
        metadata = Path(metadata_path)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats
