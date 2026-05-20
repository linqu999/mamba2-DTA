"""Dataset classes for DTA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bimamba2_proteindta.data.preprocess import REQUIRED_COLUMNS, validate_normalized_columns


@dataclass(frozen=True)
class DTASample:
    drug_id: str
    target_id: str
    smiles: str
    protein_sequence: str
    affinity: float
    dataset: str
    split: str | None = None


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for dataset loading") from exc
    return pd


class DTADataset:
    """A lightweight map-style dataset backed by a normalized CSV file."""

    def __init__(self, csv_path: str | Path, split: str | None = None, limit_rows: int | None = None) -> None:
        pd = _require_pandas()
        self.csv_path = Path(csv_path)
        table = pd.read_csv(self.csv_path)
        validate_normalized_columns(table.columns)
        if split is not None:
            if "split" not in table.columns:
                raise ValueError(f"{csv_path} has no split column")
            table = table.loc[table["split"] == split].copy()
        if limit_rows is not None:
            table = table.head(limit_rows).copy()
        if table.empty:
            raise ValueError(f"No rows available for split={split!r} in {csv_path}")

        self.table = table.reset_index(drop=True)
        self.split = split

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        sample = {
            "drug_id": str(row["drug_id"]),
            "target_id": str(row["target_id"]),
            "smiles": str(row["smiles"]),
            "protein_sequence": str(row["protein_sequence"]),
            "affinity": float(row["affinity"]),
            "dataset": str(row["dataset"]),
        }
        if "split" in row.index:
            sample["split"] = str(row["split"])
        return sample


def default_split_path(dataset: str, split: str) -> Path:
    if split == "official-deepdta":
        filename = "official_deepdta.csv"
    elif split == "cold-drug":
        filename = "cold_drug_seed42.csv"
    elif split == "cold-target":
        filename = "cold_target_seed42.csv"
    elif split == "random":
        filename = "official_deepdta.csv"
    else:
        raise ValueError(f"Unsupported split name: {split}")
    return Path("data") / "splits" / dataset / filename
