import json
import pickle

import numpy as np
import pandas as pd
import pytest

from bimamba2_proteindta.data.preprocess import (
    REQUIRED_COLUMNS,
    compute_dataset_stats,
    load_deepdta_table,
    normalize_csv,
    validate_normalized_table,
)


def sample_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["d1", "t1", "CCO", "ACDE", 7.1, "toy"],
            ["d2", "t1", "CCN", "ACDEFG", 6.8, "toy"],
            ["d1", "t2", "CCC", "MNPQ", 5.2, "toy"],
        ],
        columns=REQUIRED_COLUMNS,
    )


def test_compute_dataset_stats() -> None:
    stats = compute_dataset_stats(sample_table())

    assert stats.dataset == "toy"
    assert stats.num_rows == 3
    assert stats.num_drugs == 2
    assert stats.num_targets == 2
    assert stats.protein_length_max == 6


def test_validate_normalized_table_rejects_missing_column() -> None:
    table = sample_table().drop(columns=["smiles"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_normalized_table(table)


def test_normalize_csv_writes_clean_csv_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "processed.csv"
    metadata_path = tmp_path / "metadata.json"
    sample_table().to_csv(input_path, index=False)

    normalize_csv(input_path, output_path, metadata_path)

    assert output_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["num_rows"] == 3


def test_load_deepdta_table_converts_davis_kd_to_pkd(tmp_path) -> None:
    raw_dir = tmp_path / "davis"
    raw_dir.mkdir()
    (raw_dir / "ligands_can.txt").write_text(json.dumps({"d1": "CCO"}), encoding="utf-8")
    (raw_dir / "proteins.txt").write_text(json.dumps({"t1": "ACDE", "t2": "MNPQ"}), encoding="utf-8")
    with (raw_dir / "Y").open("wb") as handle:
        pickle.dump(np.array([[10000.0, 10.0]]), handle)

    table = load_deepdta_table(raw_dir, "davis")

    assert table["affinity"].tolist() == pytest.approx([5.0, 8.0])


def test_load_deepdta_table_filters_kiba_nan(tmp_path) -> None:
    raw_dir = tmp_path / "kiba"
    raw_dir.mkdir()
    (raw_dir / "ligands_can.txt").write_text(json.dumps({"d1": "CCO", "d2": "CCC"}), encoding="utf-8")
    (raw_dir / "proteins.txt").write_text(json.dumps({"t1": "ACDE"}), encoding="utf-8")
    with (raw_dir / "Y").open("wb") as handle:
        pickle.dump(np.array([[12.3], [np.nan]]), handle)

    table = load_deepdta_table(raw_dir, "kiba")

    assert len(table) == 1
    assert table.iloc[0]["affinity"] == pytest.approx(12.3)
