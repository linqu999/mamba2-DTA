import pandas as pd
import pytest

from bimamba2_proteindta.data.preprocess import REQUIRED_COLUMNS
from bimamba2_proteindta.data.splits import (
    assert_no_leakage,
    make_official_deepdta_split_table,
    make_split_table,
    recommended_mambatransdta_valid_fold,
    summarize_split,
    write_split,
)


def sample_split_table() -> pd.DataFrame:
    rows = []
    for drug_idx in range(6):
        for target_idx in range(4):
            rows.append(
                [
                    f"d{drug_idx}",
                    f"t{target_idx}",
                    "CCO",
                    "ACDEFG",
                    float(drug_idx + target_idx),
                    "toy",
                ]
            )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_random_split_assigns_all_rows() -> None:
    split_table = make_split_table(sample_split_table(), "random", seed=7, valid_ratio=0.2, test_ratio=0.2)

    assert set(split_table["split"]) == {"train", "valid", "test"}
    assert len(split_table) == len(sample_split_table())


def test_cold_drug_split_has_no_drug_leakage() -> None:
    split_table = make_split_table(sample_split_table(), "cold-drug", seed=7, valid_ratio=0.2, test_ratio=0.2)

    train_drugs = set(split_table.loc[split_table["split"] == "train", "drug_id"])
    test_drugs = set(split_table.loc[split_table["split"] == "test", "drug_id"])
    valid_drugs = set(split_table.loc[split_table["split"] == "valid", "drug_id"])
    assert train_drugs.isdisjoint(test_drugs)
    assert train_drugs.isdisjoint(valid_drugs)
    assert valid_drugs.isdisjoint(test_drugs)


def test_cold_target_split_has_no_target_leakage() -> None:
    split_table = make_split_table(sample_split_table(), "cold-target", seed=7, valid_ratio=0.25, test_ratio=0.25)

    train_targets = set(split_table.loc[split_table["split"] == "train", "target_id"])
    test_targets = set(split_table.loc[split_table["split"] == "test", "target_id"])
    valid_targets = set(split_table.loc[split_table["split"] == "valid", "target_id"])
    assert train_targets.isdisjoint(test_targets)
    assert train_targets.isdisjoint(valid_targets)
    assert valid_targets.isdisjoint(test_targets)


def test_assert_no_leakage_detects_bad_cold_split() -> None:
    table = sample_split_table().iloc[:3].copy()
    table["split"] = ["train", "valid", "test"]
    table["drug_id"] = ["same", "same", "other"]

    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(table, "cold-drug")


def test_write_split_outputs_csv_and_metadata(tmp_path) -> None:
    output_csv = tmp_path / "split.csv"
    metadata_json = tmp_path / "metadata.json"

    metadata = write_split(
        sample_split_table(),
        output_csv,
        metadata_json,
        "cold-drug",
        seed=7,
        valid_ratio=0.2,
        test_ratio=0.2,
    )

    assert output_csv.exists()
    assert metadata_json.exists()
    assert metadata == summarize_split(pd.read_csv(output_csv), "cold-drug", 7)


def test_make_official_deepdta_split_table_uses_validation_fold(tmp_path) -> None:
    table = sample_split_table().iloc[:8].copy().reset_index(drop=True)
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    (folds_dir / "train_fold_setting1.txt").write_text("[[0, 1], [2, 3], [4, 5]]", encoding="utf-8")
    (folds_dir / "test_fold_setting1.txt").write_text("[6, 7]", encoding="utf-8")

    split_table = make_official_deepdta_split_table(table, folds_dir, valid_fold=1)

    assert split_table["split"].tolist() == [
        "train",
        "train",
        "valid",
        "valid",
        "train",
        "train",
        "test",
        "test",
    ]


def test_mambatransdta_table1_fold_recommendations() -> None:
    assert recommended_mambatransdta_valid_fold("davis") == 1
    assert recommended_mambatransdta_valid_fold("kiba") == 0
