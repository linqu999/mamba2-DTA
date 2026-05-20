import pandas as pd
import pytest

from bimamba2_proteindta.data.collate import DTACollator
from bimamba2_proteindta.data.datasets import DTADataset
from bimamba2_proteindta.data.preprocess import REQUIRED_COLUMNS


def make_split_csv(path) -> None:
    table = pd.DataFrame(
        [
            ["d1", "t1", "CCO", "ACDE", 7.1, "toy", "train"],
            ["d2", "t2", "CCN", "MNPQRST", 6.8, "toy", "valid"],
            ["d3", "t3", "CCC", "GGGG", 5.2, "toy", "test"],
        ],
        columns=REQUIRED_COLUMNS + ["split"],
    )
    table.to_csv(path, index=False)


def test_dta_dataset_filters_split(tmp_path) -> None:
    csv_path = tmp_path / "split.csv"
    make_split_csv(csv_path)

    dataset = DTADataset(csv_path, split="train")

    assert len(dataset) == 1
    assert dataset[0]["drug_id"] == "d1"
    assert dataset[0]["affinity"] == pytest.approx(7.1)


def test_collator_builds_tensor_batch_when_torch_is_available(tmp_path) -> None:
    pytest.importorskip("torch")
    csv_path = tmp_path / "split.csv"
    make_split_csv(csv_path)
    dataset = DTADataset(csv_path)
    collator = DTACollator(max_smiles_len=5, max_fasta_len=6)

    batch = collator([dataset[0], dataset[1]])

    assert batch["smiles_input_ids"].shape == (2, 5)
    assert batch["protein_input_ids"].shape == (2, 6)
    assert batch["protein_attention_mask"].tolist() == [
        [True, True, True, True, False, False],
        [True, True, True, True, True, True],
    ]
    assert batch["y"].shape == (2,)
