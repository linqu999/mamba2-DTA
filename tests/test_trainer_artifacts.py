import json
import sys

import pytest

from bimamba2_proteindta.training.trainer import prepare_run_dir


def test_prepare_run_dir_writes_basic_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["scripts/03_train.py", "--config", "debug.yaml"])
    config = {
        "run_root": str(tmp_path),
        "run_id": "unit_run",
        "dataset": "toy",
        "split": "official-deepdta",
        "model": "mtdta_cnn",
        "seed": 42,
    }

    run_dir = prepare_run_dir(config)

    assert run_dir.name == "unit_run"
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["dataset"] == "toy"
    assert (run_dir / "command.txt").read_text(encoding="utf-8").strip()
    assert "python=" in (run_dir / "environment.txt").read_text(encoding="utf-8")


def test_prepare_run_dir_refuses_to_overwrite(tmp_path) -> None:
    config = {"run_root": str(tmp_path), "run_id": "unit_run"}
    prepare_run_dir(config)

    with pytest.raises(FileExistsError):
        prepare_run_dir(config)
