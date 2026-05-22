import json
import sys

import pytest

from bimamba2_proteindta.training.trainer import EPOCH_METRIC_FIELDS, prepare_run_dir, write_artifact_manifest, write_epoch_metrics


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


def test_epoch_metrics_writer_includes_full_validation_metrics(tmp_path) -> None:
    output = tmp_path / "metrics.csv"

    write_epoch_metrics(
        output,
        [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "valid_loss": 0.8,
                "valid_mse": 0.8,
                "valid_rmse": 0.9,
                "valid_mae": 0.7,
                "valid_ci": 0.6,
                "valid_rm2": 0.5,
                "is_best": True,
            }
        ],
    )

    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == EPOCH_METRIC_FIELDS


def test_artifact_manifest_records_files(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text('{"ok": true}\n', encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"checkpoint")

    write_artifact_manifest(run_dir)

    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in manifest["artifacts"]}
    assert {"metrics.json", "best.pt"}.issubset(paths)
    assert all(item["sha256"] for item in manifest["artifacts"])
