import csv
import json

from bimamba2_proteindta.evaluation.artifacts import validate_run_artifacts, write_validation_report
from bimamba2_proteindta.training.trainer import EPOCH_METRIC_FIELDS, write_artifact_manifest


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_complete_run(tmp_path, *, limit_batches=None, prediction_rows=2):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "run_name": "unit",
        "dataset": "davis",
        "split": "mambatransdta-table1",
        "model": "mtdta_cnn",
        "epochs": 2,
        "batch_size": 4,
    }
    metrics = {
        "train_loss": 0.4,
        "valid_loss": 0.5,
        "best_valid_loss": 0.45,
        "best_epoch": 1,
        "valid": {"mse": 0.45, "rmse": 0.67, "mae": 0.4, "ci": 0.8, "rm2": 0.6},
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "valid_loss": 0.45,
                "valid_mse": 0.45,
                "valid_rmse": 0.67,
                "valid_mae": 0.4,
                "valid_ci": 0.8,
                "valid_rm2": 0.6,
                "is_best": True,
            },
            {
                "epoch": 2,
                "train_loss": 0.4,
                "valid_loss": 0.5,
                "valid_mse": 0.5,
                "valid_rmse": 0.71,
                "valid_mae": 0.45,
                "valid_ci": 0.75,
                "valid_rm2": 0.55,
                "is_best": False,
            },
        ],
        "train_rows": 3,
        "valid_rows": 2,
        "test_rows": 1,
        "test_loss": 0.6,
        "test": {"mse": 0.6, "rmse": 0.77, "mae": 0.5, "ci": 0.7, "rm2": 0.5},
        "limit_batches": limit_batches,
    }
    summary = {
        "selection_metric": "valid_loss",
        "selection_mode": "min",
        "best_epoch": 1,
        "best_valid_loss": 0.45,
        "best_valid": {"mse": 0.45, "rmse": 0.67, "mae": 0.4, "ci": 0.8, "rm2": 0.6},
        "final_epoch": 2,
        "final_train_loss": 0.4,
        "final_valid_loss": 0.5,
        "final_valid": {"mse": 0.5, "rmse": 0.71, "mae": 0.45, "ci": 0.75, "rm2": 0.55},
    }

    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "config_source.txt").write_text("configs/experiment/unit.yaml\n", encoding="utf-8")
    (run_dir / "command.txt").write_text("python scripts/03_train.py --config unit.yaml\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text("python=3.10.16\ntorch=2.6.0\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text("commit=abc\nbranch=main\ndirty=False\n", encoding="utf-8")
    (run_dir / "train.log").write_text("epoch=1\nepoch=2\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_csv(run_dir / "metrics.csv", EPOCH_METRIC_FIELDS, metrics["history"])
    (run_dir / "best.pt").write_bytes(b"checkpoint")

    prediction_template = {
        "y_true": 1.0,
        "y_pred": 1.1,
        "drug_id": "drug",
        "target_id": "target",
        "sequence_length": 10,
    }
    _write_csv(
        run_dir / "predictions_valid.csv",
        ["y_true", "y_pred", "drug_id", "target_id", "sequence_length"],
        [prediction_template for _ in range(prediction_rows)],
    )
    _write_csv(
        run_dir / "predictions_valid_best.csv",
        ["y_true", "y_pred", "drug_id", "target_id", "sequence_length"],
        [prediction_template for _ in range(prediction_rows)],
    )
    _write_csv(
        run_dir / "predictions_test.csv",
        ["y_true", "y_pred", "drug_id", "target_id", "sequence_length"],
        [prediction_template],
    )
    write_artifact_manifest(run_dir)
    return run_dir


def test_validate_run_artifacts_accepts_complete_run(tmp_path) -> None:
    run_dir = make_complete_run(tmp_path)

    report = validate_run_artifacts(run_dir)

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["metrics_csv_rows"] == 2
    assert report["prediction_rows"]["predictions_test.csv"] == 1


def test_validate_run_artifacts_fails_on_missing_required_file(tmp_path) -> None:
    run_dir = make_complete_run(tmp_path)
    (run_dir / "best.pt").unlink()

    report = validate_run_artifacts(run_dir)

    assert report["ok"] is False
    assert "best.pt" in report["missing_files"]


def test_validate_run_artifacts_allows_partial_predictions_for_limit_batches(tmp_path) -> None:
    run_dir = make_complete_run(tmp_path, limit_batches=1, prediction_rows=1)

    report = validate_run_artifacts(run_dir)

    assert report["ok"] is True
    assert any("limit_batches" in warning for warning in report["warnings"])


def test_write_validation_report(tmp_path) -> None:
    run_dir = make_complete_run(tmp_path)

    output = write_validation_report(run_dir)

    assert output.name == "artifact_validation.json"
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def test_validate_run_artifacts_accepts_multiple_best_improvements(tmp_path) -> None:
    run_dir = make_complete_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    metrics["best_epoch"] = 2
    summary["best_epoch"] = 2
    summary["best_valid_loss"] = 0.4
    metrics["best_valid_loss"] = 0.4
    metrics["history"][1]["valid_loss"] = 0.4
    metrics["history"][1]["is_best"] = True
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_csv(run_dir / "metrics.csv", EPOCH_METRIC_FIELDS, metrics["history"])
    write_artifact_manifest(run_dir)

    report = validate_run_artifacts(run_dir)

    assert report["ok"] is True
