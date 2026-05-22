import json
import sys

from bimamba2_proteindta.evaluation.run_summary import collect_run_summaries, summarize_run, write_summary


def make_run(tmp_path):
    run_dir = tmp_path / "20260522_test_davis_table1_cnn_seed42"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_name": "davis_table1_cnn",
                "dataset": "davis",
                "split": "mambatransdta-table1",
                "model": "mtdta_cnn",
                "seed": 42,
                "device": "cuda",
                "epochs": 100,
                "batch_size": 64,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train_loss": 0.2,
                "valid_loss": 0.3,
                "best_valid_loss": 0.28,
                "best_epoch": 93,
                "valid": {"mse": 0.28, "rmse": 0.53, "mae": 0.35, "ci": 0.87, "rm2": 0.59},
                "test_loss": 0.29,
                "test": {"mse": 0.30, "rmse": 0.55, "mae": 0.36, "ci": 0.87, "rm2": 0.56},
                "train_rows": 20037,
                "valid_rows": 5009,
                "test_rows": 5010,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "selection_metric": "valid_loss",
                "best_epoch": 93,
                "best_valid_loss": 0.28,
                "best_valid": {"mse": 0.27, "rmse": 0.52, "mae": 0.34, "ci": 0.88, "rm2": 0.60},
                "final_valid": {"mse": 0.31, "rmse": 0.56, "mae": 0.37, "ci": 0.86, "rm2": 0.54},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text("commit=abc123\nbranch=main\ndirty=False\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text("python=3.10.16\ntorch=2.6.0\ncuda_device_0=RTX 3090\n", encoding="utf-8")
    (run_dir / "config_source.txt").write_text("configs/experiment/debug.yaml\n", encoding="utf-8")
    (run_dir / "command.txt").write_text("python scripts/03_train.py --config debug.yaml\n", encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "predictions_valid.csv").write_text("y_true,y_pred\n", encoding="utf-8")
    (run_dir / "predictions_valid_best.csv").write_text("y_true,y_pred\n", encoding="utf-8")
    (run_dir / "predictions_test.csv").write_text("y_true,y_pred\n", encoding="utf-8")
    (run_dir / "artifact_manifest.json").write_text('{"artifacts": []}\n', encoding="utf-8")
    (run_dir / "artifact_validation.json").write_text('{"ok": true, "errors": []}\n', encoding="utf-8")
    return run_dir


def test_summarize_run_flattens_metrics(tmp_path) -> None:
    run_dir = make_run(tmp_path)

    row = summarize_run(run_dir)

    assert row["status"] == "complete"
    assert row["dataset"] == "davis"
    assert row["split"] == "mambatransdta-table1"
    assert row["valid_mse"] == 0.27
    assert row["final_valid_mse"] == 0.31
    assert row["test_mse"] == 0.30
    assert row["test_ci"] == 0.87
    assert row["git_commit"] == "abc123"
    assert row["selection_metric"] == "valid_loss"
    assert row["has_predictions_test"] is True
    assert row["has_artifact_manifest"] is True
    assert row["has_artifact_validation"] is True
    assert row["artifact_validation_ok"] is True


def test_collect_and_write_summary(tmp_path) -> None:
    make_run(tmp_path)
    rows = collect_run_summaries(tmp_path)
    output = tmp_path / "summary.csv"
    output_json = tmp_path / "summary.json"

    write_summary(rows, output, output_json)

    assert len(rows) == 1
    assert "test_mse" in output.read_text(encoding="utf-8")
    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["best_epoch"] == 93
