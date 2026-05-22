from scripts.watch_run import current_best_epoch_row, latest_epoch_row, latest_run, tail_lines


def test_watch_helpers_read_latest_run_and_metrics(tmp_path) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    (newer / "metrics.csv").write_text(
        "epoch,train_loss,valid_mse,valid_ci,valid_rm2,is_best\n"
        "1,1.0,0.8,0.6,0.5,True\n"
        "2,0.9,0.7,0.7,0.6,False\n",
        encoding="utf-8",
    )
    (newer / "config.json").write_text('{"epochs": 2}\n', encoding="utf-8")
    (newer / "artifact_validation.json").write_text('{"ok": true, "errors": [], "warnings": []}\n', encoding="utf-8")
    (newer / "train.log").write_text("a\nb\nc\n", encoding="utf-8")

    assert latest_run(tmp_path) == newer
    assert latest_epoch_row(newer / "metrics.csv")["epoch"] == "2"
    assert current_best_epoch_row(newer / "metrics.csv")["epoch"] == "1"
    assert tail_lines(newer / "train.log", 2) == ["b", "c"]
