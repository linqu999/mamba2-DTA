"""Validate required files for completed training runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from bimamba2_proteindta.training.trainer import EPOCH_METRIC_FIELDS


REQUIRED_RUN_FILES = [
    "config.json",
    "config_source.txt",
    "command.txt",
    "environment.txt",
    "git_commit.txt",
    "train.log",
    "metrics.csv",
    "metrics.json",
    "metrics_summary.json",
    "best.pt",
    "predictions_valid.csv",
    "predictions_valid_best.csv",
    "predictions_test.csv",
    "artifact_manifest.json",
]


PREDICTION_FIELDS = ["y_true", "y_pred", "drug_id", "target_id", "sequence_length"]
VALIDATION_REPORT_FILE = "artifact_validation.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_header_and_count(path: Path) -> tuple[list[str], int]:
    if not path.exists():
        return [], 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        return header, sum(1 for _ in reader)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _metric_errors(metrics: dict[str, Any], prefix: str) -> list[str]:
    errors = []
    for name in ["mse", "rmse", "mae", "ci", "rm2"]:
        if name not in metrics:
            errors.append(f"{prefix} missing {name}")
        elif not _is_finite_number(metrics[name]):
            errors.append(f"{prefix}.{name} is not finite")
    return errors


def validate_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Return a validation report for a run directory.

    The report is strict about publication-critical files, while keeping
    row-count mismatches as warnings for debug runs that use `limit_batches`.
    """
    path = Path(run_dir)
    report: dict[str, Any] = {
        "run_dir": str(path),
        "ok": True,
        "missing_files": [],
        "empty_files": [],
        "errors": [],
        "warnings": [],
        "metrics_csv_rows": 0,
        "prediction_rows": {},
        "manifest_missing_entries": [],
    }

    if not path.exists():
        report["ok"] = False
        report["errors"].append("run_dir does not exist")
        return report

    for name in REQUIRED_RUN_FILES:
        file_path = path / name
        if not file_path.exists():
            report["missing_files"].append(name)
        elif file_path.stat().st_size == 0:
            report["empty_files"].append(name)

    config = _load_json(path / "config.json")
    metrics = _load_json(path / "metrics.json")
    metrics_summary = _load_json(path / "metrics_summary.json")

    metrics_header, metrics_rows = _csv_header_and_count(path / "metrics.csv")
    report["metrics_csv_rows"] = metrics_rows
    missing_metric_columns = [field for field in EPOCH_METRIC_FIELDS if field not in metrics_header]
    if missing_metric_columns:
        report["errors"].append(f"metrics.csv missing columns: {missing_metric_columns}")
    if metrics_rows == 0:
        report["errors"].append("metrics.csv has no epoch rows")

    history = metrics.get("history", [])
    if isinstance(history, list) and history and len(history) != metrics_rows:
        report["errors"].append(f"metrics.csv rows ({metrics_rows}) != metrics.json history rows ({len(history)})")

    expected_epochs = config.get("epochs")
    if expected_epochs is not None and metrics_rows and int(expected_epochs) != metrics_rows:
        report["errors"].append(f"metrics.csv rows ({metrics_rows}) != configured epochs ({expected_epochs})")

    epoch_rows = _csv_rows(path / "metrics.csv")
    best_epoch = metrics.get("best_epoch")
    summary_best_epoch = metrics_summary.get("best_epoch")
    if best_epoch is not None and summary_best_epoch is not None and best_epoch != summary_best_epoch:
        report["errors"].append(f"metrics.json best_epoch ({best_epoch}) != metrics_summary.json best_epoch ({summary_best_epoch})")
    best_flags = [row for row in epoch_rows if str(row.get("is_best", "")).lower() in {"true", "1"}]
    if metrics_rows and not best_flags:
        report["errors"].append("metrics.csv should contain at least one best checkpoint epoch")
    if best_epoch is not None and best_flags:
        best_epoch_flags = [row for row in best_flags if str(row.get("epoch")) == str(best_epoch)]
        if not best_epoch_flags:
            report["errors"].append(f"metrics.csv does not mark best_epoch={best_epoch} as a best checkpoint epoch")

    if not metrics_summary.get("best_valid"):
        report["errors"].append("metrics_summary.json missing best_valid metrics")
    elif isinstance(metrics_summary["best_valid"], dict):
        report["errors"].extend(_metric_errors(metrics_summary["best_valid"], "metrics_summary.best_valid"))
    final_valid = metrics_summary.get("final_valid")
    if isinstance(final_valid, dict):
        report["errors"].extend(_metric_errors(final_valid, "metrics_summary.final_valid"))
    if not metrics.get("test"):
        report["errors"].append("metrics.json missing test metrics")
    elif isinstance(metrics["test"], dict):
        report["errors"].extend(_metric_errors(metrics["test"], "metrics.test"))
    if isinstance(metrics.get("valid"), dict):
        report["errors"].extend(_metric_errors(metrics["valid"], "metrics.valid"))

    for name in ["predictions_valid.csv", "predictions_valid_best.csv", "predictions_test.csv"]:
        header, rows = _csv_header_and_count(path / name)
        report["prediction_rows"][name] = rows
        missing_prediction_columns = [field for field in PREDICTION_FIELDS if field not in header]
        if missing_prediction_columns:
            report["errors"].append(f"{name} missing columns: {missing_prediction_columns}")
        if rows == 0:
            report["errors"].append(f"{name} has no prediction rows")

    if metrics.get("limit_batches") is None:
        expected_rows = {
            "predictions_valid.csv": metrics.get("valid_rows"),
            "predictions_valid_best.csv": metrics.get("valid_rows"),
            "predictions_test.csv": metrics.get("test_rows"),
        }
        for name, expected in expected_rows.items():
            actual = report["prediction_rows"].get(name)
            if expected is not None and actual != expected:
                report["errors"].append(f"{name} rows ({actual}) != expected rows ({expected})")
    else:
        report["warnings"].append("limit_batches is set; prediction row counts are expected to be partial")

    manifest = _load_json(path / "artifact_manifest.json")
    manifest_paths = {item.get("path") for item in manifest.get("artifacts", []) if isinstance(item, dict)}
    required_manifest_entries = [name for name in REQUIRED_RUN_FILES if name != "artifact_manifest.json"]
    report["manifest_missing_entries"] = [name for name in required_manifest_entries if name not in manifest_paths]

    if report["manifest_missing_entries"]:
        report["errors"].append(f"artifact_manifest.json missing entries: {report['manifest_missing_entries']}")
    for item in manifest.get("artifacts", []):
        if not isinstance(item, dict):
            continue
        artifact_path = item.get("path", "")
        if not item.get("sha256"):
            report["errors"].append(f"artifact_manifest entry missing sha256: {artifact_path}")
        if not isinstance(item.get("bytes"), int) or item.get("bytes", 0) <= 0:
            report["errors"].append(f"artifact_manifest entry has invalid byte count: {artifact_path}")

    if report["missing_files"] or report["empty_files"] or report["errors"]:
        report["ok"] = False
    return report


def write_validation_report(run_dir: str | Path, report: dict[str, Any] | None = None) -> Path:
    """Write a machine-readable artifact validation report for a run."""
    path = Path(run_dir)
    report = validate_run_artifacts(path) if report is None else report
    output = path / VALIDATION_REPORT_FILE
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
