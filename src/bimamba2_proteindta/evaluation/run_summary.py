"""Summarize training run directories into analysis-ready tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "run_id",
    "run_dir",
    "status",
    "run_name",
    "dataset",
    "split",
    "model",
    "seed",
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "max_smiles_len",
    "max_fasta_len",
    "train_rows",
    "valid_rows",
    "test_rows",
    "best_epoch",
    "train_loss",
    "valid_loss",
    "best_valid_loss",
    "valid_mse",
    "valid_rmse",
    "valid_mae",
    "valid_ci",
    "valid_rm2",
    "test_loss",
    "test_mse",
    "test_rmse",
    "test_mae",
    "test_ci",
    "test_rm2",
    "has_best_pt",
    "has_predictions_valid",
    "has_predictions_valid_best",
    "has_predictions_test",
    "has_artifact_manifest",
    "selection_metric",
    "final_valid_mse",
    "final_valid_ci",
    "final_valid_rm2",
    "git_commit",
    "git_branch",
    "git_dirty",
    "python",
    "torch",
    "cuda_available",
    "cuda_version",
    "cuda_device_0",
    "config_source",
    "command",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _metric(metrics: dict[str, Any], section: str, name: str) -> Any:
    value = metrics.get(section, {})
    return value.get(name, "") if isinstance(value, dict) else ""


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Return a flat summary row for one run directory."""
    path = Path(run_dir)
    config = _load_json(path / "config.json")
    metrics = _load_json(path / "metrics.json")
    summary = _load_json(path / "metrics_summary.json")
    git = _parse_key_values(path / "git_commit.txt")
    env = _parse_key_values(path / "environment.txt")
    best_valid = summary.get("best_valid", {}) if isinstance(summary.get("best_valid", {}), dict) else {}
    final_valid = summary.get("final_valid", {}) if isinstance(summary.get("final_valid", {}), dict) else {}

    row: dict[str, Any] = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "run_id": path.name,
            "run_dir": str(path),
            "status": "complete" if metrics else "incomplete",
            "run_name": config.get("run_name", ""),
            "dataset": config.get("dataset", ""),
            "split": config.get("split", ""),
            "model": config.get("model", ""),
            "seed": config.get("seed", ""),
            "device": config.get("device", ""),
            "epochs": config.get("epochs", ""),
            "batch_size": config.get("batch_size", ""),
            "learning_rate": config.get("learning_rate", ""),
            "max_smiles_len": config.get("max_smiles_len", ""),
            "max_fasta_len": config.get("max_fasta_len", ""),
            "train_rows": metrics.get("train_rows", ""),
            "valid_rows": metrics.get("valid_rows", ""),
            "test_rows": metrics.get("test_rows", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "train_loss": metrics.get("train_loss", ""),
            "valid_loss": metrics.get("valid_loss", ""),
            "best_valid_loss": metrics.get("best_valid_loss", ""),
            "valid_mse": best_valid.get("mse", _metric(metrics, "valid", "mse")),
            "valid_rmse": best_valid.get("rmse", _metric(metrics, "valid", "rmse")),
            "valid_mae": best_valid.get("mae", _metric(metrics, "valid", "mae")),
            "valid_ci": best_valid.get("ci", _metric(metrics, "valid", "ci")),
            "valid_rm2": best_valid.get("rm2", _metric(metrics, "valid", "rm2")),
            "test_loss": metrics.get("test_loss", ""),
            "test_mse": _metric(metrics, "test", "mse"),
            "test_rmse": _metric(metrics, "test", "rmse"),
            "test_mae": _metric(metrics, "test", "mae"),
            "test_ci": _metric(metrics, "test", "ci"),
            "test_rm2": _metric(metrics, "test", "rm2"),
            "has_best_pt": (path / "best.pt").exists(),
            "has_predictions_valid": (path / "predictions_valid.csv").exists(),
            "has_predictions_valid_best": (path / "predictions_valid_best.csv").exists(),
            "has_predictions_test": (path / "predictions_test.csv").exists(),
            "has_artifact_manifest": (path / "artifact_manifest.json").exists(),
            "selection_metric": summary.get("selection_metric", ""),
            "final_valid_mse": final_valid.get("mse", ""),
            "final_valid_ci": final_valid.get("ci", ""),
            "final_valid_rm2": final_valid.get("rm2", ""),
            "git_commit": git.get("commit", ""),
            "git_branch": git.get("branch", ""),
            "git_dirty": git.get("dirty", ""),
            "python": env.get("python", ""),
            "torch": env.get("torch", ""),
            "cuda_available": env.get("cuda_available", ""),
            "cuda_version": env.get("cuda_version", ""),
            "cuda_device_0": env.get("cuda_device_0", ""),
            "config_source": _read_text(path / "config_source.txt"),
            "command": _read_text(path / "command.txt"),
        }
    )
    return row


def collect_run_summaries(runs_root: str | Path, include_incomplete: bool = False) -> list[dict[str, Any]]:
    root = Path(runs_root)
    if not root.exists():
        return []

    rows = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda item: item.name):
        row = summarize_run(run_dir)
        if include_incomplete or row["status"] == "complete":
            rows.append(row)
    return rows


def write_summary(rows: list[dict[str, Any]], output_csv: str | Path, output_json: str | Path | None = None) -> None:
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
