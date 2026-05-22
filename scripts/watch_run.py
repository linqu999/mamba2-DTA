"""Watch the latest or selected training run."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Run directory to watch. Defaults to the newest directory under --runs-root.")
    parser.add_argument("--runs-root", default="runs", help="Directory containing run subdirectories.")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds.")
    parser.add_argument("--tail", type=int, default=12, help="Number of train.log lines to show.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between refreshes.")
    parser.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi output.")
    return parser.parse_args()


def latest_run(runs_root: str | Path) -> Path:
    root = Path(runs_root)
    runs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not runs:
        raise SystemExit(f"No run directories found under {root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-count:]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def latest_epoch_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def file_line(path: Path) -> str:
    if not path.exists():
        return f"missing {path.name}"
    size = path.stat().st_size
    return f"{path.name}: {size / 1024:.1f} KiB"


def nvidia_smi_head() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found"
    try:
        completed = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True, timeout=5)
    except subprocess.SubprocessError as exc:
        return f"nvidia-smi unavailable: {exc}"
    return "\n".join(completed.stdout.splitlines()[:15])


def print_dashboard(run_dir: Path, tail: int, show_gpu: bool) -> None:
    summary = load_json(run_dir / "metrics_summary.json")
    metrics = load_json(run_dir / "metrics.json")
    latest = latest_epoch_row(run_dir / "metrics.csv")

    print(f"RUN_DIR={run_dir}")
    print()
    if latest:
        fields = ["epoch", "train_loss", "valid_mse", "valid_ci", "valid_rm2", "is_best"]
        print("latest: " + " ".join(f"{key}={latest.get(key, '')}" for key in fields))
    if summary:
        best = summary.get("best_valid", {})
        print(
            "best: "
            f"epoch={summary.get('best_epoch')} "
            f"valid_mse={best.get('mse')} "
            f"valid_ci={best.get('ci')} "
            f"valid_rm2={best.get('rm2')}"
        )
    elif metrics:
        valid = metrics.get("valid", {})
        print(
            "finalized: "
            f"best_epoch={metrics.get('best_epoch')} "
            f"valid_mse={valid.get('mse')} "
            f"valid_ci={valid.get('ci')} "
            f"valid_rm2={valid.get('rm2')}"
        )

    print()
    for line in tail_lines(run_dir / "train.log", tail):
        print(line)

    print()
    for name in [
        "best.pt",
        "metrics.csv",
        "metrics.json",
        "metrics_summary.json",
        "predictions_valid.csv",
        "predictions_valid_best.csv",
        "predictions_test.csv",
        "artifact_manifest.json",
    ]:
        print(file_line(run_dir / name))

    if show_gpu:
        print()
        print(nvidia_smi_head())


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run(args.runs_root)
    while True:
        if not args.no_clear:
            print("\033c", end="")
        print_dashboard(run_dir, args.tail, show_gpu=not args.no_gpu)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
