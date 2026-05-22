"""Validate the files produced by one or more training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from bimamba2_proteindta.evaluation.artifacts import validate_run_artifacts, write_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", help="Run directory to validate. Can be repeated.")
    parser.add_argument("--runs-root", default="runs", help="Directory containing run subdirectories.")
    parser.add_argument("--latest", action="store_true", help="Validate only the newest run under --runs-root.")
    parser.add_argument("--all", action="store_true", help="Validate every run under --runs-root.")
    parser.add_argument("--write-report", action="store_true", help="Write artifact_validation.json into each run.")
    return parser.parse_args()


def _latest_run(runs_root: str | Path) -> Path:
    root = Path(runs_root)
    runs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not runs:
        raise SystemExit(f"No run directories found under {root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _all_runs(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    return sorted([path for path in root.iterdir() if path.is_dir()]) if root.exists() else []


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    modes = sum(bool(value) for value in [args.run_dir, args.latest, args.all])
    if modes > 1:
        raise SystemExit("Choose only one of --run-dir, --latest, or --all")
    if args.run_dir:
        return [Path(path) for path in args.run_dir]
    if args.all:
        runs = _all_runs(args.runs_root)
        if not runs:
            raise SystemExit(f"No run directories found under {args.runs_root}")
        return runs
    return [_latest_run(args.runs_root)]


def main() -> int:
    args = parse_args()
    reports = []
    for run_dir in resolve_run_dirs(args):
        report = validate_run_artifacts(run_dir)
        if args.write_report:
            write_validation_report(run_dir, report)
        reports.append(report)

    output = reports[0] if len(reports) == 1 else {"runs": len(reports), "reports": reports}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
