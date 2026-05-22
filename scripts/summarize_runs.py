"""Create a summary table from training run directories."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from bimamba2_proteindta.evaluation.run_summary import collect_run_summaries, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs", help="Directory containing run subdirectories.")
    parser.add_argument("--output", default="results/tables/run_summary.csv", help="Output CSV path.")
    parser.add_argument("--json-output", default="results/tables/run_summary.json", help="Optional output JSON path.")
    parser.add_argument("--include-incomplete", action="store_true", help="Include runs without metrics.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_run_summaries(args.runs_root, include_incomplete=args.include_incomplete)
    write_summary(rows, args.output, args.json_output)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "output": args.output,
                "json_output": args.json_output,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
