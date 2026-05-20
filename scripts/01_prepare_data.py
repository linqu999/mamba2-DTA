"""Prepare a normalized DTA CSV and emit dataset statistics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from _bootstrap import add_src_to_path

add_src_to_path()

from bimamba2_proteindta.data.preprocess import normalize_csv, prepare_deepdta_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="CSV with the project schema.")
    parser.add_argument("--deepdta-raw-dir", help="DeepDTA raw dataset directory, e.g. data/raw/deepdta/davis.")
    parser.add_argument("--dataset", choices=["davis", "kiba"], help="Dataset name for DeepDTA raw conversion.")
    parser.add_argument("--output", required=True, help="Path for the cleaned CSV.")
    parser.add_argument("--metadata", required=True, help="Path for dataset statistics JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.deepdta_raw_dir:
        if not args.dataset:
            raise SystemExit("--dataset is required with --deepdta-raw-dir")
        stats = prepare_deepdta_dataset(args.deepdta_raw_dir, args.dataset, args.output, args.metadata)
    else:
        if not args.input:
            raise SystemExit("--input is required unless --deepdta-raw-dir is used")
        stats = normalize_csv(args.input, args.output, args.metadata)
    print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
