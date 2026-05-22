"""Create random, cold-start, and MambaTransDTA Table 1 splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from bimamba2_proteindta.data.preprocess import load_normalized_table, validate_normalized_table
from bimamba2_proteindta.data.splits import (
    make_official_deepdta_split_table,
    recommended_mambatransdta_valid_fold,
    summarize_split,
    write_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normalized dataset CSV.")
    parser.add_argument("--output", required=True, help="Output split CSV.")
    parser.add_argument("--metadata", required=True, help="Output split metadata JSON.")
    parser.add_argument(
        "--split",
        choices=["random", "cold-drug", "cold-target", "mambatransdta-table1"],
        required=True,
    )
    parser.add_argument("--folds-dir", help="DeepDTA folds directory for the MambaTransDTA Table 1 split.")
    parser.add_argument("--valid-fold", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = load_normalized_table(args.input)
    validate_normalized_table(table)
    metadata_dict: dict
    if args.split == "mambatransdta-table1":
        if not args.folds_dir:
            raise SystemExit("--folds-dir is required for the MambaTransDTA Table 1 split")
        valid_fold = args.valid_fold
        if valid_fold is None:
            valid_fold = recommended_mambatransdta_valid_fold(str(table["dataset"].iloc[0]))
        split_table = make_official_deepdta_split_table(table, args.folds_dir, valid_fold=valid_fold)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        split_table.to_csv(args.output, index=False)
        metadata = summarize_split(split_table, args.split, args.seed)
        metadata_dict = asdict(metadata)
        metadata_dict["valid_fold"] = valid_fold

        Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata).write_text(json.dumps(metadata_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        metadata = write_split(
            table,
            args.output,
            args.metadata,
            args.split,
            seed=args.seed,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
        )
        metadata_dict = asdict(metadata)
    print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
