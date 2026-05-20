"""Train a DTA model from a config file."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from bimamba2_proteindta.data.collate import DTACollator
from bimamba2_proteindta.data.datasets import DTADataset, default_split_path
from bimamba2_proteindta.models.factory import build_model
from bimamba2_proteindta.training.seed import seed_everything
from bimamba2_proteindta.training.trainer import prepare_run_dir, run_training
from bimamba2_proteindta.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--limit-batches", type=int, help="Override limit_batches from config.")
    parser.add_argument("--device", help="Override device from config.")
    parser.add_argument("--run-id", help="Override generated run_id.")
    return parser.parse_args()


def resolve_split_path(config: dict):
    if "split_path" in config:
        return config["split_path"]
    return default_split_path(str(config["dataset"]), str(config.get("split", "official-deepdta")))


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    if args.limit_batches is not None:
        config["limit_batches"] = args.limit_batches
    if args.device is not None:
        config["device"] = args.device
    if args.run_id is not None:
        config["run_id"] = args.run_id

    seed_everything(int(config.get("seed", 42)))
    split_path = resolve_split_path(config)
    train_dataset = DTADataset(split_path, split="train", limit_rows=config.get("limit_train_rows"))
    valid_dataset = DTADataset(split_path, split="valid", limit_rows=config.get("limit_valid_rows"))
    collator = DTACollator(
        max_smiles_len=int(config.get("max_smiles_len", 100)),
        max_fasta_len=int(config.get("max_fasta_len", 1000)),
    )
    model = build_model(config)
    run_dir = prepare_run_dir(config, args.config)
    metrics = run_training(config, model, train_dataset, valid_dataset, collator, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
