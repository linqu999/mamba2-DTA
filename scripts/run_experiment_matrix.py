"""Run a list of experiment configs sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="Text file with one config path per line.")
    parser.add_argument("--epochs", type=int, help="Override epochs for every run.")
    parser.add_argument("--limit-batches", type=int, help="Override limit_batches for every run.")
    parser.add_argument("--device", help="Override device for every run.")
    parser.add_argument("--start-at", help="Skip configs until this exact config path is reached.")
    return parser.parse_args()


def load_configs(path: str | Path) -> list[str]:
    configs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            configs.append(stripped)
    if not configs:
        raise ValueError(f"No configs found in {path}")
    return configs


def build_command(config: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "scripts/03_train.py", "--config", config]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.limit_batches is not None:
        command.extend(["--limit-batches", str(args.limit_batches)])
    if args.device is not None:
        command.extend(["--device", args.device])
    return command


def main() -> int:
    args = parse_args()
    configs = load_configs(args.matrix)
    if args.start_at is not None:
        if args.start_at not in configs:
            raise SystemExit(f"--start-at config not found in matrix: {args.start_at}")
        configs = configs[configs.index(args.start_at) :]

    for index, config in enumerate(configs, start=1):
        command = build_command(config, args)
        print(f"[{index}/{len(configs)}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
