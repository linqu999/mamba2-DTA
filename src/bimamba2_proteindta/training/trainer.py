"""Minimal training loop for DTA smoke and baseline runs."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from bimamba2_proteindta.training.metrics import compute_regression_metrics


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required for training") from exc
    return torch, nn, DataLoader


def make_run_id(config: dict) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset = config.get("dataset", "dataset")
    split = str(config.get("split", "split")).replace("-", "_")
    model = config.get("model", "model")
    seed = config.get("seed", 0)
    return f"{timestamp}_{dataset}_{split}_{model}_seed{seed}"


def prepare_run_dir(config: dict, config_path: str | Path | None = None) -> Path:
    run_root = Path(config.get("run_root", "runs"))
    run_id = config.get("run_id") or make_run_id(config)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if config_path is not None:
        (run_dir / "config_source.txt").write_text(str(config_path) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text(environment_summary(), encoding="utf-8")
    return run_dir


def environment_summary() -> str:
    lines = [f"python={sys.version.split()[0]}"]
    try:
        import torch

        lines.append(f"torch={torch.__version__}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        lines.append(f"cuda_version={torch.version.cuda}")
        if torch.cuda.is_available():
            lines.append(f"cuda_device_count={torch.cuda.device_count()}")
            lines.append(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    except ImportError:
        lines.append("torch=not-installed")
    return "\n".join(lines) + "\n"


def move_batch_to_device(batch: dict, device):
    torch, _, _ = _require_torch()
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def iter_limited(loader: Iterable, limit_batches: int | None):
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        yield batch


def train_one_epoch(model, loader, optimizer, loss_fn, device, limit_batches: int | None = None) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in iter_limited(loader, limit_batches):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = loss_fn(pred, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    if steps == 0:
        raise ValueError("No training batches were processed")
    return total_loss / steps


def evaluate(model, loader, loss_fn, device, limit_batches: int | None = None) -> tuple[float, list[dict]]:
    torch, _, _ = _require_torch()
    model.eval()
    total_loss = 0.0
    steps = 0
    predictions: list[dict] = []
    with torch.no_grad():
        for batch in iter_limited(loader, limit_batches):
            batch = move_batch_to_device(batch, device)
            pred = model(batch)
            loss = loss_fn(pred, batch["y"])
            total_loss += float(loss.detach().cpu())
            steps += 1
            pred_values = pred.detach().cpu().tolist()
            y_values = batch["y"].detach().cpu().tolist()
            for idx, (truth, pred_value) in enumerate(zip(y_values, pred_values)):
                predictions.append(
                    {
                        "y_true": float(truth),
                        "y_pred": float(pred_value),
                        "drug_id": batch["drug_id"][idx],
                        "target_id": batch["target_id"][idx],
                        "sequence_length": int(batch["sequence_length"][idx]),
                    }
                )
    if steps == 0:
        raise ValueError("No evaluation batches were processed")
    return total_loss / steps, predictions


def write_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["y_true", "y_pred", "drug_id", "target_id", "sequence_length"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_train_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def run_training(
    config: dict,
    model,
    train_dataset,
    valid_dataset,
    collator,
    run_dir: Path,
) -> dict:
    torch, nn, DataLoader = _require_torch()
    requested_device = config.get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(requested_device)
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(config.get("num_workers", 0)),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(config.get("num_workers", 0)),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    loss_fn = nn.MSELoss()
    epochs = int(config.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    limit_batches = config.get("limit_batches")
    limit_batches = int(limit_batches) if limit_batches is not None else None

    history = []
    best_valid_loss = float("inf")
    best_valid_predictions: list[dict] = []
    best_epoch = 0
    log_path = run_dir / "train.log"
    append_train_log(log_path, f"device={device}")
    append_train_log(log_path, f"train_rows={len(train_dataset)} valid_rows={len(valid_dataset)}")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, limit_batches)
        valid_loss, valid_predictions = evaluate(model, valid_loader, loss_fn, device, limit_batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        append_train_log(log_path, f"epoch={epoch} train_loss={train_loss:.8f} valid_loss={valid_loss:.8f}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_valid_predictions = valid_predictions
            torch.save(model.state_dict(), run_dir / "best.pt")
            write_predictions(run_dir / "predictions_valid.csv", best_valid_predictions)

    metrics_obj = compute_regression_metrics(
        [row["y_true"] for row in best_valid_predictions],
        [row["y_pred"] for row in best_valid_predictions],
    )
    metrics = {
        "train_loss": history[-1]["train_loss"],
        "valid_loss": history[-1]["valid_loss"],
        "best_valid_loss": best_valid_loss,
        "best_epoch": best_epoch,
        "valid": metrics_obj.as_dict(),
        "history": history,
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "limit_batches": limit_batches,
    }
    write_metrics(run_dir / "metrics.json", metrics)
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        writer.writerows(history)
    return metrics
