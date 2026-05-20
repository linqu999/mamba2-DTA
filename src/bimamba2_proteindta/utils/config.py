"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file using PyYAML when available."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config files") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data
