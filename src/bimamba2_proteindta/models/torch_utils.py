"""Torch import helpers for optional local GPU dependencies."""

from __future__ import annotations


class _MissingTorchModule:
    pass


class _MissingNN:
    Module = _MissingTorchModule


try:
    import torch as TORCH
    from torch import nn as NN
except ImportError:  # pragma: no cover - depends on local environment
    TORCH = None
    NN = _MissingNN()


def require_torch():
    if TORCH is None:
        raise RuntimeError("torch is required for model components")
    return TORCH, NN
