"""Masked pooling utilities for sequence encoders."""

from __future__ import annotations

from bimamba2_proteindta.models.torch_utils import require_torch


def masked_mean_pool(sequence, attention_mask, eps: float = 1e-8):
    mask = attention_mask.to(sequence.dtype).unsqueeze(-1)
    summed = (sequence * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(eps)
    return summed / denom


def masked_max_pool(sequence, attention_mask):
    torch, _ = require_torch()
    mask = attention_mask.bool().unsqueeze(-1)
    fill_value = torch.finfo(sequence.dtype).min
    masked = sequence.masked_fill(~mask, fill_value)
    pooled = masked.max(dim=1).values
    all_padding = attention_mask.sum(dim=1).eq(0).unsqueeze(-1)
    return torch.where(all_padding, torch.zeros_like(pooled), pooled)


def masked_pool(sequence, attention_mask, mode: str = "mean"):
    torch, _ = require_torch()
    if mode == "mean":
        return masked_mean_pool(sequence, attention_mask)
    if mode == "max":
        return masked_max_pool(sequence, attention_mask)
    if mode == "mean_max":
        return torch.cat(
            [masked_mean_pool(sequence, attention_mask), masked_max_pool(sequence, attention_mask)],
            dim=-1,
        )
    raise ValueError(f"Unsupported pooling mode: {mode}")
