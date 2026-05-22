"""UniMamba2 protein encoder."""

from __future__ import annotations

from bimamba2_proteindta.models.pooling import masked_pool
from bimamba2_proteindta.models.torch_utils import NN, require_torch


def _require_mamba2():
    try:
        from mamba_ssm import Mamba2
    except ImportError as exc:
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
        except ImportError:
            raise RuntimeError(
                "mamba_ssm is required for ProteinMamba2Encoder. Install mamba-ssm "
                "in a CUDA-enabled PyTorch environment."
            ) from exc
    return Mamba2


def make_mamba2_layer(Mamba2, d_model: int, d_state: int, d_conv: int, expand: int, headdim: int):
    """Create a Mamba2 layer with explicit head dimension when supported."""
    try:
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
    except TypeError:
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class ProteinMamba2Encoder(NN.Module):
    """Single-direction Mamba-2 protein encoder with masked pooling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        output_dim: int = 96,
        pooling: str = "mean_max",
        padding_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        _, nn = require_torch()
        Mamba2 = _require_mamba2()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.layers = nn.ModuleList(
            make_mamba2_layer(Mamba2, d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        pooled_dim = d_model * 2 if pooling == "mean_max" else d_model
        self.projection = nn.Linear(pooled_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        hidden = hidden * mask
        for layer, norm in zip(self.layers, self.norms):
            residual = hidden
            hidden = layer(hidden)
            hidden = norm(hidden + residual)
            hidden = hidden * mask
        pooled = masked_pool(hidden, attention_mask, self.pooling)
        return self.projection(self.dropout(pooled))
