"""Bidirectional Mamba-2 protein encoder."""

from __future__ import annotations

from bimamba2_proteindta.models.pooling import masked_pool
from bimamba2_proteindta.models.protein_mamba2 import _require_mamba2, make_mamba2_layer
from bimamba2_proteindta.models.torch_utils import NN, require_torch


def reverse_valid_tokens(sequence, attention_mask):
    """Reverse only valid tokens in each padded sequence.

    Padding remains at the end of each row, which avoids feeding leading pad
    tokens to the backward branch.
    """
    torch, _ = require_torch()
    reversed_sequence = sequence.clone()
    lengths = attention_mask.long().sum(dim=1).tolist()
    for batch_idx, length in enumerate(lengths):
        if length > 0:
            reversed_sequence[batch_idx, :length] = torch.flip(sequence[batch_idx, :length], dims=[0])
    return reversed_sequence


class ProteinBiMamba2Encoder(NN.Module):
    """Forward/backward Mamba-2 encoder with concat or gated fusion."""

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
        direction_fusion: str = "gated",
        padding_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        _, nn = require_torch()
        Mamba2 = _require_mamba2()
        super().__init__()
        if direction_fusion not in {"concat", "gated"}:
            raise ValueError("direction_fusion must be 'concat' or 'gated'")

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.forward_layers = nn.ModuleList(
            make_mamba2_layer(Mamba2, d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
            for _ in range(num_layers)
        )
        self.backward_layers = nn.ModuleList(
            make_mamba2_layer(Mamba2, d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
            for _ in range(num_layers)
        )
        self.forward_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        self.backward_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        self.direction_fusion = direction_fusion
        if direction_fusion == "concat":
            self.direction_projection = nn.Linear(d_model * 2, d_model)
        else:
            self.gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        pooled_dim = d_model * 2 if pooling == "mean_max" else d_model
        self.projection = nn.Linear(pooled_dim, output_dim)
        self.output_dim = output_dim

    def _run_layers(self, hidden, attention_mask, layers, norms):
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        hidden = hidden * mask
        for layer, norm in zip(layers, norms):
            residual = hidden
            hidden = layer(hidden)
            hidden = norm(hidden + residual)
            hidden = hidden * mask
        return hidden

    def _fuse_directions(self, forward_hidden, backward_hidden):
        torch, _ = require_torch()
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        if self.direction_fusion == "concat":
            return self.direction_projection(combined)
        gate = torch.sigmoid(self.gate(combined))
        return gate * forward_hidden + (1.0 - gate) * backward_hidden

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        forward_hidden = self._run_layers(embedded, attention_mask, self.forward_layers, self.forward_norms)

        reversed_embedded = reverse_valid_tokens(embedded, attention_mask)
        backward_reversed = self._run_layers(
            reversed_embedded,
            attention_mask,
            self.backward_layers,
            self.backward_norms,
        )
        backward_hidden = reverse_valid_tokens(backward_reversed, attention_mask)

        fused = self._fuse_directions(forward_hidden, backward_hidden)
        fused = fused * attention_mask.to(fused.dtype).unsqueeze(-1)
        pooled = masked_pool(fused, attention_mask, self.pooling)
        return self.projection(self.dropout(pooled))
