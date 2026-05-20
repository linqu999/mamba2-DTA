"""Transformer protein encoder baseline."""

from __future__ import annotations

from bimamba2_proteindta.models.pooling import masked_mean_pool
from bimamba2_proteindta.models.torch_utils import NN, require_torch


class ProteinTransformerEncoder(NN.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        output_dim: int = 96,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        _, nn = require_torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.projection = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        key_padding_mask = ~attention_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        pooled = masked_mean_pool(hidden, attention_mask)
        return self.projection(pooled)
