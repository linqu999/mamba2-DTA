"""Fusion heads for drug and protein representations."""

from __future__ import annotations

from bimamba2_proteindta.models.torch_utils import NN, require_torch


class AffinityRegressor(NN.Module):
    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ) -> None:
        _, nn = require_torch()
        super().__init__()
        layers = []
        input_dim = drug_dim + protein_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, drug_repr, protein_repr):
        torch, _ = require_torch()
        return self.network(torch.cat([drug_repr, protein_repr], dim=-1)).squeeze(-1)
