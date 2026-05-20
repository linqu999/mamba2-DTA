"""Drug-target affinity model wrapper."""

from __future__ import annotations

from bimamba2_proteindta.models.fusion import AffinityRegressor
from bimamba2_proteindta.models.torch_utils import NN


class DrugTargetAffinityModel(NN.Module):
    """Combine a drug encoder, protein encoder, and regression head."""

    def __init__(
        self,
        drug_encoder,
        protein_encoder,
        drug_dim: int,
        protein_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.regressor = AffinityRegressor(drug_dim, protein_dim, hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, batch):
        drug_repr = self.drug_encoder(batch["smiles_input_ids"], batch["smiles_attention_mask"])
        protein_repr = self.protein_encoder(batch["protein_input_ids"], batch["protein_attention_mask"])
        return self.regressor(drug_repr, protein_repr)
