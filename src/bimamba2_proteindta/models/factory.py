"""Model factory for configured DTA experiments."""

from __future__ import annotations

from bimamba2_proteindta.data.tokenizers import build_amino_acid_vocab, build_smiles_vocab
from bimamba2_proteindta.models.dta_model import DrugTargetAffinityModel
from bimamba2_proteindta.models.protein_cnn import ProteinCNNEncoder
from bimamba2_proteindta.models.smiles_mambatrans import SMILESEncoder


def build_model(config: dict):
    """Build the currently supported MambaTransDTA-like CNN baseline."""
    model_name = config.get("model", config.get("model_name", "mtdta_cnn"))
    if model_name != "mtdta_cnn":
        raise ValueError(f"Only mtdta_cnn is wired for training now, got {model_name!r}")

    drug_dim = int(config.get("drug_dim", 100))
    protein_dim = int(config.get("protein_dim", 96))
    smiles_vocab_size = len(build_smiles_vocab())
    protein_vocab_size = len(build_amino_acid_vocab())

    drug_encoder = SMILESEncoder(
        vocab_size=smiles_vocab_size,
        d_model=int(config.get("smiles_d_model", 64)),
        nhead=int(config.get("smiles_nhead", 4)),
        num_layers=int(config.get("smiles_layers", 1)),
        output_dim=drug_dim,
        dropout=float(config.get("dropout", 0.1)),
    )
    protein_encoder = ProteinCNNEncoder(
        vocab_size=protein_vocab_size,
        embed_dim=int(config.get("protein_embed_dim", 32)),
        num_filters=int(config.get("protein_num_filters", 32)),
        output_dim=protein_dim,
        dropout=float(config.get("dropout", 0.1)),
    )
    return DrugTargetAffinityModel(
        drug_encoder=drug_encoder,
        protein_encoder=protein_encoder,
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dims=tuple(config.get("regressor_hidden_dims", [128, 64])),
        dropout=float(config.get("dropout", 0.1)),
    )
