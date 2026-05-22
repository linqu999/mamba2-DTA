"""Model factory for configured DTA experiments."""

from __future__ import annotations

from bimamba2_proteindta.data.tokenizers import build_amino_acid_vocab, build_smiles_vocab
from bimamba2_proteindta.models.dta_model import DrugTargetAffinityModel
from bimamba2_proteindta.models.protein_bimamba2 import ProteinBiMamba2Encoder
from bimamba2_proteindta.models.protein_cnn import ProteinCNNEncoder
from bimamba2_proteindta.models.protein_mamba2 import ProteinMamba2Encoder
from bimamba2_proteindta.models.protein_transformer import ProteinTransformerEncoder
from bimamba2_proteindta.models.smiles_mambatrans import SMILESEncoder


MODEL_TO_PROTEIN_ENCODER = {
    "mtdta_cnn": "cnn",
    "mtdta_transformer_protein": "transformer",
    "mtdta_unimamba2_protein": "unimamba2",
    "mtdta_bimamba2_protein": "bimamba2",
}


def _resolve_protein_encoder_name(config: dict) -> str:
    explicit = config.get("protein_encoder")
    if explicit is not None:
        return str(explicit)
    model_name = str(config.get("model", config.get("model_name", "mtdta_cnn")))
    if model_name not in MODEL_TO_PROTEIN_ENCODER:
        supported = ", ".join(sorted(MODEL_TO_PROTEIN_ENCODER))
        raise ValueError(f"Unsupported model {model_name!r}; supported models: {supported}")
    return MODEL_TO_PROTEIN_ENCODER[model_name]


def _build_protein_encoder(config: dict, vocab_size: int, output_dim: int, dropout: float):
    encoder_name = _resolve_protein_encoder_name(config)
    if encoder_name == "cnn":
        return ProteinCNNEncoder(
            vocab_size=vocab_size,
            embed_dim=int(config.get("protein_embed_dim", 32)),
            num_filters=int(config.get("protein_num_filters", 32)),
            output_dim=output_dim,
            dropout=dropout,
        )
    if encoder_name == "transformer":
        return ProteinTransformerEncoder(
            vocab_size=vocab_size,
            d_model=int(config.get("protein_d_model", config.get("protein_embed_dim", 128))),
            nhead=int(config.get("protein_nhead", 4)),
            num_layers=int(config.get("protein_layers", 2)),
            dim_feedforward=int(config.get("protein_dim_feedforward", 256)),
            output_dim=output_dim,
            dropout=dropout,
        )
    if encoder_name == "unimamba2":
        return ProteinMamba2Encoder(
            vocab_size=vocab_size,
            d_model=int(config.get("protein_d_model", 128)),
            num_layers=int(config.get("protein_layers", 2)),
            d_state=int(config.get("protein_d_state", 64)),
            d_conv=int(config.get("protein_d_conv", 4)),
            expand=int(config.get("protein_expand", 2)),
            headdim=int(config.get("protein_headdim", 64)),
            output_dim=output_dim,
            pooling=str(config.get("protein_pooling", config.get("pooling", "mean_max"))),
            dropout=dropout,
        )
    if encoder_name == "bimamba2":
        return ProteinBiMamba2Encoder(
            vocab_size=vocab_size,
            d_model=int(config.get("protein_d_model", 128)),
            num_layers=int(config.get("protein_layers", 2)),
            d_state=int(config.get("protein_d_state", 64)),
            d_conv=int(config.get("protein_d_conv", 4)),
            expand=int(config.get("protein_expand", 2)),
            headdim=int(config.get("protein_headdim", 64)),
            output_dim=output_dim,
            pooling=str(config.get("protein_pooling", config.get("pooling", "mean_max"))),
            direction_fusion=str(config.get("direction_fusion", "gated")),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported protein_encoder {encoder_name!r}")


def build_model(config: dict):
    """Build a DTA model while keeping the SMILES branch stable."""
    drug_dim = int(config.get("drug_dim", 100))
    protein_dim = int(config.get("protein_dim", 96))
    dropout = float(config.get("dropout", 0.1))
    smiles_vocab_size = len(build_smiles_vocab())
    protein_vocab_size = len(build_amino_acid_vocab())

    drug_encoder = SMILESEncoder(
        vocab_size=smiles_vocab_size,
        d_model=int(config.get("smiles_d_model", 64)),
        nhead=int(config.get("smiles_nhead", 4)),
        num_layers=int(config.get("smiles_layers", 1)),
        output_dim=drug_dim,
        dropout=dropout,
    )
    protein_encoder = _build_protein_encoder(config, protein_vocab_size, protein_dim, dropout)
    return DrugTargetAffinityModel(
        drug_encoder=drug_encoder,
        protein_encoder=protein_encoder,
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dims=tuple(config.get("regressor_hidden_dims", [128, 64])),
        dropout=dropout,
    )
