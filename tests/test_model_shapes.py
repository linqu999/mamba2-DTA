import importlib.util

import pytest

from bimamba2_proteindta.data.tokenizers import build_amino_acid_vocab, build_smiles_vocab
from bimamba2_proteindta.models.protein_bimamba2 import reverse_valid_tokens


torch = pytest.importorskip("torch")


def test_cnn_protein_encoder_shape() -> None:
    from bimamba2_proteindta.models.protein_cnn import ProteinCNNEncoder

    vocab_size = len(build_amino_acid_vocab())
    encoder = ProteinCNNEncoder(vocab_size=vocab_size, embed_dim=16, num_filters=8, output_dim=96)
    input_ids = torch.randint(0, vocab_size, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.bool)

    output = encoder(input_ids, attention_mask)

    assert output.shape == (2, 96)


def test_factory_builds_cnn_and_transformer_models() -> None:
    from bimamba2_proteindta.models.factory import build_model

    base_config = {
        "drug_dim": 32,
        "protein_dim": 32,
        "smiles_d_model": 16,
        "smiles_layers": 1,
        "smiles_nhead": 4,
        "regressor_hidden_dims": [16],
        "dropout": 0.1,
    }

    cnn = build_model(
        {
            **base_config,
            "model": "mtdta_cnn",
            "protein_embed_dim": 8,
            "protein_num_filters": 4,
        }
    )
    transformer = build_model(
        {
            **base_config,
            "model": "mtdta_transformer_protein",
            "protein_d_model": 16,
            "protein_layers": 1,
            "protein_nhead": 4,
            "protein_dim_feedforward": 32,
        }
    )

    assert cnn.protein_encoder.output_dim == 32
    assert transformer.protein_encoder.output_dim == 32


def test_smiles_encoder_shape() -> None:
    from bimamba2_proteindta.models.smiles_mambatrans import SMILESEncoder

    vocab_size = len(build_smiles_vocab())
    encoder = SMILESEncoder(vocab_size=vocab_size, d_model=16, nhead=4, num_layers=1, output_dim=100)
    input_ids = torch.randint(0, vocab_size, (2, 24))
    attention_mask = torch.ones(2, 24, dtype=torch.bool)

    output = encoder(input_ids, attention_mask)

    assert output.shape == (2, 100)


def test_transformer_protein_encoder_shape() -> None:
    from bimamba2_proteindta.models.protein_transformer import ProteinTransformerEncoder

    vocab_size = len(build_amino_acid_vocab())
    encoder = ProteinTransformerEncoder(vocab_size=vocab_size, d_model=16, nhead=4, num_layers=1, output_dim=96)
    input_ids = torch.randint(0, vocab_size, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.bool)

    output = encoder(input_ids, attention_mask)

    assert output.shape == (2, 96)


def test_reverse_valid_tokens_keeps_padding_at_end() -> None:
    sequence = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [0.0], [0.0]],
            [[4.0], [5.0], [0.0], [0.0], [0.0]],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    reversed_sequence = reverse_valid_tokens(sequence, attention_mask)

    assert reversed_sequence.squeeze(-1).tolist() == [
        [3.0, 2.0, 1.0, 0.0, 0.0],
        [5.0, 4.0, 0.0, 0.0, 0.0],
    ]


@pytest.mark.skipif(importlib.util.find_spec("mamba_ssm") is None, reason="mamba_ssm is not installed")
def test_factory_builds_mamba2_models() -> None:
    from bimamba2_proteindta.models.factory import build_model

    base_config = {
        "drug_dim": 32,
        "protein_dim": 32,
        "smiles_d_model": 16,
        "smiles_layers": 1,
        "smiles_nhead": 4,
        "protein_d_model": 16,
        "protein_layers": 1,
        "protein_d_state": 16,
        "protein_d_conv": 4,
        "protein_expand": 2,
        "protein_headdim": 16,
        "regressor_hidden_dims": [16],
    }

    uni = build_model({**base_config, "model": "mtdta_unimamba2_protein"})
    bi = build_model({**base_config, "model": "mtdta_bimamba2_protein"})

    assert uni.protein_encoder.output_dim == 32
    assert bi.protein_encoder.output_dim == 32


@pytest.mark.skipif(importlib.util.find_spec("mamba_ssm") is None, reason="mamba_ssm is not installed")
def test_mamba2_protein_encoders_shape() -> None:
    from bimamba2_proteindta.models.protein_bimamba2 import ProteinBiMamba2Encoder
    from bimamba2_proteindta.models.protein_mamba2 import ProteinMamba2Encoder

    vocab_size = len(build_amino_acid_vocab())
    input_ids = torch.randint(0, vocab_size, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.bool)

    uni = ProteinMamba2Encoder(vocab_size=vocab_size, d_model=16, num_layers=1, headdim=16, output_dim=96)
    bi = ProteinBiMamba2Encoder(vocab_size=vocab_size, d_model=16, num_layers=1, headdim=16, output_dim=96)

    assert uni(input_ids, attention_mask).shape == (2, 96)
    assert bi(input_ids, attention_mask).shape == (2, 96)
