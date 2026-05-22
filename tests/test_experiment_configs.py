from pathlib import Path

import yaml


PROTEIN_CONFIGS = [
    "configs/experiment/davis_mambatransdta_table1_transformer_protein.yaml",
    "configs/experiment/davis_mambatransdta_table1_unimamba2_protein.yaml",
    "configs/experiment/davis_mambatransdta_table1_bimamba2_protein.yaml",
    "configs/experiment/kiba_mambatransdta_table1_transformer_protein.yaml",
    "configs/experiment/kiba_mambatransdta_table1_unimamba2_protein.yaml",
    "configs/experiment/kiba_mambatransdta_table1_bimamba2_protein.yaml",
]


def test_protein_encoder_configs_keep_controlled_training_protocol() -> None:
    for config_path in PROTEIN_CONFIGS:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        assert config["split"] == "mambatransdta-table1"
        assert config["epochs"] == 500
        assert config["batch_size"] == 512
        assert config["learning_rate"] == 0.0005
        assert config["drug_dim"] == 128
        assert config["protein_dim"] == 128
        assert config["smiles_d_model"] == 128
        assert config["smiles_layers"] == 2
        assert config["smiles_nhead"] == 4
        assert config["regressor_hidden_dims"] == [1024, 512]
        if "mamba2" in config["protein_encoder"]:
            assert config["protein_d_model"] == 128
            assert config["protein_d_state"] == 64
            assert config["protein_headdim"] == 64


def test_experiment_matrices_reference_existing_configs() -> None:
    for matrix_path in [
        "configs/experiment/matrix_davis_protein_encoders.txt",
        "configs/experiment/matrix_kiba_protein_encoders.txt",
    ]:
        configs = [
            line.strip()
            for line in Path(matrix_path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(configs) == 4
        assert all(Path(config).exists() for config in configs)
