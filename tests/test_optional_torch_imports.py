def test_model_modules_import_without_eager_cuda_dependencies() -> None:
    import bimamba2_proteindta.models.dta_model
    import bimamba2_proteindta.models.fusion
    import bimamba2_proteindta.models.protein_bimamba2
    import bimamba2_proteindta.models.protein_cnn
    import bimamba2_proteindta.models.protein_mamba2
    import bimamba2_proteindta.models.protein_transformer
    import bimamba2_proteindta.models.smiles_mambatrans
