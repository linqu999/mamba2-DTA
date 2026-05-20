from bimamba2_proteindta.data.tokenizers import (
    AMINO_ACIDS,
    build_amino_acid_vocab,
    encode_protein,
    encode_smiles,
)


def test_amino_acid_vocab_has_stable_special_tokens() -> None:
    vocab = build_amino_acid_vocab()

    assert vocab["pad"] == 0
    assert vocab["unk"] == 1
    assert len(vocab) == len(AMINO_ACIDS) + 2
    assert vocab["A"] == 2


def test_encode_protein_truncates_and_pads_with_mask() -> None:
    encoded = encode_protein("acdxx", max_length=4)

    assert encoded.original_length == 5
    assert encoded.truncated_length == 4
    assert encoded.attention_mask == [1, 1, 1, 1]
    assert encoded.input_ids[0] == build_amino_acid_vocab()["A"]
    assert encoded.input_ids[-1] == build_amino_acid_vocab()["unk"]


def test_encode_smiles_pads_short_sequence() -> None:
    encoded = encode_smiles("CCO", max_length=5)

    assert encoded.truncated_length == 3
    assert encoded.attention_mask == [1, 1, 1, 0, 0]
    assert encoded.input_ids[-2:] == [0, 0]
