"""Simple deterministic tokenizers for SMILES and protein sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SMILES_CHARS = "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]\\abcdefghijklmnopqrstuvwxyz"
PAD_TOKEN = "pad"
UNK_TOKEN = "unk"
SPECIAL_TOKENS = {PAD_TOKEN: 0, UNK_TOKEN: 1}


def _build_vocab(alphabet: str) -> dict[str, int]:
    vocab = dict(SPECIAL_TOKENS)
    for char in alphabet:
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def build_amino_acid_vocab() -> dict[str, int]:
    """Return a stable amino-acid vocabulary with padding at index 0."""
    return _build_vocab(AMINO_ACIDS)


def build_smiles_vocab() -> dict[str, int]:
    """Return a character-level SMILES vocabulary for baseline experiments."""
    return _build_vocab(SMILES_CHARS)


@dataclass(frozen=True)
class EncodedSequence:
    input_ids: list[int]
    attention_mask: list[int]
    original_length: int
    truncated_length: int


def encode_sequence(
    sequence: str,
    vocab: Mapping[str, int],
    max_length: int,
    *,
    uppercase: bool = False,
) -> EncodedSequence:
    """Encode one sequence with truncation and right padding.

    Unknown characters map to `unk`; padding maps to `pad`.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    normalized = sequence.upper() if uppercase else sequence
    original_length = len(normalized)
    tokens = [vocab.get(char, vocab[UNK_TOKEN]) for char in normalized[:max_length]]
    truncated_length = len(tokens)
    pad_count = max_length - truncated_length

    return EncodedSequence(
        input_ids=tokens + [vocab[PAD_TOKEN]] * pad_count,
        attention_mask=[1] * truncated_length + [0] * pad_count,
        original_length=original_length,
        truncated_length=truncated_length,
    )


def encode_protein(sequence: str, max_length: int) -> EncodedSequence:
    return encode_sequence(sequence, build_amino_acid_vocab(), max_length, uppercase=True)


def encode_smiles(sequence: str, max_length: int) -> EncodedSequence:
    return encode_sequence(sequence, build_smiles_vocab(), max_length)
