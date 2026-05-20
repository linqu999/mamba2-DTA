"""Batch collation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from bimamba2_proteindta.data.tokenizers import encode_protein, encode_smiles


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for tensor collation") from exc
    return torch


@dataclass(frozen=True)
class DTACollator:
    max_smiles_len: int = 100
    max_fasta_len: int = 1000

    def __call__(self, samples: Sequence[dict]):
        torch = _require_torch()
        smiles = [encode_smiles(sample["smiles"], self.max_smiles_len) for sample in samples]
        proteins = [encode_protein(sample["protein_sequence"], self.max_fasta_len) for sample in samples]

        return {
            "smiles_input_ids": torch.tensor([item.input_ids for item in smiles], dtype=torch.long),
            "smiles_attention_mask": torch.tensor([item.attention_mask for item in smiles], dtype=torch.bool),
            "protein_input_ids": torch.tensor([item.input_ids for item in proteins], dtype=torch.long),
            "protein_attention_mask": torch.tensor([item.attention_mask for item in proteins], dtype=torch.bool),
            "y": torch.tensor([float(sample["affinity"]) for sample in samples], dtype=torch.float32),
            "drug_id": [sample["drug_id"] for sample in samples],
            "target_id": [sample["target_id"] for sample in samples],
            "sequence_length": [len(sample["protein_sequence"]) for sample in samples],
            "smiles_length": [len(sample["smiles"]) for sample in samples],
        }
