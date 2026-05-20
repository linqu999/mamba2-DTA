# Project Instructions

This repository implements BiMamba2-ProteinDTA, a drug-target affinity
prediction project based on MambaTransDTA.

## Research Direction

- Treat MambaTransDTA as the main baseline.
- Use Mamba-DTA only as related work or an optional auxiliary baseline.
- Keep the MambaTransDTA-style SMILES drug encoder stable in the first phase.
- Replace only the FASTA/protein encoder when comparing CNN, Transformer,
  UniMamba2, and BiMamba2 variants.

## Engineering Rules

- Do not fabricate experimental results, metrics, logs, or figures.
- Do not edit files under `data/raw/`; raw datasets are immutable inputs.
- Do not delete `runs/`, checkpoints, logs, or result files unless explicitly
  asked.
- Store experiment parameters in `configs/`; avoid hard-coded training values.
- Keep official reproduction code under `external/MambaTransDTA/` read-only
  except for clearly documented minimal compatibility patches.
- Every training run should save its config, command, metrics, predictions,
  environment summary, and git commit when available.
- Prefer small, testable changes. Run tests before reporting a stage complete.

## Expected Interfaces

- Model forward interface: `forward(batch) -> y_pred`.
- Protein encoder interface:
  `forward(input_ids, attention_mask) -> protein_repr`.
- Data table schema:
  `drug_id,target_id,smiles,protein_sequence,affinity,dataset`.

## Local Workflow

- Use `make test` or `python -m pytest -q` for tests.
- Use `python scripts/00_check_env.py` to inspect local CUDA, PyTorch, and
  `mamba_ssm` availability.
- Keep generated data in `data/processed/`, `data/splits/`, and `data/cache/`.
- Keep figures, tables, and reports under `results/`.
