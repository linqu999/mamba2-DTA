# BiMamba2-ProteinDTA

Efficient full-length protein encoding for drug-target affinity prediction
based on the MambaTransDTA baseline.

## Goal

This project first reproduces MambaTransDTA, then builds a clean local
training framework where the MambaTransDTA-style protein CNN encoder can be
replaced with UniMamba2 and BiMamba2 protein encoders. The first research
question is whether a Mamba-2 protein encoder improves long-sequence
scalability while keeping competitive DTA accuracy.

## Current Phase

Completed locally:

- Project skeleton, rules, environment checks, and metric tests.
- Normalized CSV validation and dataset statistics.
- Random, cold-drug, and cold-target split generation with leakage checks.
- DeepDTA Davis/KIBA raw data download, checksum manifest, processed CSVs, and
  official/cold splits.
- Baseline model interfaces for SMILES, CNN protein, Transformer protein,
  UniMamba2 protein, BiMamba2 protein, and fusion regression.
- Dataset loading, tensor collation, model factory, and `scripts/03_train.py`
  are wired for a CNN baseline debug run.

Not completed yet:

- Official MambaTransDTA clone/reproduction.
- Local smoke training has not been executed yet because this machine does not
  currently have PyTorch installed; the CPU PyTorch install attempt stalled on
  the large wheel download and was stopped.
- GPU Mamba2 validation, benchmark figures, and real metrics.

No experimental results are included yet.

## Repository Layout

```text
external/                     official repositories, read-only by default
configs/                      data, model, and experiment configs
data/raw/                     immutable raw datasets
data/processed/               normalized dataset tables
data/splits/                  random and cold split files
data/cache/                   tokenized or derived caches
src/bimamba2_proteindta/      project package
scripts/                      command-line entrypoints
tests/                        unit and smoke tests
runs/                         training run outputs
results/                      tables, figures, logs, reports
docs/                         notes, runbooks, summaries
```

## Quick Start

Create an environment, then install the project in editable mode:

```bash
pip install -e ".[dev]"
python scripts/00_check_env.py
pytest -q
```

If CUDA/Mamba dependencies are not available locally, CPU-only tests should
still run. GPU validation is expected on the training server.

Debug training command once PyTorch is installed:

```bash
python scripts/03_train.py --config configs/experiment/debug_cpu.yaml --limit-batches 2
```

## Main Workflow

1. Clone the official MambaTransDTA repository into `external/MambaTransDTA/`.
2. Record the official reproduction in `docs/reproduction_notes.md`.
3. Prepare Davis/KIBA into the unified schema.
4. Create random, cold-drug, and cold-target splits.
5. Train the MambaTransDTA-like CNN baseline.
6. Replace the protein encoder with UniMamba2 and BiMamba2 variants.
7. Benchmark memory, latency, and throughput across protein lengths.
8. Export result tables, figures, and the teacher-facing package.

## Target Run Artifacts

Each training run should eventually save:

```text
runs/{run_id}/
  config.yaml
  metrics.json
  metrics.csv
  train.log
  best.pt
  predictions_valid.csv
  predictions_test.csv
  environment.txt
  git_commit.txt
  command.txt
```

## Status

The repository is being built from scratch. Any missing metric or figure means
the corresponding experiment has not been run yet.
