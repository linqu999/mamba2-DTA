# Data Notes

## Source

The Davis and KIBA benchmark files were obtained from the DeepDTA repository:

https://github.com/hkmztrk/DeepDTA

Network notes:

- Direct `raw.githubusercontent.com` access failed in this environment.
- GitHub blob API worked for small files but was unstable for large KIBA files.
- The full repository ZIP was downloaded from GitHub codeload, then only the
  required Davis/KIBA files were extracted.

The extracted files are recorded in `data/raw/deepdta/manifest.json` with source
Git blob SHA, expected size, actual size, and SHA256 checksum.

## Raw Files Used

For each dataset:

- `Y`
- `ligands_can.txt`
- `proteins.txt`
- `folds/train_fold_setting1.txt`
- `folds/test_fold_setting1.txt`

Similarity matrices are not required for the current sequence-based DTA
pipeline and were not extracted into `data/raw/deepdta/`.

## Processed Dataset Validation

`data/processed/davis.csv`

- rows: 30,056
- drugs: 68
- targets: 442
- affinity: pKd converted from Davis Kd nM values
- affinity range: 5.0 to 10.795880017344077
- max SMILES length: 92
- max protein length: 2,549

`data/processed/kiba.csv`

- rows: 118,254
- drugs: 2,111
- targets: 229
- affinity: KIBA score, NaN interactions removed
- affinity range: 0.0 to 17.200179498
- max SMILES length: 590
- max protein length: 4,128

These counts match the expected DeepDTA benchmark scale: Davis 68 drugs, 442
targets, 30,056 interactions; KIBA 2,111 drugs, 229 targets, 118,254 observed
interactions.

## Splits Generated

Official DeepDTA setting1 split with fold 0 used as validation:

- Davis: train 20,036 / valid 5,010 / test 5,010
- KIBA: train 78,836 / valid 19,709 / test 19,709

Cold split files with seed 42:

- `data/splits/davis/cold_drug_seed42.csv`
- `data/splits/davis/cold_target_seed42.csv`
- `data/splits/kiba/cold_drug_seed42.csv`
- `data/splits/kiba/cold_target_seed42.csv`

Leakage checks:

- Davis cold-drug: train/valid/test drug overlaps are all 0.
- Davis cold-target: train/valid/test target overlaps are all 0.
- KIBA cold-drug: train/valid/test drug overlaps are all 0.
- KIBA cold-target: train/valid/test target overlaps are all 0.
