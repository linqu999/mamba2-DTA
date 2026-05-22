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

## Primary Paper-Aligned Splits

For formal training and paper-level comparison, use the MambaTransDTA Table 1
split files:

- `data/splits/davis/mambatransdta_table1.csv`
- `data/splits/kiba/mambatransdta_table1.csv`

These splits are generated from the DeepDTA setting1 fold files with validation
fold choices selected to match the counts reported in MambaTransDTA Table 1:

- Davis: validation fold 1 -> train 20,037 / valid 5,009 / test 5,010.
- KIBA: validation fold 0 -> train 78,836 / valid 19,709 / test 19,709.

Generate them with:

```bash
python scripts/02_make_splits.py --input data/processed/davis.csv --output data/splits/davis/mambatransdta_table1.csv --metadata data/splits/davis/mambatransdta_table1_metadata.json --split mambatransdta-table1 --folds-dir data/raw/deepdta/davis/folds

python scripts/02_make_splits.py --input data/processed/kiba.csv --output data/splits/kiba/mambatransdta_table1.csv --metadata data/splits/kiba/mambatransdta_table1_metadata.json --split mambatransdta-table1 --folds-dir data/raw/deepdta/kiba/folds
```

Acceptance counts:

| Dataset | Train | Valid | Test |
| --- | ---: | ---: | ---: |
| Davis | 20,037 | 5,009 | 5,010 |
| KIBA | 78,836 | 19,709 | 19,709 |

## Removed Development Splits

Early smoke testing used a fold0 derivative named `official_deepdta.csv`.
For Davis, that produced train 20,036 / valid 5,010 / test 5,010, which differs
from MambaTransDTA Table 1 by one train/validation row. The file and its config
references have been removed to avoid accidental use in formal runs.

Cold-start split helpers remain in code, but generated cold split CSVs are not
part of the current training workflow. They should be regenerated later only
when the project reaches the cold-start evaluation stage.
