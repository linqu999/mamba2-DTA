# MambaTransDTA Reproduction Notes

Official reproduction has not been run yet. This file records verified access
attempts and read-only architecture notes; it must not be treated as completed
reproduction evidence.

## Access Status

Attempted commands on 2026-05-20:

```bash
git clone https://github.com/pdssunny/MambaTransDTA.git external/MambaTransDTA
curl.exe -L https://api.github.com/repos/pdssunny/MambaTransDTA
```

Observed result:

- `git clone` connected but failed with `Recv failure: Connection was reset`.
- GitHub API returned `404 Not Found` for the repository endpoint.
- `external/MambaTransDTA/` is therefore not populated yet.

Next action: retry the clone from a stable network or from the GPU server. If
the repository remains inaccessible, use the official Figshare/code archive
linked by the paper page and record the exact source URL here.

## Preliminary Read-Only Notes

These notes are based on public indexed documentation available during local
intake and must be confirmed against the actual official source files once the
repository is accessible.

Expected official files:

- `main.py`: experiment entrypoint.
- `model.py`: model definitions for `SMILESModel`, `FASTAModel`, and the final
  predictor/classifier.
- `process_data.py`: dataset loading, sequence encoding, and tensor/CSV
  preparation.
- `train_and_test.py`: training and evaluation loop.
- `metrics.py`: DTA regression metrics such as MSE and CI.
- `data/`: bundled or expected dataset inputs.

Expected architecture:

```text
SMILES [B, 100]
  -> SMILESModel
  -> drug_repr [B, 100]

FASTA [B, 1000]
  -> FASTAModel CNN
  -> protein_repr [B, 96]

concat([drug_repr, protein_repr]) [B, 196]
  -> MLP regressor
  -> affinity prediction [B, 1]
```

The critical modification point for this project is the protein branch:
`FASTAModel` should be reproduced first, then replaced by UniMamba2 and
BiMamba2 encoders in the local `src/` implementation.

## Defaults To Confirm

Confirm these from source before running official reproduction:

| Item | Expected value | Status |
| --- | --- | --- |
| Entrypoint | `python main.py` | unconfirmed |
| Python | 3.9.19 | from project plan |
| PyTorch | 2.0.0 + CUDA 11.1 | from project plan |
| SMILES max length | 100 | unconfirmed |
| FASTA max length | 1000 | unconfirmed |
| FASTAModel output dim | 96 | unconfirmed |
| Fusion input dim | 196 | unconfirmed |
| Datasets | Davis, KIBA, Metz, BindingDB | unconfirmed |
| Default dataset | unknown | must inspect `main.py` |
| Batch size | unknown | must inspect `main.py` |
| Epochs | unknown | must inspect `main.py` |
| Learning rate | unknown | must inspect `main.py` |

## Reproduction Checklist

1. Clone or unpack official code into `external/MambaTransDTA/`.
2. Read `main.py`, `model.py`, `process_data.py`, `metrics.py`, and
   `train_and_test.py`.
3. Record default dataset, sequence lengths, batch size, epochs, optimizer, and
   metrics.
4. Run the official entrypoint in a matching GPU environment.
5. Save logs to `results/logs/official_mambatransdta.log`.
6. Save parsed metrics to `results/tables/official_metrics.json` or a clearly
   labeled manual table.
7. Document any required compatibility patch without silently changing the
   official method.
