# Research Positioning

This project treats MambaTransDTA as the external target baseline and studies
whether replacing the protein encoder improves drug-target affinity prediction
under a controlled MambaTransDTA-style framework.

## Core Hypothesis

MambaTransDTA strengthens sequence modeling mainly through the drug/SMILES
branch while retaining an embedding plus 1D-CNN protein branch. Our hypothesis
is that target-side sequence modeling remains a bottleneck, especially for
long proteins and unseen target generalization.

The controlled question is:

> If the drug branch, splits, training code, and regressor are held stable, can
> Transformer, UniMamba2, or BiMamba2 protein encoders outperform the CNN
> protein baseline?

## Baseline Interpretation

External literature target:

| Dataset | Method | MSE | CI | rm2 |
| --- | --- | ---: | ---: | ---: |
| Davis | MambaTransDTA | 0.191 | 0.894 | 0.737 |
| KIBA | MambaTransDTA | 0.173 | 0.871 | 0.722 |

These numbers are treated as external reference targets from the paper, not as
results produced by this repository.

Internal controlled baseline:

- `mtdta_cnn`: stable MambaTransDTA-style SMILES branch plus CNN protein
  encoder.
- This is not an exact official MambaTransDTA reproduction unless the official
  implementation details are fully matched.
- Its main role is to anchor controlled protein-encoder comparisons inside this
  repository.

## Main Model Comparison

All main experiments should use the same Davis/KIBA `mambatransdta-table1`
splits and the same training protocol where possible.

| Model | Drug branch | Protein branch | Purpose |
| --- | --- | --- | --- |
| `mtdta_cnn` | fixed SMILES encoder | CNN | Internal controlled baseline |
| `mtdta_transformer_protein` | fixed SMILES encoder | Transformer | Attention-based protein baseline |
| `mtdta_unimamba2_protein` | fixed SMILES encoder | UniMamba2 | Single-direction state-space protein model |
| `mtdta_bimamba2_protein` | fixed SMILES encoder | BiMamba2 | Proposed bidirectional state-space protein model |

Primary metrics:

- MSE: lower is better.
- CI: higher is better.
- rm2: higher is better.

Publication-ready runs must have `artifact_validation_ok=True`,
`publication_ready=True`, and no `limit_batches`.

## Where The Proposed Method Should Win

The strongest claim is not only that BiMamba2 improves one random Davis/KIBA
split. The stronger claim is that protein-side state-space modeling improves:

1. Accuracy under the same fixed split.
2. Long-protein robustness when increasing `max_fasta_len`.
3. Cold-target generalization, where test proteins are unseen during training.
4. Efficiency relative to Transformer protein encoders on long sequences.

## Suggested Experiment Order

1. Davis fixed split: CNN, Transformer, UniMamba2, BiMamba2.
2. KIBA fixed split: run the same comparison after Davis behavior is stable.
3. Long-sequence ablation: compare `max_fasta_len` values such as 1000, 2000,
   and 4000.
4. Cold-target split: compare CNN and BiMamba2 first, then add Transformer and
   UniMamba2 if the signal is promising.
5. Multi-seed confirmation for the best comparison, at least seeds 42, 43, and
   44.

Smoke-test the Davis model matrix after installing `mamba_ssm`:

```bash
python scripts/run_experiment_matrix.py \
  --matrix configs/experiment/matrix_davis_protein_encoders.txt \
  --epochs 1 \
  --limit-batches 2
```

Run the full Davis comparison after every smoke run has
`artifact_validation.ok=true`:

```bash
python scripts/run_experiment_matrix.py \
  --matrix configs/experiment/matrix_davis_protein_encoders.txt
```

Use `--start-at <config-path>` to resume from the first unfinished model.

## Reporting Rule

Do not claim exact MambaTransDTA reproduction unless the official code and
supporting hyperparameters are matched. Until then, report MambaTransDTA paper
numbers as external reference values and this repository's CNN as an internal
controlled baseline.
