# MambaTransDTA Paper Notes

Paper:

- Title: MambaTransDTA: A Hybrid Mamba-Transformer Architecture for Accurate Drug-Target Binding Affinity Prediction
- Authors: Xinpo Lou, Jianxiu Cai, Qidong Liu, Shirley W. I. Siu
- DOI shown in the PDF: `10.1021/acs.jcim.5c02361`
- Official code/data URL stated by the paper: `https://github.com/pdssunny/MambaTransDTA`
- Notes verified from the local PDF on 2026-05-22.

These notes are for planning and reproduction alignment. They are not
experimental evidence from this repository.

## Core Problem

The paper treats drug-target affinity prediction as a regression task. The
model receives:

- drug molecular sequence: SMILES
- target protein sequence: FASTA/amino acid sequence
- output: a numerical binding affinity prediction

For Davis, the paper states that Kd values are converted to pKd. KIBA labels
are treated as KIBA scores.

## Main Contributions

The paper proposes MambaTransDTA, a hybrid Mamba-Transformer architecture for
DTA prediction. The stated motivation is to combine:

- Mamba-style long-range sequence dependency modeling with linear complexity.
- Transformer attention for short-range/local interactions.
- Sequence-only drug/protein inputs, avoiding hand-engineered features.

For this repository, the most important design point is that the paper keeps
the protein branch as an embedding plus 1D-CNN branch, while the drug SMILES
branch receives the Mamba/Transformer hybrid treatment. Our project direction
therefore remains: reproduce or approximate the MambaTransDTA-style baseline,
then replace only the protein encoder with Transformer, UniMamba2, and
BiMamba2 variants for controlled comparison.

## Dataset Summary From Paper Table 1

The paper reports four benchmark datasets and fixed train/validation/test
partitions.

| Dataset | Compounds | Proteins | Binding entities | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Davis | 68 | 442 | 30,056 | 20,037 | 5,009 | 5,010 |
| KIBA | 2,111 | 229 | 118,254 | 78,836 | 19,709 | 19,709 |
| Metz | 1,421 | 156 | 35,259 | 28,207 | 3,526 | 3,526 |
| BindingDB | 803,234 | 5,560 | 1,048,564 | 966,844 | 81,720 | 20,001 |

The paper explicitly states that comparative experiments use the same
train/validation/test partitions from Table 1. Therefore, formal reproduction
and paper-level comparison should match these counts or document any exact
source-code split convention that explains a difference.

## Current Repository Split Status

Current generated Davis/KIBA files use the DeepDTA benchmark files from
`hkmztrk/DeepDTA`.

Validated processed dataset counts:

| Dataset | Rows | Drugs | Targets | Max protein length |
| --- | ---: | ---: | ---: | ---: |
| Davis | 30,056 | 68 | 442 | 2,549 |
| KIBA | 118,254 | 2,111 | 229 | 4,128 |

The repository now treats `mambatransdta-table1` as the primary formal split
protocol. It is generated from the DeepDTA setting1 fold files, with validation
fold choices selected to match MambaTransDTA Table 1 counts:

| Dataset | Valid fold | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: |
| Davis | 1 | 20,037 | 5,009 | 5,010 |
| KIBA | 0 | 78,836 | 19,709 | 19,709 |

The earlier Davis fold0 derivative produced 20,036 train / 5,010 validation /
5,010 test rows. That file has been removed from the project workflow to avoid
accidental use.

## MambaTransDTA Architecture

The paper describes the framework as follows:

1. Protein FASTA sequences are label-encoded.
2. Protein tokens pass through an embedding layer.
3. Protein features are extracted by a 1D-CNN module.
4. Drug SMILES strings are label-encoded and embedded.
5. A Mamba preprocessing block is used instead of conventional explicit
   positional encoding for SMILES.
6. The MambaTrans encoder combines Mamba and multi-head attention layers:
   attention captures shorter-range dependencies, while Mamba captures
   longer-range sequence dependencies.
7. Drug and protein features are pooled into one-dimensional vectors.
8. Drug and protein representations are concatenated.
9. A three-layer fully connected network predicts binding affinity.

Important project implication:

- In phase 1, do not change the SMILES branch when comparing protein encoders.
- The controlled variable should be the protein encoder only: CNN,
  Transformer, UniMamba2, BiMamba2.

## Drug Representation Details

The paper treats SMILES as a character sequence. Characters are mapped to
integer IDs, then embedded into a sequence representation. The drug
representation is processed by the Mamba preprocessing block and the
MambaTrans encoder.

The paper emphasizes that Mamba can implicitly encode sequence order through
state updates, while Transformer components usually require explicit positional
information. This motivates the Mamba preprocessing block.

## Protein Representation Details

The paper treats protein sequences as amino acid sequences in FASTA format.
Protein characters are label-encoded, embedded, and passed through a 1D-CNN
feature extractor. The extracted protein representation is then pooled and
concatenated with the drug representation.

For this repository, the protein CNN branch is the baseline to preserve before
introducing protein Mamba2 variants.

## Evaluation Metrics

The paper uses three main regression metrics:

- MSE: lower is better.
- CI: concordance index; higher is better.
- rm2: external predictive performance measure; higher is better, and the
  paper states that values at least 0.5 are acceptable.

Our repository currently computes MSE, RMSE, MAE, CI, and rm2. MSE, CI, and
rm2 are the paper-critical metrics.

Implementation note: CI is computed with an exact sorting/counting algorithm
rather than an explicit double loop over all sample pairs. The metric definition
is unchanged: pairs with equal ground-truth affinity are ignored; correctly
ordered prediction pairs receive full credit; tied predictions receive half
credit. The faster implementation is needed so per-epoch validation CI can be
recorded without making KIBA runs impractically slow.

## Experimental Setup From Paper Table 2

Default MambaTransDTA hyperparameters reported in the paper:

| Hyperparameter | Search space | Default |
| --- | --- | --- |
| Learning rate | 0.01, 0.005, 0.001, 0.0005 | 0.0005 |
| Batch size | 128, 256, 512, 1024 | 512 |
| Dropout | 0.1, 0.2, 0.5, 0.7 | 0.1 |
| Attention heads | 1, 2, 4, 8 | 4 |
| Mamba state dimension | 32, 64, 128, 256 | 128 |
| Drug representation dimension | 64, 128, 256, 512 | 128 |
| Protein representation dimension | 64, 128, 256, 512 | 128 |
| Epochs | 100, 300, 500, 1000 | 500 |

Items not confirmed from the main PDF text:

- Optimizer type.
- Weight decay or scheduler.
- Early stopping policy.
- Random seed policy and number of repeated runs.
- Exact official sequence truncation lengths.
- Exact train/validation split file format used by the official repository.
- Exact CNN filter count and kernel settings for the protein branch.

The paper says dataset-specific hyperparameters are in Supporting Table S1.
Those should be collected before formal reproduction.

Repository configs named `*_cnn_paper_hparams.yaml` align the settings that are
available from Table 2: learning rate 0.0005, batch size 512, dropout 0.1,
attention heads 4, drug/protein representation dimensions 128, and 500 epochs.
They remain paper-hyperparameter-aligned engineering baselines, not exact
official MambaTransDTA reproductions, until the official source and Supporting
Table S1 are inspected.

## Main Test Results From Paper Table 3

Davis test set:

| Method | MSE | CI | rm2 |
| --- | ---: | ---: | ---: |
| KronRLS | 0.373 | 0.866 | 0.524 |
| SimBoost | 0.282 | 0.872 | 0.644 |
| DeepDTA | 0.261 | 0.878 | 0.630 |
| TransformerCPI | 0.199 | 0.874 | 0.661 |
| GraphDTA | 0.251 | 0.876 | 0.680 |
| DeepGLSTM | 0.252 | 0.885 | 0.680 |
| DeepGS | 0.255 | 0.880 | 0.681 |
| TransVAEDTA | 0.327 | 0.866 | 0.572 |
| MambaTransDTA | 0.191 | 0.894 | 0.737 |

KIBA test set:

| Method | MSE | CI | rm2 |
| --- | ---: | ---: | ---: |
| KronRLS | 0.411 | 0.782 | 0.342 |
| SimBoost | 0.224 | 0.836 | 0.627 |
| DeepDTA | 0.194 | 0.863 | 0.673 |
| TransformerCPI | 0.201 | 0.877 | 0.720 |
| GraphDTA | 0.203 | 0.867 | 0.713 |
| DeepGLSTM | 0.179 | 0.865 | 0.675 |
| DeepGS | 0.183 | 0.862 | 0.667 |
| TransVAEDTA | 0.239 | 0.841 | 0.654 |
| MambaTransDTA | 0.173 | 0.871 | 0.722 |

The paper reports standard-deviation-like values in parentheses, but the main
planning targets above are the central values.

## Metz And BindingDB Results From Paper Table 4

| Dataset | Method | MSE | CI | rm2 |
| --- | --- | ---: | ---: | ---: |
| Metz | DeepDTA | 0.353 | 0.703 | 0.537 |
| Metz | GraphDTA | 0.317 | 0.801 | 0.620 |
| Metz | MambaTransDTA | 0.302 | 0.804 | 0.636 |
| BindingDB | DeepDTA | 0.812 | 0.795 | 0.618 |
| BindingDB | GraphDTA | 0.799 | 0.812 | 0.624 |
| BindingDB | MambaTransDTA | 0.715 | 0.817 | 0.637 |

These are useful later, but the current repository phase focuses on Davis and
KIBA first.

## Ablation Study From Paper Table 5

The ablation is reported on Davis:

| Variant | Change in MSE | MSE | CI | rm2 |
| --- | ---: | ---: | ---: | ---: |
| Model 1: remove MambaTrans block | 19.9 percent | 0.229 | 0.871 | 0.674 |
| Model 2: replace MambaTrans block with Transformer encoder | 12.6 percent | 0.215 | 0.891 | 0.698 |
| Model 3: replace Mamba preprocessing with traditional positional encoding | 12.0 percent | 0.214 | 0.885 | 0.688 |
| Full MambaTransDTA | baseline | 0.191 | 0.894 | 0.737 |

Interpretation:

- The MambaTrans encoder block contributes substantially to drug
  representation quality.
- Mamba-based positional preprocessing also contributes.
- The paper's ablation is mostly about the drug branch, not the protein branch.

This matters for our project because our proposed novelty is the opposite
controlled experiment: keep the MambaTransDTA-style SMILES branch stable and
change only the protein encoder.

## Cold-Start Evaluation

The paper evaluates three cold-start settings on Davis and KIBA:

- Drug cold-start: test drugs are unseen during training and validation.
- Target/protein cold-start: test targets are unseen during training and
  validation.
- Drug-target cold-start: both test drugs and test targets are unseen during
  training and validation.

The paper states that entity leakage must be avoided across train, validation,
and test splits. It compares MambaTransDTA with GraphDTA, DeepGLSTM, and
TransformerCPI in these settings. The full cold-start numbers are said to be in
Supporting Table S3.

Project implication:

- Our cold-drug and cold-target split generators are aligned with this idea.
- We still need an all-cold drug-target split before claiming coverage of all
  cold-start protocols from the paper.

## Case Study

The paper includes an EGFR case study:

- Five EGFR inhibitors with shared scaffold and substituent differences were
  selected from BindingDB.
- The model predicts pKd values and compares them to experimental values.
- The paper reports a small five-compound mini-set with Pearson correlation and
  MSE.
- Attention heatmaps are used to discuss local structural motifs, including
  halogen-related effects in representative compounds.

This is not required for the first Davis/KIBA baseline phase, but it is useful
as a later qualitative analysis template.

## MambaTrans Family Comparison

The paper compares several hybrid layouts:

- MambaTrans
- Attention-Mamba
- Mamba-Attention
- Mamba-only
- Transformer-only

Paper Table 7 reports that MambaTrans is best among these families on Davis and
KIBA:

| Model | Davis MSE | Davis CI | Davis rm2 | KIBA MSE | KIBA CI | KIBA rm2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaTrans | 0.191 | 0.894 | 0.737 | 0.173 | 0.875 | 0.722 |
| Attention-Mamba | 0.235 | 0.886 | 0.684 | 0.197 | 0.850 | 0.692 |
| Mamba-Attention | 0.242 | 0.840 | 0.667 | 0.201 | 0.841 | 0.682 |
| Mamba | 0.218 | 0.890 | 0.695 | 0.211 | 0.845 | 0.691 |
| Transformer | 0.215 | 0.891 | 0.698 | 0.222 | 0.836 | 0.676 |

Note: the KIBA CI value for MambaTrans in Table 7 is 0.875, while Table 3 lists
0.871 for MambaTransDTA on KIBA. Treat this as a paper-table context
difference until the official code and supporting information are inspected.

## Limitations And Future Work Mentioned By The Paper

The paper acknowledges limitations:

- Extremely rare or newly discovered targets with limited data remain hard.
- The model may capture local structural variations, but complex protein
  conformational changes and dynamic binding-process rearrangements remain
  challenging.
- Future work points toward multimodal data, 2D molecular structures, 3D drug
  conformations, protein structures, and pretrained structural embeddings.

## Reproduction Checklist For This Repository

Before claiming exact MambaTransDTA reproduction:

1. Obtain the official code/data package from the paper URL or supporting
   materials.
2. Confirm whether the official code uses the same Davis fold choice now used
   here: validation fold 1, producing 20,037 train and 5,009 validation rows.
3. Confirm optimizer, scheduler, seed policy, truncation lengths, and
   dataset-specific hyperparameters from code or Supporting Table S1.
4. Save every run's config, command, git commit, environment, metrics,
   predictions, and checkpoint.
5. Report current repository results as internal baseline results unless all
   paper-level reproduction conditions are matched.
