# Experiment Log

Use `python scripts/summarize_runs.py` to generate the machine-readable summary
under `results/tables/`. This file records human decisions and interpretation.

| run_id | dataset | split | model | seed | status | key result | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20260522_043047_davis_mambatransdta_table1_mtdta_cnn_seed42 | Davis | mambatransdta-table1 | mtdta_cnn | 42 | complete | test MSE 0.2975 / CI 0.8710 / rm2 0.5560 | Engineering baseline v0; uses lr 1e-4, batch 64, 100 epochs. Kept as a valid baseline record, but not paper-hyperparameter aligned. |

## Planned Formal Baselines

Next formal CNN baselines should use the `*_cnn_paper_hparams.yaml` configs,
which align the available settings with MambaTransDTA Table 2: learning rate
0.0005, batch size 512, dropout 0.1, attention heads 4, drug/protein
representation dimensions 128, and 500 epochs.

## Formal Run Requirements

Do not treat a run as publication-ready unless it contains:

- `config.json`, `config_source.txt`, and `command.txt`.
- `environment.txt` and `git_commit.txt`.
- `train.log`.
- `metrics.csv` with per-epoch `valid_mse`, `valid_rmse`, `valid_mae`,
  `valid_ci`, `valid_rm2`, and `is_best`.
- `metrics.json` and `metrics_summary.json`.
- `best.pt`.
- `predictions_valid.csv`, `predictions_valid_best.csv`, and
  `predictions_test.csv`.
- `artifact_manifest.json`.

Validation metrics are used for model selection. Test metrics are calculated
only once from the best validation checkpoint and should not be used for
hyperparameter tuning.
