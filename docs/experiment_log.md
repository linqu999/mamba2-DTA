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
