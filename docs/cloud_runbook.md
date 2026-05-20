# Cloud Validation Runbook

This runbook is for the first cloud-platform validation pass. The goal is not
formal training yet. The goal is to prove that code, data, PyTorch, and the
CNN baseline training loop work end to end before spending GPU time on Mamba2.

## Gate 1: Code And Data On Cloud

Choose one transfer path.

### Option A: Git For Code, Regenerate Data On Cloud

Use this when the cloud machine has stable GitHub access.

```bash
cd /root/autodl-tmp
git clone <YOUR_REPO_URL> mamba2-DTA
cd mamba2-DTA
```

Because `data/raw/`, `data/processed/`, and `data/splits/` are ignored by Git,
regenerate data on the cloud:

```bash
python scripts/download_deepdta_data.py --from-zip data/cache/DeepDTA-master.zip
```

If `data/cache/DeepDTA-master.zip` is not present, download it first:

```bash
mkdir -p data/cache
curl -L --retry 5 --retry-delay 5 --connect-timeout 60 \
  -o data/cache/DeepDTA-master.zip \
  https://codeload.github.com/hkmztrk/DeepDTA/zip/refs/heads/master
python scripts/download_deepdta_data.py --from-zip data/cache/DeepDTA-master.zip
```

Then process datasets and splits:

```bash
python scripts/01_prepare_data.py --deepdta-raw-dir data/raw/deepdta/davis --dataset davis --output data/processed/davis.csv --metadata data/processed/davis_metadata.json
python scripts/01_prepare_data.py --deepdta-raw-dir data/raw/deepdta/kiba --dataset kiba --output data/processed/kiba.csv --metadata data/processed/kiba_metadata.json

python scripts/02_make_splits.py --input data/processed/davis.csv --output data/splits/davis/official_deepdta.csv --metadata data/splits/davis/official_deepdta_metadata.json --split official-deepdta --folds-dir data/raw/deepdta/davis/folds --valid-fold 0
python scripts/02_make_splits.py --input data/processed/kiba.csv --output data/splits/kiba/official_deepdta.csv --metadata data/splits/kiba/official_deepdta_metadata.json --split official-deepdta --folds-dir data/raw/deepdta/kiba/folds --valid-fold 0
```

### Option B: Upload The Whole Project Folder

Use this when cloud GitHub access is unreliable. Upload the local project folder
including `data/raw/deepdta`, `data/processed`, and `data/splits`.

Expected key files after upload:

```bash
ls data/raw/deepdta/manifest.json
ls data/processed/davis.csv data/processed/kiba.csv
ls data/splits/davis/official_deepdta.csv data/splits/kiba/official_deepdta.csv
```

### Gate 1 Acceptance

Run:

```bash
python - <<'PY'
import pandas as pd
for ds in ["davis", "kiba"]:
    df = pd.read_csv(f"data/processed/{ds}.csv")
    print(ds, len(df), df.drug_id.nunique(), df.target_id.nunique(), df.protein_sequence.str.len().max())
PY
```

Expected:

```text
davis 30056 68 442 2549
kiba 118254 2111 229 4128
```

## Gate 2: Environment And Tests

Create a clean environment:

```bash
conda create -n bimamba2dta python=3.10 -y
conda activate bimamba2dta
```

Install CPU/GPU PyTorch according to the cloud image. For a CUDA 11.8 image:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

If the cloud image already includes PyTorch, do not reinstall immediately; just
check it first.

Install the project and test dependencies:

```bash
pip install -e ".[dev]"
python scripts/00_check_env.py
pytest -q
```

### Gate 2 Acceptance

`00_check_env.py` should report:

- `torch: yes`
- `cuda_available: true` if using a GPU image, or `false` if doing CPU-only
  smoke validation

`pytest -q` should pass. It is acceptable for Mamba2 tests to skip before
`mamba_ssm` is installed.

## Gate 3: CNN Baseline Debug Train

Run a tiny CPU debug first, even on a GPU machine:

```bash
python scripts/03_train.py --config configs/experiment/debug_cpu.yaml --limit-batches 2
```

Find the latest run:

```bash
ls -td runs/* | head -1
```

Inspect required artifacts:

```bash
RUN_DIR=$(ls -td runs/* | head -1)
ls "$RUN_DIR"/metrics.json "$RUN_DIR"/metrics.csv "$RUN_DIR"/predictions_valid.csv "$RUN_DIR"/best.pt "$RUN_DIR"/train.log "$RUN_DIR"/environment.txt "$RUN_DIR"/command.txt
cat "$RUN_DIR"/metrics.json
head "$RUN_DIR"/predictions_valid.csv
cat "$RUN_DIR"/train.log
```

### Gate 3 Acceptance

The run directory must contain:

- `metrics.json`
- `metrics.csv`
- `predictions_valid.csv`
- `best.pt`
- `train.log`
- `environment.txt`
- `command.txt`

The debug train does not need good metrics. It only needs finite loss and valid
prediction rows.

## After The Three Gates

Only after all three gates pass:

1. Install Mamba dependencies.
2. Run Mamba2 import/shape tests.
3. Run `debug_gpu.yaml`.
4. Start Davis CNN baseline training.

Suggested Mamba install attempt:

```bash
pip install causal-conv1d
pip install mamba-ssm
python scripts/00_check_env.py
pytest -q
```

Do not start formal KIBA or full-length BiMamba2 training before the CNN debug
run has produced all required artifacts.
