# Expedia Learning-to-Rank

Python pipeline for the Expedia hotel ranking assignment. The project trains grouped learning-to-rank models, builds validation metrics, and can generate Kaggle-style submission files from the Expedia train/test datasets.

## Repository Contents

- `expedia_ltr/` - reusable training, feature engineering, validation, and CLI code.
- `tests/` - focused tests for ranking validation and feature behavior.
- `notebooks/` - exploratory notebooks.
- `scripts/` - helper scripts for analysis and plotting.
- `report.tex` and `process-report.tex` - report source files.
- `requirements.txt` - Python dependencies.

Large local files are intentionally excluded from the GitHub copy, including raw/processed datasets, generated artifacts, model pickles, plots, metrics, submissions, validation predictions, caches, and virtual environments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Expedia data locally under `data/processed/` using these default names:

```text
data/processed/training_set_VU_DM.parquet
data/processed/test_set_VU_DM.parquet
```

## Run

```bash
python -m expedia_ltr
```

Useful options:

```bash
python -m expedia_ltr --help
python -m expedia_ltr --no-final
python -m expedia_ltr --skip-validation
```

## Tests

```bash
pytest
```
