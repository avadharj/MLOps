# MLflow Lab 1: Wine Quality Experiment Tracking (Revamped)

This lab now tracks experiments on a **new default dataset/model setup**:

- Default dataset: **white wine quality** (`winequality-white.csv`)
- Default model: **RandomForestRegressor**

Backward compatibility is preserved:

- You can still run **ElasticNet**
- You can switch between **red** and **white** datasets from CLI arguments

## What Changed

The training script in `linear_regression.py` was upgraded to:

- support `dataset_type` (`white` or `red`)
- support `model_type` (`rf` or `elasticnet`)
- log model/data choices as MLflow params
- keep evaluation metrics (`rmse`, `mae`, `r2`) in MLflow
- log the trained sklearn model artifact with signature

## Project Files

- `linear_regression.py`: main training + MLflow tracking script
- `serving.py`: model logging requirements demonstration
- `requirements.txt`: Python dependencies
- notebooks (`starter.ipynb`, `linear_regression.ipynb`, `serving.ipynb`)

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Experiments

### Default run (new baseline)

Runs RandomForest on the white wine dataset:

```bash
python linear_regression.py
```

### Choose model + dataset

Argument order:

`python linear_regression.py <model_type> <dataset_type> [alpha] [l1_ratio]`

- `model_type`: `rf` or `elasticnet`
- `dataset_type`: `white` or `red`
- `alpha`, `l1_ratio`: only used when `model_type=elasticnet`

Examples:

```bash
# RandomForest on red wine
python linear_regression.py rf red

# ElasticNet on white wine
python linear_regression.py elasticnet white 0.5 0.5

# ElasticNet on red wine
python linear_regression.py elasticnet red 0.1 0.8
```

## View Results in MLflow UI

From this lab directory:

```bash
mlflow ui
```

Open:

`http://127.0.0.1:5000`

In the UI, compare runs by:

- model type (`rf` vs `elasticnet`)
- dataset type (`white` vs `red`)
- tracked metrics (`rmse`, `mae`, `r2`)

## Notes

- If your tracking URI is a local file store, model registry registration is skipped.
- The script logs to model artifact path `model` for every run.

## Suggested Lab Experiments

- Compare `rf` vs `elasticnet` on the same dataset
- Compare `red` vs `white` using the same model
- Tune ElasticNet (`alpha`, `l1_ratio`) and inspect metric trends

