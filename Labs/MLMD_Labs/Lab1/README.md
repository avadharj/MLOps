# Lab 1 — ML Metadata Walkthrough

## What This Lab Does

This notebook (`C2_W3_Lab_1_MLMetadata.ipynb`) demonstrates how to use the
[ML Metadata (MLMD)](https://www.tensorflow.org/tfx/guide/mlmd) library
**directly** — without a full TFX pipeline — to record and query metadata about
an end-to-end ML workflow.

The lab covers all core MLMD concepts:

| Concept | What it represents |
|---|---|
| `ArtifactType` | Category of data (dataset, schema, model, …) |
| `Artifact` | A concrete instance of an artifact type |
| `ExecutionType` | A type of pipeline step (validation, training, …) |
| `Execution` | A single run of a pipeline step |
| `Event` | Input / output relationship between artifact and execution |
| `ContextType` / `Context` | A logical grouping (experiment, project, …) |
| `Attribution` / `Association` | Links artifacts and executions to a context |

### Pipeline modelled in this notebook

```
Wine CSV splits
      │
      ▼  (Data Validation execution)
  TFDV Schema ──────────────────────────────┐
      │                                     │
Wine CSV splits                             │ (both associated
      │                                     │  with "Demo" context)
      ▼  (Model Training execution)         │
  RF Model (accuracy, n_estimators) ────────┘
```

After running the notebook you can query the store to reconstruct this lineage
from any artifact backwards to its source data.

---

## Changes from the Original Lab

| Area | Original | Updated |
|---|---|---|
| **Dataset** | Chicago Taxi (downloaded from GCP) | **Wine Quality** (`sklearn.datasets.load_wine()`) |
| **Data prep** | `urllib` + `zipfile` download | `pandas` DataFrame split into `train / eval / serving` CSVs |
| **Execution types** | `Data Validation` only | `Data Validation` **+** `Model Training` |
| **Artifact types** | `DataSet`, `Schema`, `statistics` | Added **`Model`** (stores `accuracy`, `n_estimators`, `uri`) |
| **Model** | None | **`RandomForestClassifier`** (100 trees, `sklearn`) trained on Wine train split, evaluated on eval split, saved via `joblib` |
| **Context linkage** | Schema + DV execution | Schema + DV execution + RF Model + MT execution |

---

## Key Files

```
Lab1/
├── C2_W3_Lab_1_MLMetadata.ipynb   # Main notebook
├── schema.pbtxt                   # TFDV-inferred schema (generated at runtime)
├── model.joblib                   # Saved Random Forest model (generated at runtime)
├── data/
│   ├── train/data.csv             # 70 % of Wine dataset (generated at runtime)
│   ├── eval/data.csv              # 20 % of Wine dataset (generated at runtime)
│   └── serving/data.csv           # 10 % of Wine dataset (generated at runtime)
└── img/
    └── mlmd_overview.png          # MLMD architecture diagram
```

---

## Requirements

```
tensorflow
tensorflow-data-validation
ml-metadata
scikit-learn
pandas
numpy
joblib
```

---

## Running the Notebook

```bash
jupyter notebook C2_W3_Lab_1_MLMetadata.ipynb
```

Run all cells in order. The Wine dataset CSVs and the `schema.pbtxt` /
`model.joblib` files are generated at runtime — no external downloads needed.
