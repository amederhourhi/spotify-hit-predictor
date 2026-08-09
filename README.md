# Spotify Hit Predictor

A practical machine-learning study that explores how audio features can be used to estimate a synthetic popularity score. The project moves from a simple baseline model to a more robust preprocessing pipeline and finally to a tree-based ensemble approach, all while keeping the workflow easy to follow for beginners.

This repository is intentionally educational: it uses synthetic data to demonstrate core ML concepts such as data generation, cleaning, preprocessing, train/validation/test splitting, feature engineering, and model evaluation.

---

## Why this project exists

The goal is not to build a production-grade music recommendation engine. Instead, the project demonstrates a full ML workflow in a compact, understandable format:

- generate realistic-looking tabular data
- introduce common data quality issues
- preprocess mixed numeric and categorical features
- compare simple and more expressive models
- evaluate models using meaningful metrics

It is especially useful for learning how data science projects evolve from a naive first version to a more professional pipeline.

---

## What the project does

The repository trains regression models to predict a synthetic target column named popularity_score using features such as:

- tempo
- danceability
- energy
- valence
- duration_ms
- genre

The workflow progresses through three versions:

1. V1: a simple baseline regression model on clean data
2. V2: a structured preprocessing pipeline for messy data
3. V3: a stronger non-linear model comparison using train/validation/test splits

---

## Project structure

```text
spotify-hit-predictor/
├── data/
│   ├── raw/
│   │   ├── synthetic_spotify_v1.csv
│   │   └── synthetic_spotify_v2_messy.csv
│   └── processed/
├── notebooks/
├── src/
│   ├── generate_v1_data.py
│   ├── generate_v2_data.py
│   ├── train_v1_model.py
│   ├── train_v2_pipeline.py
│   └── train_v3_rf.py
└── README.md
```

---

## Dataset overview

The synthetic datasets are designed to resemble Spotify-like track metadata and audio characteristics.

### Input features

| Feature | Type | Description |
| --- | --- | --- |
| tempo | numeric | Beats per minute |
| danceability | numeric | Value between 0 and 1 |
| energy | numeric | Value between 0 and 1 |
| valence | numeric | Musical positiveness, between 0 and 1 |
| duration_ms | numeric | Track duration in milliseconds |
| genre | categorical | One of several synthetic genres |

### Target

| Target | Type | Description |
| --- | --- | --- |
| popularity_score | numeric | Synthetic popularity score, scaled roughly from 0 to 100 |

---

## Workflow by stage

### 1. Data generation

The project begins by creating synthetic data so the modeling process is fully reproducible.

- [src/generate_v1_data.py](src/generate_v1_data.py) creates a clean dataset with numeric features and a target signal derived from them.
- [src/generate_v2_data.py](src/generate_v2_data.py) takes that dataset and introduces realistic issues such as:
  - missing values
  - invalid durations
  - a categorical genre column

This step teaches an important lesson: real-world data is rarely clean, so preprocessing is as important as the model itself.

### 2. Baseline modeling

- [src/train_v1_model.py](src/train_v1_model.py) trains a simple linear regression model on the clean dataset.
- It is the first version of the project and acts as a baseline for comparison.
- The script reports MAE, RMSE, and R² so beginners can see how model quality is measured.

### 3. Preprocessing pipeline

- [src/train_v2_pipeline.py](src/train_v2_pipeline.py) introduces a more professional pipeline architecture.
- It uses scikit-learn tools to handle:
  - missing numeric values with median imputation
  - categorical values with constant imputation and one-hot encoding
  - feature scaling
- The model is still linear regression, but the workflow is dramatically more realistic.

### 4. Advanced comparison

- [src/train_v3_rf.py](src/train_v3_rf.py) builds a stronger experiment.
- It creates train/validation/test splits and compares:
  - Linear Regression
  - Random Forest Regressor
- The script evaluates both models on validation data and then reports the final performance on the untouched test set.

---

## How to run the project

### Prerequisites

Make sure you have Python 3.10+ installed.

### Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install numpy pandas scikit-learn
```

### Generate the data and train the models

Run the scripts from the project root:

```bash
python src/generate_v1_data.py
python src/generate_v2_data.py
python src/train_v1_model.py
python src/train_v2_pipeline.py
python src/train_v3_rf.py
```

---

## Verified results

The current project run produced the following results:

| Experiment | Metric | Result |
| --- | --- | --- |
| V1 Baseline | MAE | 4.47 |
| V1 Baseline | RMSE | 5.61 |
| V1 Baseline | R² | 0.8285 |
| V2 Pipeline | MAE | 4.71 |
| V2 Pipeline | R² | 0.8126 |
| V3 Validation - Linear Regression | MAE | 4.32 |
| V3 Validation - Linear Regression | R² | 0.8419 |
| V3 Validation - Random Forest | MAE | 4.56 |
| V3 Validation - Random Forest | R² | 0.8278 |
| V3 Test Set - Random Forest | MAE | 4.84 |
| V3 Test Set - Random Forest | R² | 0.7927 |

These numbers are not meant to be state-of-the-art; they demonstrate a realistic modeling progression and show how different design choices affect performance.

---

## Why this is a good learning project

This repository is a strong educational example because it exposes several important ideas in a compact form:

- data generation and experiment reproducibility
- the difference between clean and messy data
- preprocessing for both numeric and categorical features
- train/validation/test separation
- model comparison and metric-driven decision making

For beginners, it offers a clear path from code to intuition. For more experienced readers, it is a compact reference for how a simple regression problem can be structured in a professional way.

---

## Recommended learning path

If you are new to machine learning, follow this order:

1. Read the data generation scripts first
2. Run the baseline model
3. Study the pipeline version
4. Compare the random forest experiment
5. Then explore how changing preprocessing or model parameters affects the metrics

---

## Notes and next steps

Possible extensions for future versions include:

- adding feature interaction terms
- trying XGBoost or LightGBM
- performing hyperparameter tuning
- saving trained models to disk
- creating a small web app or notebook-based demo

The repository is intentionally simple so that the core ideas remain clear and accessible.

---

## Summary

This project is a compact, well-structured introduction to supervised regression with scikit-learn. It shows how a machine-learning workflow can evolve from a minimal baseline into a more robust, production-style pipeline while remaining approachable for learners.
