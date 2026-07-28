# Predict Drug Response

Predicting cancer cell line drug response from gene expression using machine learning.

This project uses gene expression data from the Cancer Dependency Map (DepMap) and PRISM drug response measurements to build and evaluate machine learning models capable of predicting how different cancer cell lines respond to small-molecule compounds.

The project is designed with a modular architecture so multiple machine learning models can be trained and evaluated using the exact same preprocessing pipeline and train/test split.

---

## Dataset

### Gene Expression

- Source: DepMap Public Release (https://depmap.org/portal/data_page/?tab=currentRelease)
- Features: RNA-seq gene expression
- ~19,000 gene expression features per cell line

### Drug Response

- Source: PRISM Repurposing Primary Screen (https://depmap.org/repurposing/)
- Target: Drug response (log fold change) for a selected treatment

The datasets are merged on `ModelID`, which uniquely identifies each cancer cell line.

---

## Project Structure

```text
predict-drug-response/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── preprocess.py
│   ├── split_data.py
│   └── evaluate.py
│
├── models/
│   └── ridge_regression/
│       └── model.py
│
├── artifacts/
│   ├── trained_models/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## Pipeline

### 1. Preprocessing

`src/preprocess.py`

- Load DepMap gene expression data
- Load PRISM drug response data
- Select a treatment
- Merge datasets using `ModelID`
- Save processed feature and target datasets

Outputs:

```
data/processed/
    features_<treatment>.csv
    target_<treatment>.csv
```

---

### 2. Train/Test Split

`src/split_data.py`

Creates a reproducible train/test split once and saves the selected `ModelID`s.

Outputs:

```
data/splits/
    train_model_ids.csv
    test_model_ids.csv
```

Every model loads these same files to ensure fair comparisons.

---

### 3. Model Training

Example:

```
models/ridge_regression/model.py
```

The model:

- loads processed data
- validates inputs
- loads the saved train/test split
- trains using cross validation
- evaluates on the held-out test set
- saves predictions
- saves metrics
- saves model coefficients
- saves diagnostic plots

---

## Evaluation

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Pearson Correlation
- Spearman Correlation

A mean predictor baseline is also evaluated for comparison.

---

## Current Model

### Ridge Regression

Pipeline:

```
StandardScaler
        ↓
Ridge Regression
```

Hyperparameters are selected using GridSearchCV with 5-fold cross validation.

---

## Outputs

Running a model produces:

```
artifacts/

trained_models/
    ridge_<treatment>.joblib

predictions/
    ridge_<treatment>.csv

metrics/
    ridge_<treatment>.json
    ridge_coefficients_<treatment>.csv

figures/
    target_distribution.png
    actual_vs_predicted.png
    residuals.png
```

---

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Preprocess the data

```bash
python -m src.preprocess
```

---

### Create the train/test split

```bash
python -m src.split_data
```

---

### Train Ridge Regression

```bash
python -m models.ridge_regression.model
```

---

## Extending the Project

Adding a new model only requires creating a new directory under `models/`.

Example:

```
models/
    ridge_regression/
    elastic_net/
    random_forest/
    xgboost/
```

Each model can reuse:

- preprocessing
- train/test split
- evaluation utilities

allowing direct comparison using identical datasets.

---

## Future Work

- Elastic Net Regression
- Random Forest Regression
- XGBoost
- Support Vector Regression
- Neural Networks
- Feature selection
- SHAP model interpretation
- Hyperparameter optimization with Optuna
