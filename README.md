# Predict Drug Response

Predicting cancer cell line drug response from gene expression using machine learning.

This project uses **Cancer Dependency Map (DepMap)** gene expression data and **PRISM Repurposing** drug response measurements to train and compare multiple machine learning models capable of predicting how cancer cell lines respond to therapeutic compounds.

The repository provides a reproducible end-to-end pipeline for:

- Preprocessing DepMap and PRISM datasets
- Creating reproducible train/test splits
- Training multiple machine learning models
- Evaluating model performance using identical datasets
- Comparing feature selection strategies
- Generating artifacts and visualizations for downstream analysis

## Current Models

- Ridge Regression
- Elastic Net Regression
- Random Forest Feature Selection → Ridge Regression

---

# Dataset

## Gene Expression

**Source:** Cancer Dependency Map (DepMap)

Downloaded files:

- `Gene.csv`
- `Model.csv`
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `PortalCompounds.csv`

Features:

- RNA-seq gene expression
- ~19,000 protein-coding genes
- One row per cancer cell line

---

## Drug Response

**Source:** PRISM Repurposing Primary Screen

Downloaded files:

- `primary-screen-cell-line-info.csv`
- `primary-screen-replicate-collapsed-logfold-change.csv`
- `primary-screen-replicate-collapsed-treatment-info.csv`

Target:

- Drug response (log fold change) for a selected compound

The datasets are merged using `ModelID`, which uniquely identifies each cancer cell line.

---

# Project Structure

```text
predict-drug-response/

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
│   ├── ridge_regression/
│   ├── elastic_net/
│   └── random_forest/
│
├── artifacts/
│   ├── trained_models/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Pipeline

## 1. Data Preprocessing

`src/preprocess.py`

The preprocessing pipeline:

- Loads DepMap gene expression data
- Loads PRISM drug response data
- Selects a treatment
- Merges datasets using `ModelID`
- Validates data integrity
- Generates modeling datasets

Outputs:

```text
data/processed/

<treatment-id>/
    features.csv
    target.csv
    modeling_data.csv
    metadata.json
```

---

## 2. Train/Test Split

`src/split_data.py`

Creates a reproducible train/test split once and saves the selected `ModelID`s.

Outputs:

```text
data/splits/

<treatment-id>/
    train_model_ids.csv
    test_model_ids.csv
```

Every model uses the same train/test split, ensuring fair comparisons between algorithms.

---

## 3. Model Training

Each model follows the same workflow:

- Load processed data
- Validate inputs
- Load the saved train/test split
- Train using cross-validation
- Evaluate on the held-out test set
- Save predictions
- Save metrics
- Save diagnostic plots

### Ridge Regression

```text
StandardScaler
        ↓
Ridge Regression
```

A linear regression model using L2 regularization.

---

### Elastic Net Regression

```text
StandardScaler
        ↓
Elastic Net
```

Combines L1 and L2 regularization to simultaneously perform regression and embedded feature selection, automatically selecting informative genes by shrinking many coefficients to zero.

---

### Random Forest → Ridge Regression

```text
Random Forest
        ↓
Rank genes by importance
        ↓
Select Top K genes
        ↓
StandardScaler
        ↓
Ridge Regression
```

Random Forest is used only for nonlinear feature ranking. The selected genes are then used to train a Ridge regression model.

---

# Evaluation

All models are evaluated on the same held-out test set using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Pearson Correlation
- Spearman Correlation

A mean-prediction baseline is also evaluated for comparison.

Hyperparameters are selected using 5-fold cross-validation.

---

# Outputs

Running a model generates:

```text
artifacts/

trained_models/
    <model>_<treatment>.joblib

predictions/
    <model>_<treatment>.csv

metrics/
    <model>_<treatment>.json
    <model>_coefficients_<treatment>.csv

figures/
    target_distribution_<treatment>.png
    actual_vs_predicted_<treatment>.png
    residuals_<treatment>.png
```

Additional outputs are produced for model-specific analyses such as:

- Elastic Net selected genes
- Random Forest feature rankings
- Random Forest selected genes
- Cross-validation summaries

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Preprocess the data

```bash
python -m src.preprocess
```

---

## Create the train/test split

```bash
python -m src.split_data \
    --treatment-id BRD-K12343256-001-08-9_2.5_HTS
```

---

## Train all models

```bash
python main.py \
    --treatment-id BRD-K12343256-001-08-9_2.5_HTS
```

---

## Train an individual model

### Ridge Regression

```bash
python -m models.ridge_regression.model \
    --treatment-id BRD-K12343256-001-08-9_2.5_HTS
```

### Elastic Net

```bash
python -m models.elastic_net.model \
    --treatment-id BRD-K12343256-001-08-9_2.5_HTS
```

### Random Forest → Ridge

```bash
python -m models.random_forest.model \
    --treatment-id BRD-K12343256-001-08-9_2.5_HTS
```

---

# Model Comparison

This project investigates different strategies for predicting drug response from high-dimensional gene expression data.

| Model | Question Answered |
|--------|-------------------|
| Ridge Regression | Can a linear model learn from all available genes? |
| Elastic Net | Can sparse linear feature selection improve prediction while identifying informative genes? |
| Random Forest → Ridge | Can nonlinear feature importance improve a linear prediction model? |

All models share the same preprocessing pipeline and train/test split, enabling direct and reproducible comparisons.

---

# Future Work

- Evaluate additional PRISM compounds
- Compare model performance across many drugs
- Deploy an interactive AWS dashboard for exploring results
- SHAP-based model interpretation
- XGBoost
- Support Vector Regression
- Neural Networks
- Graph Neural Networks using biological pathway information
- Pathway-guided regularization (e.g., GELnet)
- Hyperparameter optimization with Optuna
