import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    calculate_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_target_distribution,
    print_metrics,
)


TREATMENT_ID = "BRD-K12343256-001-08-9_2.5_HTS"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / f"features_{TREATMENT_ID}.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / f"target_{TREATMENT_ID}.csv"
)

TRAIN_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train_model_ids.csv"
)

TEST_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test_model_ids.csv"
)

MODEL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "trained_models"
    / f"elastic_net_{TREATMENT_ID}.joblib"
)

PREDICTIONS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / f"elastic_net_{TREATMENT_ID}.csv"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / f"elastic_net_{TREATMENT_ID}.json"
)

COEFFICIENTS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / f"elastic_net_coefficients_{TREATMENT_ID}.csv"
)

TARGET_PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
    / f"target_distribution_{TREATMENT_ID}.png"
)

PREDICTION_PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
    / f"elastic_net_actual_vs_predicted_{TREATMENT_ID}.png"
)

RESIDUAL_PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
    / f"elastic_net_residuals_{TREATMENT_ID}.png"
)

RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# Elastic Net is slower than Ridge because GridSearchCV searches
# combinations of alpha and l1_ratio.
ELASTIC_NET_ALPHA_VALUES = np.logspace(-4, 2, 9)

ELASTIC_NET_L1_RATIO_VALUES = [
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
]

MAX_ITERATIONS = 50_000
CONVERGENCE_TOLERANCE = 1e-4
COEFFICIENT_ZERO_TOLERANCE = 1e-10


def validate_file_exists(path: Path) -> None:
    """Raise an informative error when a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )


def load_processed_data() -> pd.DataFrame:
    """Load and merge processed features and target data."""

    validate_file_exists(FEATURES_PATH)
    validate_file_exists(TARGET_PATH)

    features = pd.read_csv(FEATURES_PATH)
    target = pd.read_csv(TARGET_PATH)

    if "ModelID" not in features.columns:
        raise ValueError(
            "The features file must contain a ModelID column."
        )

    required_target_columns = {
        "ModelID",
        "drug_response",
    }

    if not required_target_columns.issubset(target.columns):
        raise ValueError(
            "The target file must contain ModelID and drug_response."
        )

    if features["ModelID"].duplicated().any():
        raise ValueError(
            "Duplicate ModelIDs were found in the features file."
        )

    if target["ModelID"].duplicated().any():
        raise ValueError(
            "Duplicate ModelIDs were found in the target file."
        )

    data = features.merge(
        target,
        on="ModelID",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(features):
        raise ValueError(
            "Some feature rows do not have a matching target."
        )

    if len(data) != len(target):
        raise ValueError(
            "Some target rows do not have matching features."
        )

    return data


def validate_modeling_data(
    data: pd.DataFrame,
) -> None:
    """Validate feature and target values before training."""

    if data["ModelID"].isna().any():
        raise ValueError(
            "Missing ModelID values were found."
        )

    if data["drug_response"].isna().any():
        raise ValueError(
            "Missing drug-response values were found."
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in {"ModelID", "drug_response"}
    ]

    if not feature_columns:
        raise ValueError(
            "No gene-expression feature columns were found."
        )

    non_numeric_columns = (
        data[feature_columns]
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            "All model features must be numeric. "
            f"Non-numeric columns: {non_numeric_columns}"
        )

    missing_feature_count = int(
        data[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_feature_count > 0:
        raise ValueError(
            f"Features contain {missing_feature_count} missing values."
        )

    feature_values = data[feature_columns].to_numpy()

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Features contain infinite or non-finite values."
        )

    if not np.isfinite(
        data["drug_response"].to_numpy()
    ).all():
        raise ValueError(
            "The target contains infinite or non-finite values."
        )


def load_split_ids() -> tuple[pd.Series, pd.Series]:
    """Load the saved train and test ModelID lists."""

    validate_file_exists(TRAIN_IDS_PATH)
    validate_file_exists(TEST_IDS_PATH)

    train_ids_data = pd.read_csv(TRAIN_IDS_PATH)
    test_ids_data = pd.read_csv(TEST_IDS_PATH)

    if "ModelID" not in train_ids_data.columns:
        raise ValueError(
            "The training split file must contain ModelID."
        )

    if "ModelID" not in test_ids_data.columns:
        raise ValueError(
            "The test split file must contain ModelID."
        )

    train_ids = train_ids_data["ModelID"]
    test_ids = test_ids_data["ModelID"]

    if train_ids.duplicated().any():
        raise ValueError(
            "Duplicate ModelIDs were found in the training split."
        )

    if test_ids.duplicated().any():
        raise ValueError(
            "Duplicate ModelIDs were found in the test split."
        )

    overlapping_ids = set(train_ids) & set(test_ids)

    if overlapping_ids:
        raise ValueError(
            "The training and test splits overlap. "
            f"Example overlapping IDs: {list(overlapping_ids)[:10]}"
        )

    return train_ids, test_ids


def apply_saved_split(
    data: pd.DataFrame,
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create X and y datasets using the saved ModelID split."""

    available_ids = set(data["ModelID"])

    missing_train_ids = set(train_ids) - available_ids
    missing_test_ids = set(test_ids) - available_ids

    if missing_train_ids:
        raise ValueError(
            f"{len(missing_train_ids)} training ModelIDs "
            "are missing from the modeling data."
        )

    if missing_test_ids:
        raise ValueError(
            f"{len(missing_test_ids)} test ModelIDs "
            "are missing from the modeling data."
        )

    split_ids = set(train_ids) | set(test_ids)
    unused_data_ids = available_ids - split_ids

    if unused_data_ids:
        raise ValueError(
            f"{len(unused_data_ids)} modeling rows are not "
            "assigned to either split."
        )

    # Set ModelID as the index so rows are selected in the exact
    # order stored in the split files.
    indexed_data = data.set_index("ModelID")

    train_data = indexed_data.loc[
        train_ids.tolist()
    ].copy()

    test_data = indexed_data.loc[
        test_ids.tolist()
    ].copy()

    feature_columns = [
        column
        for column in data.columns
        if column not in {"ModelID", "drug_response"}
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["drug_response"]
    y_test = test_data["drug_response"]

    model_ids_train = pd.Series(
        train_data.index,
        name="ModelID",
    )

    model_ids_test = pd.Series(
        test_data.index,
        name="ModelID",
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        model_ids_train,
        model_ids_test,
    )


def create_mean_baseline(
    y_train: pd.Series,
    test_sample_count: int,
) -> np.ndarray:
    """Predict the training-set mean for every test sample."""

    training_mean = float(y_train.mean())

    return np.full(
        shape=test_sample_count,
        fill_value=training_mean,
    )


def create_elastic_net_search() -> GridSearchCV:
    """Create a cross-validated Elastic Net search."""

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "elastic_net",
                ElasticNet(
                    max_iter=MAX_ITERATIONS,
                    tol=CONVERGENCE_TOLERANCE,
                    selection="cyclic",
                ),
            ),
        ]
    )

    parameter_grid = {
        "elastic_net__alpha": ELASTIC_NET_ALPHA_VALUES,
        "elastic_net__l1_ratio": (
            ELASTIC_NET_L1_RATIO_VALUES
        ),
    }

    cross_validation = KFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )


def create_predictions_dataframe(
    model_ids_test: pd.Series,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Create the test prediction results table."""

    actual_values = y_test.to_numpy()

    return pd.DataFrame(
        {
            "ModelID": model_ids_test.to_numpy(),
            "actual_response": actual_values,
            "predicted_response": predictions,
            "residual": actual_values - predictions,
            "absolute_error": np.abs(
                actual_values - predictions
            ),
        }
    )


def create_coefficients_dataframe(
    model: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract Elastic Net coefficients for each gene feature."""

    elastic_net_model = model.named_steps[
        "elastic_net"
    ]

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": elastic_net_model.coef_,
        }
    )

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients["selected"] = (
        coefficients["absolute_coefficient"]
        > COEFFICIENT_ZERO_TOLERANCE
    )

    return coefficients.sort_values(
        by="absolute_coefficient",
        ascending=False,
    )


def convert_to_json_serializable(
    value: Any,
) -> Any:
    """Convert NumPy values into JSON-safe Python values."""

    if isinstance(value, dict):
        return {
            key: convert_to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            convert_to_json_serializable(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            convert_to_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a dictionary as standards-compliant JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_data = convert_to_json_serializable(
        data
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_data,
            file,
            indent=4,
            allow_nan=False,
        )


def save_outputs(
    model: Pipeline,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    """Save model artifacts and evaluation results."""

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COEFFICIENTS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    predictions.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_OUTPUT_PATH,
        index=False,
    )

    save_json(
        data=metrics,
        output_path=METRICS_OUTPUT_PATH,
    )


def main() -> None:
    data = load_processed_data()
    validate_modeling_data(data)

    train_ids, test_ids = load_split_ids()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        model_ids_test,
    ) = apply_saved_split(
        data=data,
        train_ids=train_ids,
        test_ids=test_ids,
    )

    print(f"Treatment: {TREATMENT_ID}")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Gene features: {X_train.shape[1]}")

    plot_target_distribution(
        y=data["drug_response"],
        output_path=TARGET_PLOT_OUTPUT_PATH,
    )

    baseline_predictions = create_mean_baseline(
        y_train=y_train,
        test_sample_count=len(y_test),
    )

    baseline_metrics = calculate_regression_metrics(
        y_true=y_test,
        y_pred=baseline_predictions,
    )

    elastic_net_search = create_elastic_net_search()

    elastic_net_search.fit(
        X_train,
        y_train,
    )

    best_model = elastic_net_search.best_estimator_
    best_index = elastic_net_search.best_index_

    cross_validation_mae = (
        -elastic_net_search.best_score_
    )

    cross_validation_mae_std = (
        elastic_net_search.cv_results_[
            "std_test_score"
        ][best_index]
    )

    elastic_net_predictions = best_model.predict(
        X_test
    )

    elastic_net_metrics = (
        calculate_regression_metrics(
            y_true=y_test,
            y_pred=elastic_net_predictions,
        )
    )

    print_metrics(
        model_name="Mean baseline",
        metrics=baseline_metrics,
    )

    print_metrics(
        model_name="Elastic Net",
        metrics=elastic_net_metrics,
    )

    best_alpha = float(
        elastic_net_search.best_params_[
            "elastic_net__alpha"
        ]
    )

    best_l1_ratio = float(
        elastic_net_search.best_params_[
            "elastic_net__l1_ratio"
        ]
    )

    elastic_net_model = best_model.named_steps[
        "elastic_net"
    ]

    nonzero_coefficient_count = int(
        np.count_nonzero(
            np.abs(elastic_net_model.coef_)
            > COEFFICIENT_ZERO_TOLERANCE
        )
    )

    zero_coefficient_count = int(
        X_train.shape[1]
        - nonzero_coefficient_count
    )

    print(
        f"\nBest Elastic Net alpha: "
        f"{best_alpha:.6f}"
    )

    print(
        f"Best Elastic Net l1_ratio: "
        f"{best_l1_ratio:.2f}"
    )

    print(
        "Cross-validation MAE:",
        f"{cross_validation_mae:.4f} ± "
        f"{cross_validation_mae_std:.4f}",
    )

    print(
        "Selected gene features:",
        f"{nonzero_coefficient_count} / "
        f"{X_train.shape[1]}",
    )

    prediction_results = (
        create_predictions_dataframe(
            model_ids_test=model_ids_test,
            y_test=y_test,
            predictions=elastic_net_predictions,
        )
    )

    coefficients = create_coefficients_dataframe(
        model=best_model,
        feature_names=X_train.columns.tolist(),
    )

    metrics = {
        "model": "elastic_net",
        "treatment_id": TREATMENT_ID,
        "total_sample_count": int(len(data)),
        "training_sample_count": int(
            len(X_train)
        ),
        "test_sample_count": int(len(X_test)),
        "feature_count": int(
            X_train.shape[1]
        ),
        "selected_feature_count": (
            nonzero_coefficient_count
        ),
        "zero_coefficient_count": (
            zero_coefficient_count
        ),
        "selected_feature_fraction": float(
            nonzero_coefficient_count
            / X_train.shape[1]
        ),
        "split_files": {
            "train": str(
                TRAIN_IDS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "test": str(
                TEST_IDS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "random_state": RANDOM_STATE,
        "cross_validation_folds": (
            CROSS_VALIDATION_FOLDS
        ),
        "scoring": (
            "negative_mean_absolute_error"
        ),
        "tested_alpha_values": (
            ELASTIC_NET_ALPHA_VALUES
        ),
        "tested_l1_ratio_values": (
            ELASTIC_NET_L1_RATIO_VALUES
        ),
        "best_parameters": (
            elastic_net_search.best_params_
        ),
        "cross_validation_mae": float(
            cross_validation_mae
        ),
        "cross_validation_mae_std": float(
            cross_validation_mae_std
        ),
        "baseline_metrics": baseline_metrics,
        "test_metrics": elastic_net_metrics,
    }

    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=elastic_net_predictions,
        output_path=PREDICTION_PLOT_OUTPUT_PATH,
    )

    plot_residuals(
        y_true=y_test,
        y_pred=elastic_net_predictions,
        output_path=RESIDUAL_PLOT_OUTPUT_PATH,
    )

    save_outputs(
        model=best_model,
        metrics=metrics,
        predictions=prediction_results,
        coefficients=coefficients,
    )

    print("\nSaved outputs:")
    print(f"  Model: {MODEL_OUTPUT_PATH}")
    print(f"  Metrics: {METRICS_OUTPUT_PATH}")
    print(
        f"  Predictions: "
        f"{PREDICTIONS_OUTPUT_PATH}"
    )
    print(
        f"  Coefficients: "
        f"{COEFFICIENTS_OUTPUT_PATH}"
    )
    print(
        f"  Prediction plot: "
        f"{PREDICTION_PLOT_OUTPUT_PATH}"
    )
    print(
        f"  Residual plot: "
        f"{RESIDUAL_PLOT_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()