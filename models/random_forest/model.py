import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    calculate_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_target_distribution,
    print_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

TOP_K_VALUES = [
    100,
    250,
    500,
    1000,
    2000,
    5000,
    10000,
    15000,
    19215,
]

RIDGE_ALPHA_VALUES = np.logspace(
    -2,
    6,
    17,
)

RANDOM_FOREST_ESTIMATORS = 300
RANDOM_FOREST_MAX_FEATURES = "sqrt"
RANDOM_FOREST_MIN_SAMPLES_LEAF = 2

# Grid-search parallelism is handled manually through fold iteration.
# Random Forest can use all available cores during each fit.
RANDOM_FOREST_N_JOBS = -1


@dataclass(frozen=True)
class ModelPaths:
    """Input and output paths for one treatment."""

    features: Path
    target: Path
    train_ids: Path
    test_ids: Path
    model_output: Path
    predictions_output: Path
    metrics_output: Path
    feature_ranking_output: Path
    selected_genes_output: Path
    cross_validation_results_output: Path
    target_plot_output: Path
    prediction_plot_output: Path
    residual_plot_output: Path
    feature_selection_plot_output: Path


def build_paths(treatment_id: str) -> ModelPaths:
    """Build all Random Forest → Ridge paths for a treatment."""

    treatment_output = (
        PROJECT_ROOT / "artifacts" / "treatments" / treatment_id
    )
    model_output = treatment_output / "random_forest_ridge"

    return ModelPaths(
        features=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"features_{treatment_id}.csv"
        ),
        target=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"target_{treatment_id}.csv"
        ),
        train_ids=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "train_model_ids.csv"
        ),
        test_ids=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "test_model_ids.csv"
        ),
        model_output=model_output / "model.joblib",
        predictions_output=model_output / "predictions.csv",
        metrics_output=model_output / "metrics.json",
        feature_ranking_output=model_output / "feature_ranking.csv",
        selected_genes_output=model_output / "selected_genes.csv",
        cross_validation_results_output=(
            model_output / "cross_validation_results.csv"
        ),
        target_plot_output=(
            treatment_output / "figures" / "target_distribution.png"
        ),
        prediction_plot_output=(
            model_output / "figures" / "actual_vs_predicted.png"
        ),
        residual_plot_output=(
            model_output / "figures" / "residuals.png"
        ),
        feature_selection_plot_output=(
            model_output / "figures" / "feature_selection.png"
        ),
    )


class RandomForestRidgeModel(
    BaseEstimator,
    RegressorMixin,
):
    """
    Fit a Random Forest to rank features, retain the top K features,
    and train Ridge regression using only those selected features.
    """

    def __init__(
        self,
        top_k: int,
        ridge_alpha: float,
        random_state: int = RANDOM_STATE,
        n_estimators: int = RANDOM_FOREST_ESTIMATORS,
        max_features: str | float | int = (
            RANDOM_FOREST_MAX_FEATURES
        ),
        min_samples_leaf: int = (
            RANDOM_FOREST_MIN_SAMPLES_LEAF
        ),
        n_jobs: int = RANDOM_FOREST_N_JOBS,
    ) -> None:
        self.top_k = top_k
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "RandomForestRidgeModel":
        """Fit Random Forest selection and the final Ridge model."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame so feature names "
                "can be retained."
            )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.top_k > X.shape[1]:
            raise ValueError(
                f"top_k={self.top_k} exceeds the number "
                f"of available features ({X.shape[1]})."
            )

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        self.random_forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.random_forest_.fit(
            X,
            y,
        )

        self.feature_importances_ = (
            self.random_forest_.feature_importances_
        )

        ranked_indices = np.argsort(
            self.feature_importances_
        )[::-1]

        self.selected_feature_indices_ = (
            ranked_indices[: self.top_k]
        )

        self.selected_feature_names_ = (
            self.feature_names_in_[
                self.selected_feature_indices_
            ]
        )

        selected_X = X.loc[
            :,
            self.selected_feature_names_,
        ]

        self.ridge_pipeline_ = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=self.ridge_alpha,
                    ),
                ),
            ]
        )

        self.ridge_pipeline_.fit(
            selected_X,
            y,
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Predict using the selected features and Ridge pipeline."""

        if not hasattr(
            self,
            "ridge_pipeline_",
        ):
            raise RuntimeError(
                "The model must be fit before prediction."
            )

        missing_features = (
            set(self.selected_feature_names_)
            - set(X.columns)
        )

        if missing_features:
            raise ValueError(
                "Prediction data is missing selected features. "
                f"Examples: {list(missing_features)[:10]}"
            )

        selected_X = X.loc[
            :,
            self.selected_feature_names_,
        ]

        return self.ridge_pipeline_.predict(
            selected_X
        )


def validate_file_exists(
    path: Path,
) -> None:
    """Raise an informative error when a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )


def load_processed_data(paths: ModelPaths) -> pd.DataFrame:
    """Load and merge processed feature and target data."""

    validate_file_exists(paths.features)
    validate_file_exists(paths.target)

    features = pd.read_csv(paths.features)
    target = pd.read_csv(paths.target)

    if "ModelID" not in features.columns:
        raise ValueError(
            "The features file must contain a ModelID column."
        )

    required_target_columns = {
        "ModelID",
        "drug_response",
    }

    if not required_target_columns.issubset(
        target.columns
    ):
        raise ValueError(
            "The target file must contain ModelID "
            "and drug_response."
        )

    if features["ModelID"].duplicated().any():
        raise ValueError(
            "Duplicate ModelIDs were found in the "
            "features file."
        )

    if target["ModelID"].duplicated().any():
        raise ValueError(
            "Duplicate ModelIDs were found in the "
            "target file."
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
        if column not in {
            "ModelID",
            "drug_response",
        }
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
            f"Features contain {missing_feature_count} "
            "missing values."
        )

    feature_values = data[
        feature_columns
    ].to_numpy()

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Features contain infinite or non-finite values."
        )

    target_values = data[
        "drug_response"
    ].to_numpy()

    if not np.isfinite(target_values).all():
        raise ValueError(
            "The target contains infinite or "
            "non-finite values."
        )


def load_split_ids(paths: ModelPaths) -> tuple[
    pd.Series,
    pd.Series,
]:
    """Load the saved train and test ModelID lists."""

    validate_file_exists(paths.train_ids)
    validate_file_exists(paths.test_ids)

    train_ids_data = pd.read_csv(
        paths.train_ids
    )

    test_ids_data = pd.read_csv(
        paths.test_ids
    )

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
            "Duplicate ModelIDs were found in the "
            "training split."
        )

    if test_ids.duplicated().any():
        raise ValueError(
            "Duplicate ModelIDs were found in the "
            "test split."
        )

    overlapping_ids = (
        set(train_ids)
        & set(test_ids)
    )

    if overlapping_ids:
        raise ValueError(
            "The training and test splits overlap. "
            f"Example IDs: {list(overlapping_ids)[:10]}"
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
    """Create X and y datasets using saved ModelID splits."""

    available_ids = set(
        data["ModelID"]
    )

    missing_train_ids = (
        set(train_ids)
        - available_ids
    )

    missing_test_ids = (
        set(test_ids)
        - available_ids
    )

    if missing_train_ids:
        raise ValueError(
            f"{len(missing_train_ids)} training "
            "ModelIDs are missing from the data."
        )

    if missing_test_ids:
        raise ValueError(
            f"{len(missing_test_ids)} test "
            "ModelIDs are missing from the data."
        )

    split_ids = (
        set(train_ids)
        | set(test_ids)
    )

    unused_data_ids = (
        available_ids
        - split_ids
    )

    if unused_data_ids:
        raise ValueError(
            f"{len(unused_data_ids)} modeling rows "
            "are not assigned to either split."
        )

    indexed_data = data.set_index(
        "ModelID"
    )

    train_data = indexed_data.loc[
        train_ids.tolist()
    ].copy()

    test_data = indexed_data.loc[
        test_ids.tolist()
    ].copy()

    feature_columns = [
        column
        for column in data.columns
        if column not in {
            "ModelID",
            "drug_response",
        }
    ]

    X_train = train_data[
        feature_columns
    ]

    X_test = test_data[
        feature_columns
    ]

    y_train = train_data[
        "drug_response"
    ]

    y_test = test_data[
        "drug_response"
    ]

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
    """Predict the training mean for every test sample."""

    training_mean = float(
        y_train.mean()
    )

    return np.full(
        shape=test_sample_count,
        fill_value=training_mean,
    )


def validate_top_k_values(
    feature_count: int,
) -> list[int]:
    """Return valid, unique Top K values."""

    valid_values = sorted(
        {
            int(k)
            for k in TOP_K_VALUES
            if 0 < int(k) <= feature_count
        }
    )

    if not valid_values:
        raise ValueError(
            "No valid Top K values remain after "
            "checking the feature count."
        )

    return valid_values


def create_random_forest() -> RandomForestRegressor:
    """Create the Random Forest feature-ranking model."""

    return RandomForestRegressor(
        n_estimators=RANDOM_FOREST_ESTIMATORS,
        max_features=RANDOM_FOREST_MAX_FEATURES,
        min_samples_leaf=(
            RANDOM_FOREST_MIN_SAMPLES_LEAF
        ),
        random_state=RANDOM_STATE,
        n_jobs=RANDOM_FOREST_N_JOBS,
    )


def perform_cross_validation_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Select Top K and Ridge alpha using five-fold CV.

    Random Forest ranking is fit only on the training portion
    of each fold. This prevents validation-data leakage.
    """

    top_k_values = validate_top_k_values(
        feature_count=X_train.shape[1]
    )

    cross_validation = KFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results: list[
        dict[str, Any]
    ] = []

    total_candidates = (
        len(top_k_values)
        * len(RIDGE_ALPHA_VALUES)
    )

    print(
        f"Evaluating {len(top_k_values)} Top K values "
        f"and {len(RIDGE_ALPHA_VALUES)} Ridge alpha values."
    )

    print(
        f"Total parameter combinations: "
        f"{total_candidates}"
    )

    print(
        f"Total Ridge fits across "
        f"{CROSS_VALIDATION_FOLDS} folds: "
        f"{total_candidates * CROSS_VALIDATION_FOLDS}"
    )

    for fold_number, (
        training_indices,
        validation_indices,
    ) in enumerate(
        cross_validation.split(
            X_train,
            y_train,
        ),
        start=1,
    ):
        print(
            f"\nCross-validation fold "
            f"{fold_number}/{CROSS_VALIDATION_FOLDS}"
        )

        X_fold_train = X_train.iloc[
            training_indices
        ]

        X_fold_validation = X_train.iloc[
            validation_indices
        ]

        y_fold_train = y_train.iloc[
            training_indices
        ]

        y_fold_validation = y_train.iloc[
            validation_indices
        ]

        random_forest = create_random_forest()

        print(
            "  Fitting Random Forest "
            "for feature ranking..."
        )

        random_forest.fit(
            X_fold_train,
            y_fold_train,
        )

        ranked_feature_indices = np.argsort(
            random_forest.feature_importances_
        )[::-1]

        for top_k in top_k_values:
            selected_indices = (
                ranked_feature_indices[:top_k]
            )

            selected_feature_names = (
                X_train.columns[
                    selected_indices
                ]
            )

            X_selected_train = (
                X_fold_train.loc[
                    :,
                    selected_feature_names,
                ]
            )

            X_selected_validation = (
                X_fold_validation.loc[
                    :,
                    selected_feature_names,
                ]
            )

            scaler = StandardScaler()

            X_scaled_train = scaler.fit_transform(
                X_selected_train
            )

            X_scaled_validation = scaler.transform(
                X_selected_validation
            )

            for alpha in RIDGE_ALPHA_VALUES:
                ridge = Ridge(
                    alpha=float(alpha)
                )

                ridge.fit(
                    X_scaled_train,
                    y_fold_train,
                )

                training_predictions = ridge.predict(
                    X_scaled_train
                )

                validation_predictions = ridge.predict(
                    X_scaled_validation
                )

                training_mae = mean_absolute_error(
                    y_fold_train,
                    training_predictions,
                )

                validation_mae = mean_absolute_error(
                    y_fold_validation,
                    validation_predictions,
                )

                fold_results.append(
                    {
                        "fold": fold_number,
                        "top_k": int(top_k),
                        "ridge_alpha": float(alpha),
                        "training_mae": float(
                            training_mae
                        ),
                        "validation_mae": float(
                            validation_mae
                        ),
                    }
                )

    fold_results_dataframe = pd.DataFrame(
        fold_results
    )

    summary_results = (
        fold_results_dataframe
        .groupby(
            [
                "top_k",
                "ridge_alpha",
            ],
            as_index=False,
        )
        .agg(
            mean_training_mae=(
                "training_mae",
                "mean",
            ),
            std_training_mae=(
                "training_mae",
                "std",
            ),
            mean_validation_mae=(
                "validation_mae",
                "mean",
            ),
            std_validation_mae=(
                "validation_mae",
                "std",
            ),
        )
        .sort_values(
            by=[
                "mean_validation_mae",
                "std_validation_mae",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    summary_results.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary_results) + 1,
        ),
    )

    return summary_results


def create_predictions_dataframe(
    model_ids_test: pd.Series,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Create the test prediction results table."""

    actual_values = y_test.to_numpy()

    return pd.DataFrame(
        {
            "ModelID": (
                model_ids_test.to_numpy()
            ),
            "actual_response": actual_values,
            "predicted_response": predictions,
            "residual": (
                actual_values
                - predictions
            ),
            "absolute_error": np.abs(
                actual_values
                - predictions
            ),
        }
    )


def create_feature_ranking_dataframe(
    model: RandomForestRidgeModel,
) -> pd.DataFrame:
    """
    Create a table containing every gene's Random Forest
    importance and its final rank.
    """

    ranking = pd.DataFrame(
        {
            "feature": (
                model.feature_names_in_
            ),
            "random_forest_importance": (
                model.feature_importances_
            ),
        }
    )

    ranking = ranking.sort_values(
        by="random_forest_importance",
        ascending=False,
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    ranking["selected"] = (
        ranking["rank"]
        <= model.top_k
    )

    ridge_coefficients = pd.DataFrame(
        {
            "feature": (
                model.selected_feature_names_
            ),
            "ridge_coefficient": (
                model.ridge_pipeline_
                .named_steps["ridge"]
                .coef_
            ),
        }
    )

    ranking = ranking.merge(
        ridge_coefficients,
        on="feature",
        how="left",
        validate="one_to_one",
    )

    ranking[
        "ridge_coefficient"
    ] = (
        ranking["ridge_coefficient"]
        .fillna(0.0)
    )

    ranking[
        "absolute_ridge_coefficient"
    ] = (
        ranking["ridge_coefficient"]
        .abs()
    )

    return ranking


def plot_feature_selection(
    cross_validation_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the best cross-validation result for each Top K value."""

    best_per_k = (
        cross_validation_results
        .sort_values("mean_validation_mae")
        .groupby("top_k", as_index=False)
        .first()
        .sort_values("top_k")
    )
    best = best_per_k.loc[
        best_per_k["mean_validation_mae"].idxmin()
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        best_per_k["top_k"],
        best_per_k["mean_validation_mae"],
        yerr=best_per_k["std_validation_mae"],
        marker="o",
        capsize=4,
        linewidth=2,
        label="Cross-validation MAE",
    )
    plt.scatter(
        best["top_k"],
        best["mean_validation_mae"],
        s=120,
        color="red",
        zorder=5,
        label=f"Selected K = {int(best['top_k'])}",
    )
    plt.xscale("log")
    plt.xlabel("Top K selected genes")
    plt.ylabel("Cross-validation MAE")
    plt.title("Random Forest feature selection performance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def convert_to_json_serializable(
    value: Any,
) -> Any:
    """Convert NumPy values to JSON-safe Python values."""

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

    serializable_data = (
        convert_to_json_serializable(
            data
        )
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
    model: RandomForestRidgeModel,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    feature_ranking: pd.DataFrame,
    cross_validation_results: pd.DataFrame,
    paths: ModelPaths,
) -> None:
    """Save all model artifacts and evaluation results."""

    paths.model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.predictions_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.feature_ranking_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        paths.model_output,
    )

    predictions.to_csv(
        paths.predictions_output,
        index=False,
    )

    feature_ranking.to_csv(
        paths.feature_ranking_output,
        index=False,
    )

    selected_genes = feature_ranking[
        feature_ranking["selected"]
    ].copy()

    selected_genes.to_csv(
        paths.selected_genes_output,
        index=False,
    )

    cross_validation_results.to_csv(
        paths.cross_validation_results_output,
        index=False,
    )

    save_json(
        data=metrics,
        output_path=paths.metrics_output,
    )


def run(treatment_id: str) -> None:
    """Train and evaluate Random Forest → Ridge for one treatment."""

    paths = build_paths(treatment_id)
    data = load_processed_data(paths)

    validate_modeling_data(
        data
    )

    train_ids, test_ids = (
        load_split_ids(paths)
    )

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

    print(
        f"Treatment: {treatment_id}"
    )

    print(
        f"Total samples: {len(data)}"
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    print(
        f"Gene features: {X_train.shape[1]}"
    )

    plot_target_distribution(
        y=data["drug_response"],
        output_path=paths.target_plot_output,
    )

    baseline_predictions = (
        create_mean_baseline(
            y_train=y_train,
            test_sample_count=len(y_test),
        )
    )

    baseline_metrics = (
        calculate_regression_metrics(
            y_true=y_test,
            y_pred=baseline_predictions,
        )
    )

    cross_validation_results = (
        perform_cross_validation_search(
            X_train=X_train,
            y_train=y_train,
        )
    )

    best_result = (
        cross_validation_results.iloc[0]
    )

    best_top_k = int(
        best_result["top_k"]
    )

    best_ridge_alpha = float(
        best_result["ridge_alpha"]
    )

    cross_validation_mae = float(
        best_result[
            "mean_validation_mae"
        ]
    )

    cross_validation_mae_std = float(
        best_result[
            "std_validation_mae"
        ]
    )

    print(
        "\nBest cross-validation parameters:"
    )

    print(
        f"  Top K genes: {best_top_k}"
    )

    print(
        f"  Ridge alpha: "
        f"{best_ridge_alpha:.4f}"
    )

    print(
        "  Cross-validation MAE: "
        f"{cross_validation_mae:.4f} ± "
        f"{cross_validation_mae_std:.4f}"
    )

    best_model = RandomForestRidgeModel(
        top_k=best_top_k,
        ridge_alpha=best_ridge_alpha,
    )

    print(
        "\nFitting final Random Forest ranking "
        "on all training samples..."
    )

    best_model.fit(
        X_train,
        y_train,
    )

    predictions = best_model.predict(
        X_test
    )

    model_metrics = (
        calculate_regression_metrics(
            y_true=y_test,
            y_pred=predictions,
        )
    )

    print_metrics(
        model_name="Mean baseline",
        metrics=baseline_metrics,
    )

    print_metrics(
        model_name="Random Forest → Ridge",
        metrics=model_metrics,
    )

    print(
        f"\nSelected gene features: "
        f"{best_top_k} / {X_train.shape[1]}"
    )

    prediction_results = (
        create_predictions_dataframe(
            model_ids_test=model_ids_test,
            y_test=y_test,
            predictions=predictions,
        )
    )

    feature_ranking = (
        create_feature_ranking_dataframe(
            model=best_model
        )
    )

    metrics = {
        "model": "random_forest_ridge",
        "treatment_id": treatment_id,
        "total_sample_count": int(
            len(data)
        ),
        "training_sample_count": int(
            len(X_train)
        ),
        "test_sample_count": int(
            len(X_test)
        ),
        "original_feature_count": int(
            X_train.shape[1]
        ),
        "selected_feature_count": (
            best_top_k
        ),
        "selected_feature_fraction": float(
            best_top_k
            / X_train.shape[1]
        ),
        "split_files": {
            "train": str(
                paths.train_ids.relative_to(
                    PROJECT_ROOT
                )
            ),
            "test": str(
                paths.test_ids.relative_to(
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
        "tested_top_k_values": (
            validate_top_k_values(
                X_train.shape[1]
            )
        ),
        "tested_ridge_alpha_values": (
            RIDGE_ALPHA_VALUES
        ),
        "best_parameters": {
            "top_k": best_top_k,
            "ridge_alpha": (
                best_ridge_alpha
            ),
        },
        "model_parameters": {
            "top_k": best_top_k,
            "ridge_alpha": best_ridge_alpha,
            "random_forest_n_estimators": (
                RANDOM_FOREST_ESTIMATORS
            ),
            "random_forest_max_features": (
                RANDOM_FOREST_MAX_FEATURES
            ),
            "random_forest_min_samples_leaf": (
                RANDOM_FOREST_MIN_SAMPLES_LEAF
            ),
        },
        "cross_validation_mae": (
            cross_validation_mae
        ),
        "cross_validation_mae_std": (
            cross_validation_mae_std
        ),
        "random_forest_parameters": {
            "n_estimators": (
                RANDOM_FOREST_ESTIMATORS
            ),
            "max_features": (
                RANDOM_FOREST_MAX_FEATURES
            ),
            "min_samples_leaf": (
                RANDOM_FOREST_MIN_SAMPLES_LEAF
            ),
        },
        "baseline_metrics": (
            baseline_metrics
        ),
        "test_metrics": model_metrics,
    }

    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=predictions,
        output_path=paths.prediction_plot_output,
    )

    plot_residuals(
        y_true=y_test,
        y_pred=predictions,
        output_path=paths.residual_plot_output,
    )

    plot_feature_selection(
        cross_validation_results=cross_validation_results,
        output_path=paths.feature_selection_plot_output,
    )

    save_outputs(
        model=best_model,
        metrics=metrics,
        predictions=prediction_results,
        feature_ranking=feature_ranking,
        cross_validation_results=(
            cross_validation_results
        ),
        paths=paths,
    )

    print("\nSaved outputs:")

    print(
        f"  Model: {paths.model_output}"
    )

    print(
        f"  Metrics: {paths.metrics_output}"
    )

    print(
        f"  Predictions: "
        f"{paths.predictions_output}"
    )

    print(
        f"  Feature ranking: "
        f"{paths.feature_ranking_output}"
    )

    print(
        f"  Selected genes: "
        f"{paths.selected_genes_output}"
    )

    print(
        f"  CV results: "
        f"{paths.cross_validation_results_output}"
    )

    print(
        f"  Prediction plot: "
        f"{paths.prediction_plot_output}"
    )

    print(
        f"  Residual plot: "
        f"{paths.residual_plot_output}"
    )

    print(
        f"  Feature-selection plot: "
        f"{paths.feature_selection_plot_output}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone model execution."""

    parser = argparse.ArgumentParser(
        description="Train a Random Forest → Ridge drug-response model."
    )
    parser.add_argument(
        "--treatment-id",
        required=True,
        help="Treatment ID to model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.treatment_id)


if __name__ == "__main__":
    main()
