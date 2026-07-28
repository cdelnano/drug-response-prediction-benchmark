from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ArrayLike = pd.Series | np.ndarray

def calculate_correlation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float | None, float | None]:
    """Calculate correlations, returning None for constant inputs."""

    true_is_constant = np.all(y_true == y_true[0])
    prediction_is_constant = np.all(y_pred == y_pred[0])

    if true_is_constant or prediction_is_constant:
        return None, None

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConstantInputWarning,
        )

        pearson = float(
            pearsonr(
                y_true,
                y_pred,
            ).statistic
        )

        spearman = float(
            spearmanr(
                y_true,
                y_pred,
            ).statistic
        )

    if not np.isfinite(pearson):
        pearson = None

    if not np.isfinite(spearman):
        spearman = None

    return pearson, spearman

def calculate_regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Calculate regression and correlation metrics."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            "y_true and y_pred must have the same number of values."
        )

    
    if len(y_true_array) == 0:
        raise ValueError(
            "Cannot calculate metrics for empty arrays."
        )

    pearson, spearman = calculate_correlation_metrics(
        y_true=y_true_array,
        y_pred=y_pred_array,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true_array,
                y_pred_array,
            )
        ),
        "rmse": float(
            mean_squared_error(
                y_true_array,
                y_pred_array,
            )
            ** 0.5
        ),
        "r2": float(
            r2_score(
                y_true_array,
                y_pred_array,
            )
        ),
        "pearson": pearson,
        "spearman": spearman,
    }


def print_metrics(
    model_name: str,
    metrics: dict[str, Any],
) -> None:
    """Print metrics using a consistent format."""

    print(f"\n{model_name}")

    for metric_name, metric_value in metrics.items():
        if metric_value is None:
            print(f"  {metric_name}: undefined")
        elif isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")


def plot_target_distribution(
    y: ArrayLike,
    output_path: Path,
) -> None:
    """Save the drug-response target distribution."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=30)
    plt.xlabel("Drug response")
    plt.ylabel("Number of cell lines")
    plt.title("Drug-response distribution")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
    )
    plt.close()


def plot_actual_vs_predicted(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    output_path: Path,
) -> None:
    """Save an actual-versus-predicted plot."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    minimum = min(
        y_true_array.min(),
        y_pred_array.min(),
    )

    maximum = max(
        y_true_array.max(),
        y_pred_array.max(),
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(
        y_true_array,
        y_pred_array,
        alpha=0.7,
    )
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
    )
    plt.xlabel("Actual drug response")
    plt.ylabel("Predicted drug response")
    plt.title("Actual versus predicted response")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
    )
    plt.close()


def plot_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    output_path: Path,
) -> None:
    """Save a residual-versus-predicted plot."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    residuals = y_true_array - y_pred_array

    plt.figure(figsize=(7, 6))
    plt.scatter(
        y_pred_array,
        residuals,
        alpha=0.7,
    )
    plt.axhline(0)
    plt.xlabel("Predicted drug response")
    plt.ylabel("Residual")
    plt.title("Prediction residuals")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
    )
    plt.close()