import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Load cross-validation results
results = pd.read_csv(
    "artifacts/metrics/random_forest_ridge_cv_results_BRD-K12343256-001-08-9_2.5_HTS.csv"
)

# Keep only the best alpha for each Top K
best_per_k = (
    results.sort_values("mean_validation_mae")
    .groupby("top_k", as_index=False)
    .first()
)

# Best overall point
best = best_per_k.loc[
    best_per_k["mean_validation_mae"].idxmin()
]

# Create figure
plt.figure(figsize=(8, 5))

# Plot CV MAE with error bars
plt.errorbar(
    best_per_k["top_k"],
    best_per_k["mean_validation_mae"],
    yerr=best_per_k["std_validation_mae"],
    marker="o",
    capsize=4,
    linewidth=2,
    label="Cross-validation MAE",
)

# Highlight the selected K
plt.scatter(
    best["top_k"],
    best["mean_validation_mae"],
    s=120,
    color="red",
    zorder=5,
    label=f"Selected K = {int(best['top_k'])}",
)

# Logarithmic x-axis for readability
plt.xscale("log")

plt.xticks(
    [100, 250, 500, 1000, 2000, 5000, 10000, 15000, 19215],
    ["100", "250", "500", "1k", "2k", "5k", "10k", "15k", "19k"],
)

plt.xlabel("Top K Selected Genes")
plt.ylabel("Cross-Validation MAE")
plt.title("Random Forest Feature Selection Performance")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig(
    "artifacts/figures/random_forest_feature_selection.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()