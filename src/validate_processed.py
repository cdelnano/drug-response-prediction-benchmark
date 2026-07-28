"""Validate the processed drug-response modeling dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")


def validate_processed_data(file_path: Path) -> None:
    """Run integrity checks on a processed modeling dataset."""
    df = pd.read_csv(file_path, low_memory=False)

    print("=" * 60)
    print("Processed Dataset Validation")
    print("=" * 60)

    print(f"File: {file_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")

    required_columns = {
        "ModelID",
        "drug_response",
        "treatment_id",
    }

    missing_required = required_columns.difference(df.columns)

    assert not missing_required, (
        f"Missing required columns: {sorted(missing_required)}"
    )

    assert not df.empty, "The processed dataset is empty."

    assert df["ModelID"].notna().all(), (
        "Some rows have missing ModelID values."
    )

    assert df["drug_response"].notna().all(), (
        "Some rows have missing drug-response values."
    )

    assert df["ModelID"].is_unique, (
        "ModelID is not unique. Some cell lines appear more than once."
    )

    assert df["treatment_id"].nunique() == 1, (
        "The dataset contains more than one treatment."
    )

    metadata_columns = {
        "ModelID",
        "prism_row_name",
        "ccle_name",
        "primary_tissue",
        "secondary_tissue",
        "tertiary_tissue",
        "treatment_id",
        "drug_response",
    }

    gene_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    assert gene_columns, "No gene-expression columns were found."

    gene_data = df[gene_columns]

    non_numeric_genes = [
        column
        for column in gene_columns
        if not pd.api.types.is_numeric_dtype(gene_data[column])
    ]

    assert not non_numeric_genes, (
        "Some gene columns are not numeric: "
        f"{non_numeric_genes[:10]}"
    )

    missing_expression_values = int(
        gene_data.isna().sum().sum()
    )

    assert missing_expression_values == 0, (
        f"Found {missing_expression_values:,} missing expression values."
    )

    assert pd.api.types.is_numeric_dtype(df["drug_response"]), (
        "drug_response is not numeric."
    )

    print("\nIntegrity checks")
    print("----------------")
    print("✓ Required columns exist")
    print("✓ ModelID contains no missing values")
    print("✓ Every ModelID is unique")
    print("✓ Drug response contains no missing values")
    print("✓ Dataset contains exactly one treatment")
    print("✓ Gene-expression columns are numeric")
    print("✓ Gene-expression values contain no missing values")

    print("\nTreatment")
    print("---------")
    print(df["treatment_id"].iloc[0])

    print("\nDrug-response distribution")
    print("--------------------------")
    print(df["drug_response"].describe())

    print("\nExpression summary")
    print("------------------")
    print(f"Gene features: {len(gene_columns):,}")
    print(
        f"Minimum expression value: "
        f"{gene_data.min().min():.4f}"
    )
    print(
        f"Maximum expression value: "
        f"{gene_data.max().max():.4f}"
    )

    constant_genes = (
        gene_data.nunique(dropna=False)
        .loc[lambda values: values <= 1]
        .index
        .tolist()
    )

    print(f"Constant genes: {len(constant_genes):,}")

    print("\nValidation passed.")


def main() -> None:
    """Validate the first modeling dataset in data/processed."""
    matching_files = sorted(
        PROCESSED_DIR.glob("modeling_data_*.csv")
    )

    if not matching_files:
        raise FileNotFoundError(
            "No modeling_data_*.csv files found in data/processed."
        )

    if len(matching_files) > 1:
        print("Found multiple processed datasets:")
        for file_path in matching_files:
            print(f"  - {file_path}")

        print(f"\nValidating most recent file: {matching_files[-1]}")

    validate_processed_data(matching_files[-1])


if __name__ == "__main__":
    main()