"""Create a model-ready drug-response dataset.

The script combines:

1. DepMap gene expression
2. PRISM cell-line metadata
3. PRISM drug-response measurements
4. PRISM treatment metadata

Example
-------
Select a drug by name:

    python src/preprocess.py --drug trametinib

Select a treatment directly by its response-matrix column identifier:

    python src/preprocess.py --treatment-id BRD-K12345678
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_loader import DepMapLoader, PrismLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LOGGER = logging.getLogger(__name__)

DRUG_NAME = "trametinib"
TREATMENT_ID = None

OUTPUT_DIR = Path("data/processed")


# These columns describe the sequencing record rather than gene expression.
EXPRESSION_METADATA_COLUMNS = {
    "SequencingID",
    "ModelConditionID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
}


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def normalize_text(value: object) -> str:
    """Normalize text for case-insensitive matching."""
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value).strip().lower())


def find_first_existing_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    """Return the first candidate that exists, ignoring case."""
    column_lookup = {column.lower(): column for column in columns}

    for candidate in candidates:
        match = column_lookup.get(candidate.lower())
        if match is not None:
            return match

    return None


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Convert common Boolean representations to a Boolean mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def prepare_expression(expression: pd.DataFrame) -> pd.DataFrame:
    """Clean the DepMap expression matrix.

    The output contains exactly one row per ModelID followed by numeric
    gene-expression columns.
    """
    expression = expression.copy()

    # Remove CSV index columns accidentally saved in the raw dataset.
    unnamed_columns = [
        column
        for column in expression.columns
        if str(column).startswith("Unnamed:")
    ]

    if unnamed_columns:
        expression = expression.drop(columns=unnamed_columns)

    if "ModelID" not in expression.columns:
        raise ValueError("Expression dataset does not contain ModelID.")

    initial_rows = len(expression)

    # Prefer the canonical/default expression record for each model.
    if "IsDefaultEntryForModel" in expression.columns:
        default_mask = parse_boolean_series(
            expression["IsDefaultEntryForModel"]
        )

        if default_mask.any():
            expression = expression.loc[default_mask].copy()
            LOGGER.info(
                "Filtered expression data to default model entries: "
                "%s -> %s rows",
                f"{initial_rows:,}",
                f"{len(expression):,}",
            )
        else:
            LOGGER.warning(
                "IsDefaultEntryForModel exists, but no true values were found. "
                "Duplicate ModelIDs will be resolved by keeping the first row."
            )

    columns_to_drop = [
        column
        for column in EXPRESSION_METADATA_COLUMNS
        if column in expression.columns
    ]

    expression = expression.drop(columns=columns_to_drop)

    expression["ModelID"] = (
        expression["ModelID"]
        .astype("string")
        .str.strip()
    )

    expression = expression.dropna(subset=["ModelID"])
    expression = expression.loc[expression["ModelID"] != ""]

    duplicate_count = expression["ModelID"].duplicated().sum()

    if duplicate_count:
        LOGGER.warning(
            "Found %s duplicate ModelID rows. Keeping the first occurrence.",
            f"{duplicate_count:,}",
        )
        expression = expression.drop_duplicates(
            subset=["ModelID"],
            keep="first",
        )

    gene_columns = [
        column
        for column in expression.columns
        if column != "ModelID"
    ]

    if not gene_columns:
        raise ValueError("No gene-expression columns remain after cleaning.")

    # Convert expression features to numeric. Invalid values become NaN and
    # are handled later.
    expression[gene_columns] = expression[gene_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Remove genes with no usable values across any model.
    all_missing_genes = expression[gene_columns].columns[
        expression[gene_columns].isna().all()
    ].tolist()

    if all_missing_genes:
        LOGGER.warning(
            "Dropping %s genes containing only missing values.",
            f"{len(all_missing_genes):,}",
        )
        expression = expression.drop(columns=all_missing_genes)

    LOGGER.info(
        "Prepared expression matrix: %s models and %s genes",
        f"{len(expression):,}",
        f"{expression.shape[1] - 1:,}",
    )

    return expression


def prepare_cell_lines(cell_lines: pd.DataFrame) -> pd.DataFrame:
    """Clean PRISM cell-line metadata and standardize identifiers."""
    cell_lines = cell_lines.copy()

    required_columns = {"row_name", "depmap_id"}
    missing_columns = required_columns.difference(cell_lines.columns)

    if missing_columns:
        raise ValueError(
            "PRISM cell-line metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    optional_columns = [
        "ccle_name",
        "primary_tissue",
        "secondary_tissue",
        "tertiary_tissue",
    ]

    columns_to_keep = [
        "row_name",
        "depmap_id",
        *[
            column
            for column in optional_columns
            if column in cell_lines.columns
        ],
    ]

    cell_lines = cell_lines.loc[:, columns_to_keep]

    cell_lines = cell_lines.rename(
        columns={
            "row_name": "prism_row_name",
            "depmap_id": "ModelID",
        }
    )

    cell_lines["prism_row_name"] = (
        cell_lines["prism_row_name"]
        .astype("string")
        .str.strip()
    )

    cell_lines["ModelID"] = (
        cell_lines["ModelID"]
        .astype("string")
        .str.strip()
    )

    cell_lines = cell_lines.dropna(
        subset=["prism_row_name", "ModelID"]
    )

    cell_lines = cell_lines.loc[
        (cell_lines["prism_row_name"] != "")
        & (cell_lines["ModelID"] != "")
    ]

    duplicate_count = cell_lines["prism_row_name"].duplicated().sum()

    if duplicate_count:
        LOGGER.warning(
            "Found %s duplicated PRISM row identifiers. "
            "Keeping the first mapping.",
            f"{duplicate_count:,}",
        )
        cell_lines = cell_lines.drop_duplicates(
            subset=["prism_row_name"],
            keep="first",
        )

    LOGGER.info(
        "Prepared PRISM metadata for %s cell lines",
        f"{len(cell_lines):,}",
    )

    return cell_lines


def detect_response_row_column(
    response: pd.DataFrame,
    cell_lines: pd.DataFrame,
) -> str:
    """Detect which response column contains PRISM cell-line row names."""
    preferred_candidates = [
        "row_name",
        "cell_line",
        "cell_line_name",
        "prism_row_name",
    ]

    preferred_match = find_first_existing_column(
        response.columns,
        preferred_candidates,
    )

    if preferred_match is not None:
        return preferred_match

    cell_line_ids = set(
        cell_lines["prism_row_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    best_column: str | None = None
    best_overlap = 0

    # The identifying column is normally near the beginning of the matrix.
    for column in response.columns[:10]:
        values = set(
            response[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        overlap = len(values.intersection(cell_line_ids))

        if overlap > best_overlap:
            best_column = column
            best_overlap = overlap

    if best_column is None or best_overlap == 0:
        raise ValueError(
            "Could not identify the cell-line identifier column in the "
            "PRISM response matrix."
        )

    LOGGER.info(
        "Detected PRISM response row identifier column '%s' "
        "with %s matching cell lines",
        best_column,
        f"{best_overlap:,}",
    )

    return best_column


def detect_treatment_identifier_column(
    treatments: pd.DataFrame,
    response_treatment_ids: list[str],
) -> str:
    """Find the treatment metadata column matching response matrix columns."""
    response_ids = {
        str(value).strip()
        for value in response_treatment_ids
    }

    best_column: str | None = None
    best_overlap = 0

    for column in treatments.columns:
        treatment_values = set(
            treatments[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        overlap = len(treatment_values.intersection(response_ids))

        if overlap > best_overlap:
            best_column = column
            best_overlap = overlap

    if best_column is None or best_overlap == 0:
        raise ValueError(
            "Could not find a treatment metadata column whose values match "
            "the PRISM response-matrix column identifiers."
        )

    LOGGER.info(
        "Matched response columns to treatment metadata using '%s' "
        "(%s matching identifiers)",
        best_column,
        f"{best_overlap:,}",
    )

    return best_column


def find_matching_treatments(
    treatments: pd.DataFrame,
    treatment_id_column: str,
    drug_query: str,
) -> pd.DataFrame:
    """Find treatment records containing the requested drug name.

    The query is searched across all treatment metadata columns so that the
    script can tolerate different PRISM metadata schemas.
    """
    normalized_query = normalize_text(drug_query)

    if not normalized_query:
        raise ValueError("Drug query cannot be empty.")

    search_mask = pd.Series(
        False,
        index=treatments.index,
        dtype=bool,
    )

    for column in treatments.columns:
        normalized_column = (
            treatments[column]
            .astype("string")
            .fillna("")
            .map(normalize_text)
        )

        search_mask = search_mask | normalized_column.str.contains(
            re.escape(normalized_query),
            regex=True,
            na=False,
        )

    matches = treatments.loc[search_mask].copy()

    if matches.empty:
        raise ValueError(
            f"No PRISM treatments matched drug query: {drug_query!r}"
        )

    matches = matches.dropna(subset=[treatment_id_column])
    matches[treatment_id_column] = (
        matches[treatment_id_column]
        .astype(str)
        .str.strip()
    )

    return matches.drop_duplicates(subset=[treatment_id_column])


def select_treatment_id(
    treatments: pd.DataFrame,
    treatment_id_column: str,
    response_treatment_ids: list[str],
    drug_query: str | None,
    requested_treatment_id: str | None,
) -> tuple[str, pd.Series]:
    """Select exactly one PRISM treatment."""
    available_response_ids = {
        str(value).strip()
        for value in response_treatment_ids
    }

    if requested_treatment_id is not None:
        selected_id = requested_treatment_id.strip()

        if selected_id not in available_response_ids:
            raise ValueError(
                f"Treatment ID {selected_id!r} does not exist in the "
                "PRISM response matrix."
            )

        matching_metadata = treatments.loc[
            treatments[treatment_id_column]
            .astype(str)
            .str.strip()
            .eq(selected_id)
        ]

        if matching_metadata.empty:
            LOGGER.warning(
                "Treatment %s exists in the response matrix but has no "
                "matching treatment metadata.",
                selected_id,
            )
            metadata = pd.Series(
                {treatment_id_column: selected_id},
                dtype="object",
            )
        else:
            metadata = matching_metadata.iloc[0]

        return selected_id, metadata

    if drug_query is None:
        raise ValueError(
            "Provide either --drug or --treatment-id."
        )

    matches = find_matching_treatments(
        treatments=treatments,
        treatment_id_column=treatment_id_column,
        drug_query=drug_query,
    )

    matches = matches.loc[
        matches[treatment_id_column].isin(available_response_ids)
    ]

    if matches.empty:
        raise ValueError(
            f"Treatments matched {drug_query!r}, but none have response "
            "measurements in the loaded PRISM matrix."
        )

    if len(matches) > 1:
        display_columns = [
            treatment_id_column,
            *[
                column
                for column in matches.columns
                if column != treatment_id_column
            ][:5],
        ]

        candidate_text = matches.loc[
            :,
            display_columns,
        ].head(20).to_string(index=False)

        raise ValueError(
            f"Drug query {drug_query!r} matched {len(matches)} treatments.\n"
            "Use --treatment-id to select one of these candidates:\n\n"
            f"{candidate_text}"
        )

    selected_row = matches.iloc[0]
    selected_id = str(selected_row[treatment_id_column]).strip()

    return selected_id, selected_row


def extract_single_drug_response(
    response: pd.DataFrame,
    response_row_column: str,
    treatment_id: str,
) -> pd.DataFrame:
    """Extract response measurements for one treatment."""
    if treatment_id not in response.columns:
        raise ValueError(
            f"Treatment ID {treatment_id!r} is not a response column."
        )

    selected_response = response.loc[
        :,
        [response_row_column, treatment_id],
    ].copy()

    selected_response = selected_response.rename(
        columns={
            response_row_column: "prism_row_name",
            treatment_id: "drug_response",
        }
    )

    selected_response["prism_row_name"] = (
        selected_response["prism_row_name"]
        .astype("string")
        .str.strip()
    )

    selected_response["drug_response"] = pd.to_numeric(
        selected_response["drug_response"],
        errors="coerce",
    )

    initial_rows = len(selected_response)

    selected_response = selected_response.dropna(
        subset=["prism_row_name", "drug_response"]
    )

    selected_response = selected_response.loc[
        selected_response["prism_row_name"] != ""
    ]

    LOGGER.info(
        "Removed %s rows with missing cell-line identifiers or responses",
        f"{initial_rows - len(selected_response):,}",
    )

    duplicate_count = selected_response[
        "prism_row_name"
    ].duplicated().sum()

    if duplicate_count:
        LOGGER.warning(
            "Found %s duplicate cell-line responses. "
            "Averaging duplicate measurements.",
            f"{duplicate_count:,}",
        )

        selected_response = (
            selected_response
            .groupby("prism_row_name", as_index=False)["drug_response"]
            .mean()
        )

    return selected_response


def merge_modeling_data(
    expression: pd.DataFrame,
    cell_lines: pd.DataFrame,
    drug_response: pd.DataFrame,
) -> pd.DataFrame:
    """Merge expression and response data through PRISM cell-line metadata."""
    response_with_ids = drug_response.merge(
        cell_lines,
        on="prism_row_name",
        how="inner",
        validate="one_to_one",
    )

    LOGGER.info(
        "PRISM responses with valid DepMap identifiers: %s",
        f"{len(response_with_ids):,}",
    )

    merged = response_with_ids.merge(
        expression,
        on="ModelID",
        how="inner",
        validate="one_to_one",
    )

    LOGGER.info(
        "Cell lines with both response and expression data: %s",
        f"{len(merged):,}",
    )

    if merged.empty:
        raise ValueError(
            "The merge produced zero rows. Check ModelID and depmap_id "
            "formatting and confirm the datasets come from compatible releases."
        )

    if merged["ModelID"].duplicated().any():
        raise ValueError(
            "Merged dataset contains duplicate ModelID values."
        )

    return merged


def filter_missing_expression(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove missing expression features and incomplete model rows."""
    metadata_columns = {
        "prism_row_name",
        "ModelID",
        "ccle_name",
        "primary_tissue",
        "secondary_tissue",
        "tertiary_tissue",
        "drug_response",
        "treatment_id",
    }

    gene_columns = [
        column
        for column in dataframe.columns
        if column not in metadata_columns
    ]

    if not gene_columns:
        raise ValueError("Merged dataset contains no gene features.")

    all_missing_genes = [
        column
        for column in gene_columns
        if dataframe[column].isna().all()
    ]

    if all_missing_genes:
        dataframe = dataframe.drop(columns=all_missing_genes)
        gene_columns = [
            column
            for column in gene_columns
            if column not in all_missing_genes
        ]

        LOGGER.warning(
            "Dropped %s genes with no expression values.",
            f"{len(all_missing_genes):,}",
        )

    missing_row_mask = dataframe[gene_columns].isna().any(axis=1)
    missing_row_count = int(missing_row_mask.sum())

    if missing_row_count:
        LOGGER.warning(
            "Dropping %s cell lines containing at least one missing "
            "gene-expression value.",
            f"{missing_row_count:,}",
        )
        dataframe = dataframe.loc[~missing_row_mask].copy()

    if dataframe.empty:
        raise ValueError(
            "No rows remain after filtering missing expression values."
        )

    return dataframe, gene_columns


def validate_modeling_data(
    dataframe: pd.DataFrame,
    gene_columns: list[str],
) -> None:
    """Verify that each model has complete expression and response data."""
    required_columns = {"ModelID", "drug_response"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Final dataset is missing columns: {sorted(missing_columns)}"
        )

    if dataframe["ModelID"].isna().any():
        raise ValueError("Final dataset contains missing ModelID values.")

    if dataframe["drug_response"].isna().any():
        raise ValueError(
            "Final dataset contains missing drug-response values."
        )

    if dataframe["ModelID"].duplicated().any():
        raise ValueError(
            "Final dataset contains more than one row per ModelID."
        )

    if dataframe[gene_columns].isna().any().any():
        raise ValueError(
            "Final dataset contains missing gene-expression values."
        )

    if not pd.api.types.is_numeric_dtype(dataframe["drug_response"]):
        raise ValueError("drug_response is not numeric.")

    LOGGER.info(
        "Validation passed: every ModelID has expression and response data."
    )


def save_outputs(
    dataframe: pd.DataFrame,
    gene_columns: list[str],
    treatment_id: str,
    treatment_metadata: pd.Series,
    output_dir: Path,
) -> None:
    """Save processed model data and preprocessing metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_treatment_id = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        treatment_id,
    )

    modeling_data_path = (
        output_dir / f"modeling_data_{safe_treatment_id}.csv"
    )
    features_path = (
        output_dir / f"features_{safe_treatment_id}.csv"
    )
    target_path = (
        output_dir / f"target_{safe_treatment_id}.csv"
    )
    metadata_path = (
        output_dir / f"metadata_{safe_treatment_id}.json"
    )

    dataframe.to_csv(modeling_data_path, index=False)

    dataframe.loc[
        :,
        ["ModelID", *gene_columns],
    ].to_csv(features_path, index=False)

    dataframe.loc[
        :,
        ["ModelID", "drug_response"],
    ].to_csv(target_path, index=False)

    treatment_metadata_dict = {
        str(key): None if pd.isna(value) else str(value)
        for key, value in treatment_metadata.to_dict().items()
    }

    processing_metadata = {
        "selected_treatment_id": treatment_id,
        "number_of_models": len(dataframe),
        "number_of_gene_features": len(gene_columns),
        "response_column": "drug_response",
        "treatment_metadata": treatment_metadata_dict,
        "outputs": {
            "modeling_data": str(modeling_data_path),
            "features": str(features_path),
            "target": str(target_path),
        },
    }

    metadata_path.write_text(
        json.dumps(processing_metadata, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Saved modeling data to %s", modeling_data_path)
    LOGGER.info("Saved feature matrix to %s", features_path)
    LOGGER.info("Saved target data to %s", target_path)
    LOGGER.info("Saved processing metadata to %s", metadata_path)


def preprocess(
    drug_query: str | None,
    requested_treatment_id: str | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the full preprocessing workflow."""
    depmap_loader = DepMapLoader()
    prism_loader = PrismLoader()

    LOGGER.info("Loading raw datasets")

    raw_expression = depmap_loader.load_expression()
    raw_cell_lines = prism_loader.load_cell_lines()
    raw_response = prism_loader.load_drug_response()
    raw_treatments = prism_loader.load_treatments()

    LOGGER.info(
        "Raw expression shape: %s",
        raw_expression.shape,
    )
    LOGGER.info(
        "Raw PRISM response shape: %s",
        raw_response.shape,
    )

    expression = prepare_expression(raw_expression)
    cell_lines = prepare_cell_lines(raw_cell_lines)

    response_row_column = detect_response_row_column(
        response=raw_response,
        cell_lines=cell_lines,
    )

    response_treatment_ids = [
        str(column).strip()
        for column in raw_response.columns
        if column != response_row_column
    ]

    treatment_id_column = detect_treatment_identifier_column(
        treatments=raw_treatments,
        response_treatment_ids=response_treatment_ids,
    )

    selected_treatment_id, treatment_metadata = select_treatment_id(
        treatments=raw_treatments,
        treatment_id_column=treatment_id_column,
        response_treatment_ids=response_treatment_ids,
        drug_query=drug_query,
        requested_treatment_id=requested_treatment_id,
    )

    LOGGER.info(
        "Selected treatment: %s",
        selected_treatment_id,
    )

    drug_response = extract_single_drug_response(
        response=raw_response,
        response_row_column=response_row_column,
        treatment_id=selected_treatment_id,
    )

    modeling_data = merge_modeling_data(
        expression=expression,
        cell_lines=cell_lines,
        drug_response=drug_response,
    )

    modeling_data["treatment_id"] = selected_treatment_id

    modeling_data, gene_columns = filter_missing_expression(
        modeling_data
    )

    validate_modeling_data(
        dataframe=modeling_data,
        gene_columns=gene_columns,
    )

    # Put identifiers and metadata before the high-dimensional features.
    leading_columns = [
        column
        for column in [
            "ModelID",
            "prism_row_name",
            "ccle_name",
            "primary_tissue",
            "secondary_tissue",
            "tertiary_tissue",
            "treatment_id",
            "drug_response",
        ]
        if column in modeling_data.columns
    ]

    modeling_data = modeling_data.loc[
        :,
        [*leading_columns, *gene_columns],
    ]

    save_outputs(
        dataframe=modeling_data,
        gene_columns=gene_columns,
        treatment_id=selected_treatment_id,
        treatment_metadata=treatment_metadata,
        output_dir=output_dir,
    )

    return modeling_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge DepMap expression with PRISM response measurements "
            "for one drug."
        )
    )

    selection_group = parser.add_mutually_exclusive_group(
        required=True
    )

    selection_group.add_argument(
        "--drug",
        type=str,
        help=(
            "Drug name or other text to search for in PRISM treatment "
            "metadata, such as 'trametinib'. The query must resolve to "
            "exactly one treatment."
        ),
    )

    selection_group.add_argument(
        "--treatment-id",
        type=str,
        help=(
            "Exact treatment identifier corresponding to a column in the "
            "PRISM response matrix."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for processed outputs. "
            "Default: data/processed"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    modeling_data = preprocess(
        drug_query=args.drug,
        requested_treatment_id=args.treatment_id,
        output_dir=args.output_dir,
    )

    print("\nPreprocessing complete")
    print("----------------------")
    print(f"Models: {len(modeling_data):,}")
    print(f"Columns: {modeling_data.shape[1]:,}")
    print(
        "Response range: "
        f"{modeling_data['drug_response'].min():.4f} to "
        f"{modeling_data['drug_response'].max():.4f}"
    )


if __name__ == "__main__":
    """Run the preprocessing pipeline."""

    print("=" * 60)
    print("Drug Response Prediction")
    print("=" * 60)

    modeling_data = preprocess(
        drug_query=DRUG_NAME,
        requested_treatment_id=TREATMENT_ID,
        output_dir=OUTPUT_DIR,
    )

    print("\nPreprocessing Complete")
    print("-" * 60)
    print(f"Samples: {len(modeling_data):,}")
    print(f"Features: {modeling_data.shape[1] - 2:,}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")