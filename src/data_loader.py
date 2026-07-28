"""Load raw DepMap and PRISM datasets.

This module is responsible only for reading and validating raw datasets.
It does not clean, merge, filter, or transform the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


class DatasetLoadError(RuntimeError):
    """Raised when a required dataset cannot be loaded or validated."""


def _validate_file(file_path: Path) -> None:
    """Verify that a dataset file exists and is a regular file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Check that the file is in the expected data/raw directory."
        )

    if not file_path.is_file():
        raise DatasetLoadError(f"Expected a file but found: {file_path}")


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Verify that a DataFrame contains required columns."""
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise DatasetLoadError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _read_csv(file_path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV file and provide a more informative error message."""
    _validate_file(file_path)

    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to load CSV file: {file_path}"
        ) from exc


@dataclass(frozen=True)
class DepMapLoader:
    """Load raw datasets from the DepMap release."""

    data_dir: Path = DEFAULT_RAW_DATA_DIR / "DepMap"

    expression_filename: str = (
        "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    model_filename: str = "Model.csv"
    gene_filename: str = "Gene.csv"
    compound_filename: str = "PortalCompounds.csv"

    def load_expression(self) -> pd.DataFrame:
        """Load the protein-coding gene-expression matrix.

        Returns
        -------
        pandas.DataFrame
            Expression data containing ModelID, sequencing metadata, and
            gene-expression features.
        """
        file_path = self.data_dir / self.expression_filename
        dataframe = _read_csv(file_path, low_memory=False)

        _validate_columns(
            dataframe,
            required_columns={"ModelID"},
            dataset_name="DepMap expression dataset",
        )

        return dataframe

    def load_models(self) -> pd.DataFrame:
        """Load DepMap model metadata."""
        file_path = self.data_dir / self.model_filename
        dataframe = _read_csv(file_path, low_memory=False)

        _validate_columns(
            dataframe,
            required_columns={"ModelID"},
            dataset_name="DepMap model metadata",
        )

        return dataframe

    def load_genes(self) -> pd.DataFrame:
        """Load DepMap gene metadata."""
        file_path = self.data_dir / self.gene_filename
        return _read_csv(file_path, low_memory=False)

    def load_compounds(self) -> pd.DataFrame:
        """Load DepMap compound metadata."""
        file_path = self.data_dir / self.compound_filename
        return _read_csv(file_path, low_memory=False)


@dataclass(frozen=True)
class PrismLoader:
    """Load raw PRISM primary-screen datasets."""

    data_dir: Path = DEFAULT_RAW_DATA_DIR / "PRISM"

    response_filename: str = (
        "primary-screen-replicate-collapsed-logfold-change.csv"
    )
    cell_line_filename: str = "primary-screen-cell-line-info.csv"
    treatment_filename: str = (
        "primary-screen-replicate-collapsed-treatment-info.csv"
    )

    def load_drug_response(self) -> pd.DataFrame:
        """Load the replicate-collapsed PRISM log-fold-change matrix."""
        file_path = self.data_dir / self.response_filename
        dataframe = _read_csv(file_path, low_memory=False)

        if dataframe.empty:
            raise DatasetLoadError("PRISM drug-response dataset is empty.")

        return dataframe

    def load_cell_lines(self) -> pd.DataFrame:
        """Load PRISM cell-line metadata and DepMap identifiers."""
        file_path = self.data_dir / self.cell_line_filename
        dataframe = _read_csv(file_path, low_memory=False)

        _validate_columns(
            dataframe,
            required_columns={"row_name", "depmap_id"},
            dataset_name="PRISM cell-line metadata",
        )

        return dataframe

    def load_treatments(self) -> pd.DataFrame:
        """Load PRISM treatment and compound metadata."""
        file_path = self.data_dir / self.treatment_filename
        dataframe = _read_csv(file_path, low_memory=False)

        if dataframe.empty:
            raise DatasetLoadError("PRISM treatment metadata is empty.")

        return dataframe


def print_dataset_summary(
    name: str,
    dataframe: pd.DataFrame,
    preview_columns: int = 10,
) -> None:
    """Print a compact dataset summary for initial validation."""
    visible_columns = list(dataframe.columns[:preview_columns])

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Rows: {dataframe.shape[0]:,}")
    print(f"Columns: {dataframe.shape[1]:,}")
    print(f"First columns: {visible_columns}")


def main() -> None:
    """Load every raw dataset and print its dimensions."""
    depmap_loader = DepMapLoader()
    prism_loader = PrismLoader()

    datasets = {
        "DepMap expression": depmap_loader.load_expression(),
        "DepMap model metadata": depmap_loader.load_models(),
        "DepMap gene metadata": depmap_loader.load_genes(),
        "DepMap compound metadata": depmap_loader.load_compounds(),
        "PRISM drug response": prism_loader.load_drug_response(),
        "PRISM cell-line metadata": prism_loader.load_cell_lines(),
        "PRISM treatment metadata": prism_loader.load_treatments(),
    }

    for dataset_name, dataframe in datasets.items():
        print_dataset_summary(dataset_name, dataframe)


if __name__ == "__main__":
    main()