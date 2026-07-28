"""
Main entry point for the Drug Response Prediction pipeline.
"""

from pathlib import Path

from src.preprocess import preprocess

# =============================================================================
# Configuration
# =============================================================================

# Choose either a drug name OR a treatment ID
DRUG_NAME = "trametinib"
TREATMENT_ID = None

OUTPUT_DIR = Path("data/processed")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
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


if __name__ == "__main__":
    main()