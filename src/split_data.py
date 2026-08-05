import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitPaths:
    """Input and output paths for one treatment's data split."""

    features: Path
    target: Path
    split_directory: Path
    train_ids: Path
    test_ids: Path


def build_paths(treatment_id: str) -> SplitPaths:
    """Build processed-data and split paths for one treatment."""

    processed_directory = (
        PROJECT_ROOT / "data" / "processed" / treatment_id
    )
    split_directory = PROJECT_ROOT / "data" / "splits" / treatment_id

    return SplitPaths(
        features=processed_directory / "features.csv",
        target=processed_directory / "target.csv",
        split_directory=split_directory,
        train_ids=split_directory / "train_model_ids.csv",
        test_ids=split_directory / "test_model_ids.csv",
    )


def load_model_ids(paths: SplitPaths) -> pd.Series:
    """Load and validate ModelIDs shared by the features and target files."""

    features = pd.read_csv(
        paths.features,
        usecols=["ModelID"],
    )

    target = pd.read_csv(
        paths.target,
        usecols=["ModelID"],
    )

    if features["ModelID"].duplicated().any():
        duplicate_ids = features.loc[
            features["ModelID"].duplicated(),
            "ModelID",
        ].tolist()

        raise ValueError(
            "Duplicate ModelIDs found in features: "
            f"{duplicate_ids[:10]}"
        )

    if target["ModelID"].duplicated().any():
        duplicate_ids = target.loc[
            target["ModelID"].duplicated(),
            "ModelID",
        ].tolist()

        raise ValueError(
            "Duplicate ModelIDs found in target: "
            f"{duplicate_ids[:10]}"
        )

    feature_ids = set(features["ModelID"])
    target_ids = set(target["ModelID"])

    missing_from_target = feature_ids - target_ids
    missing_from_features = target_ids - feature_ids

    if missing_from_target:
        raise ValueError(
            f"{len(missing_from_target)} ModelIDs are present in "
            "features but missing from target."
        )

    if missing_from_features:
        raise ValueError(
            f"{len(missing_from_features)} ModelIDs are present in "
            "target but missing from features."
        )

    # Sorting ensures that the starting order is deterministic.
    model_ids = (
        features["ModelID"]
        .sort_values()
        .reset_index(drop=True)
    )

    return model_ids


def create_and_save_split(
    model_ids: pd.Series,
    paths: SplitPaths,
) -> None:
    """Create and save a reproducible train/test split."""

    train_ids, test_ids = train_test_split(
        model_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_ids = train_ids.sort_values().reset_index(drop=True)
    test_ids = test_ids.sort_values().reset_index(drop=True)

    paths.split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ids.to_frame(name="ModelID").to_csv(
        paths.train_ids,
        index=False,
    )

    test_ids.to_frame(name="ModelID").to_csv(
        paths.test_ids,
        index=False,
    )

    if set(train_ids) & set(test_ids):
        raise RuntimeError(
            "A ModelID appears in both the training and test splits."
        )

    if len(train_ids) + len(test_ids) != len(model_ids):
        raise RuntimeError(
            "The saved split does not contain every ModelID."
        )

    print(f"Total samples: {len(model_ids)}")
    print(f"Training samples: {len(train_ids)}")
    print(f"Test samples: {len(test_ids)}")
    print(f"Training IDs saved to: {paths.train_ids}")
    print(f"Test IDs saved to: {paths.test_ids}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create a treatment-specific train/test split."
    )
    parser.add_argument(
        "--treatment-id",
        required=True,
        help="Treatment ID whose processed data should be split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths(args.treatment_id)
    model_ids = load_model_ids(paths)
    create_and_save_split(model_ids, paths)


if __name__ == "__main__":
    main()
