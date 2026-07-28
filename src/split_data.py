from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TREATMENT_ID = "BRD-K12343256-001-08-9_2.5_HTS"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

SPLITS_DIRECTORY = PROJECT_ROOT / "data" / "splits"

TRAIN_IDS_PATH = SPLITS_DIRECTORY / "train_model_ids.csv"
TEST_IDS_PATH = SPLITS_DIRECTORY / "test_model_ids.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_model_ids() -> pd.Series:
    """Load and validate ModelIDs shared by the features and target files."""

    features = pd.read_csv(
        FEATURES_PATH,
        usecols=["ModelID"],
    )

    target = pd.read_csv(
        TARGET_PATH,
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

    SPLITS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ids.to_frame(name="ModelID").to_csv(
        TRAIN_IDS_PATH,
        index=False,
    )

    test_ids.to_frame(name="ModelID").to_csv(
        TEST_IDS_PATH,
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
    print(f"Training IDs saved to: {TRAIN_IDS_PATH}")
    print(f"Test IDs saved to: {TEST_IDS_PATH}")


def main() -> None:
    model_ids = load_model_ids()
    create_and_save_split(model_ids)


if __name__ == "__main__":
    main()