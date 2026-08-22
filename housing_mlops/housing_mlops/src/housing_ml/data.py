"""Loading, validating, and splitting the housing dataset.

Kept deliberately separate from feature selection (features.py) and model
training (train.py): this module only knows about the *raw* data on disk. It
has no opinion about which columns end up in the model -- that decision
lives in config/features.yaml and is applied later, in features.py. This
separation is what lets scripts/select_features.py and train.py both start
from the same raw dataframe without duplicating the loading/validation logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Columns we expect in the raw CSV, independent of which ones ultimately get
# used as model inputs. If the upstream data source changes shape (a column
# renamed or dropped upstream), we want a clear error here rather than a
# confusing failure three steps later inside a sklearn Pipeline.
EXPECTED_COLUMNS = {
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
}


def load_raw(raw_path: Path) -> pd.DataFrame:
    """Read the raw CSV and validate it has the columns we expect."""
    df = pd.read_csv(raw_path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a clear error if the dataframe doesn't look like what the rest
    of the pipeline assumes it looks like.

    This is intentionally a light schema check (column presence + non-empty),
    not a full statistical validation library like pandera/great_expectations
    -- proportionate to a single well-understood dataset. Swap in a heavier
    tool if this ever ingests data from an external, less-trusted source.
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw data is missing expected column(s): {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}"
        )
    if df.empty:
        raise ValueError("Raw data file loaded but contains zero rows.")
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"Raw data contains null values in column(s): {bad_cols}. "
            "This pipeline does not implement imputation -- fix the source data "
            "or add an imputation step to features.py before proceeding."
        )


def split_features_target(
    df: pd.DataFrame, feature_columns: list[str], target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice a dataframe into (X, y) using an explicit feature list rather
    than "everything except the target" -- this is the hook that lets
    config/features.yaml drop low-value columns without touching this code."""
    X = df[feature_columns]
    y = df[target]
    return X, y


def train_test_split_df(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
):
    """Thin wrapper around sklearn's train_test_split, kept here so every
    caller (training script, feature-selection script, tests) draws the
    split the exact same way with the exact same random_state."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
