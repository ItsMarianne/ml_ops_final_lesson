import pandas as pd
import pytest

from housing_ml.data import (
    EXPECTED_COLUMNS,
    split_features_target,
    train_test_split_df,
    validate_schema,
)


def test_validate_schema_accepts_well_formed_data(sample_housing_df):
    # Should not raise.
    validate_schema(sample_housing_df)


def test_validate_schema_rejects_missing_column(sample_housing_df):
    broken = sample_housing_df.drop(columns=["MedInc"])
    with pytest.raises(ValueError, match="missing expected column"):
        validate_schema(broken)


def test_validate_schema_rejects_empty_dataframe():
    empty = pd.DataFrame(columns=list(EXPECTED_COLUMNS))
    with pytest.raises(ValueError, match="zero rows"):
        validate_schema(empty)


def test_validate_schema_rejects_nulls(sample_housing_df):
    broken = sample_housing_df.copy()
    broken.loc[0, "MedInc"] = None
    with pytest.raises(ValueError, match="null values"):
        validate_schema(broken)


def test_split_features_target_uses_explicit_column_list(sample_housing_df):
    X, y = split_features_target(sample_housing_df, ["MedInc", "HouseAge"], "MedHouseVal")
    assert list(X.columns) == ["MedInc", "HouseAge"]
    assert y.name == "MedHouseVal"
    assert len(X) == len(y) == len(sample_housing_df)


def test_train_test_split_df_is_deterministic(sample_housing_df):
    X, y = split_features_target(sample_housing_df, ["MedInc"], "MedHouseVal")
    split_a = train_test_split_df(X, y, test_size=0.5, random_state=4)
    split_b = train_test_split_df(X, y, test_size=0.5, random_state=4)
    pd.testing.assert_frame_equal(split_a[0], split_b[0])
