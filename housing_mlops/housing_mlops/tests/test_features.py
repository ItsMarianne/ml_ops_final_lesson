import pytest

from housing_ml.config import FeaturesConfig
from housing_ml.features import selected_columns


def test_selected_columns_returns_configured_list(sample_housing_df):
    config = FeaturesConfig(
        target="MedHouseVal",
        selected=["MedInc", "HouseAge"],
        dropped={"AveBedrms": "test reason"},
    )
    assert selected_columns(config, sample_housing_df) == ["MedInc", "HouseAge"]


def test_selected_columns_rejects_column_missing_from_data(sample_housing_df):
    config = FeaturesConfig(
        target="MedHouseVal",
        selected=["MedInc", "NotAColumn"],
        dropped={},
    )
    with pytest.raises(ValueError, match="not present in the data"):
        selected_columns(config, sample_housing_df)
