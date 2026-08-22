"""Shared pytest fixtures.

Tests build their own small, synthetic dataframe instead of loading the real
data/raw/california_housing.csv. That keeps the test suite fast, makes each
test's expectations easy to reason about (you can eyeball the 6-row fixture
below), and means tests keep passing even if the real dataset is
regenerated or replaced.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_housing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MedInc": [8.3, 7.2, 5.6, 3.8, 4.1, 2.0],
            "HouseAge": [41, 21, 52, 34, 15, 30],
            "AveRooms": [6.9, 6.2, 5.8, 5.1, 6.0, 4.5],
            "AveBedrms": [1.0, 1.1, 1.0, 1.0, 1.0, 1.1],
            "Population": [322, 2401, 496, 558, 620, 300],
            "AveOccup": [2.5, 2.1, 2.8, 2.3, 2.6, 3.0],
            "Latitude": [37.9, 37.8, 37.7, 34.1, 34.0, 33.9],
            "Longitude": [-122.2, -122.1, -122.3, -118.3, -118.2, -118.1],
            "MedHouseVal": [4.5, 3.9, 3.4, 2.1, 2.4, 1.2],
        }
    )
