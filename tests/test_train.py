import os
import sys

# Add the parent directory of the training directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import pytest

from train.train_flow import data_prep, data_split


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3],
            "carat": [0.23, 0.21, 0.23, 0.29],
            "cut": ["Ideal", "Premium", "Good", "Premium"],
            "color": ["E", "E", "E", "I"],
            "clarity": ["SI2", "SI1", "VS1", "VS2"],
            "depth": [61.5, 59.8, 56.9, 62.4],
            "table": [55, 61, 65, 58],
            "price": [326, 326, 327, 334],
            "x": [3.95, 3, 4.05, 4.2],
            "y": [3.89, 3.84, 4.07, 4.23],
            "z": [2.43, 2.31, 2.31, 2.63],
        }
    )


def test_data_prep(test_data: pd.DataFrame):
    expected_columns = [
        # "Unnamed: 0"
        "carat",
        "cut",
        "color",
        "clarity",
        "depth",
        "table",
        "price",
        "x",
        "y",
        "z",
    ]
    expected_sample_size = min(1000, len(test_data))
    prepared_data = data_prep.fn(test_data)

    assert list(prepared_data.columns) == expected_columns
    assert len(prepared_data) == expected_sample_size


def test_data_split(test_data: pd.DataFrame):
    prepared_data = data_prep.fn(test_data)
    X_train, x_test, y_train, y_test = data_split.fn(prepared_data)

    assert len(X_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_test) < len(test_data)
