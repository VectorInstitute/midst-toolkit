import pandas as pd

from midst_toolkit.data_processing.utils import (
    get_categorical_columns,
    infer_categorical_and_numerical_columns,
    is_column_type_numerical,
)


TEST_DATAFRAME = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4, 5],
        "column_b": [1, 1.5, 2.0, 3.0, -1.0],
        "column_c": ["house", "cat", "cat", "car", "dog"],
        "column_d": [1, 1, 3, 2, 2],
        "column_e": [1.0, 3.0, 1.0, 2.0, 1.0],
        "column_f": [
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-06"),
            pd.Timestamp("2018-01-07"),
            pd.Timestamp("2018-01-08"),
            pd.Timestamp("2018-01-09"),
        ],
    }
)


def test_is_column_type_numerical() -> None:
    assert is_column_type_numerical(TEST_DATAFRAME, "column_a")
    assert is_column_type_numerical(TEST_DATAFRAME, "column_b")
    assert not is_column_type_numerical(TEST_DATAFRAME, "column_c")
    assert is_column_type_numerical(TEST_DATAFRAME, "column_d")
    assert is_column_type_numerical(TEST_DATAFRAME, "column_e")
    assert not is_column_type_numerical(TEST_DATAFRAME, "column_f")


def test_get_categorical_columns() -> None:
    # Low threshold
    categorical_columns = get_categorical_columns(TEST_DATAFRAME, 2)
    # Note that this does not include the date time column, as it isn't a categorical, as the detection algorithm
    # functions at the moment.
    assert categorical_columns == ["column_c"]

    # Higher threshold
    categorical_columns = get_categorical_columns(TEST_DATAFRAME, 4)
    # Note that this does not include the date time column, as it isn't a categorical, as the detection algorithm
    # functions at the moment.
    assert sorted(categorical_columns) == ["column_c", "column_d", "column_e"]


def test_infer_categorical_and_numerical_columns() -> None:
    # Low threshold
    categorical_columns, numerical_columns = infer_categorical_and_numerical_columns(
        TEST_DATAFRAME, categorical_threshold=2
    )
    assert categorical_columns == ["column_c"]
    # Note that this includes the date time column, as it isn't a categorical, as the detection algorithm functions
    # at the moment.
    assert sorted(numerical_columns) == ["column_a", "column_b", "column_d", "column_e", "column_f"]

    # Higher threshold
    categorical_columns, numerical_columns = infer_categorical_and_numerical_columns(
        TEST_DATAFRAME, categorical_threshold=4
    )
    assert sorted(categorical_columns) == ["column_c", "column_d", "column_e"]
    assert sorted(numerical_columns) == ["column_a", "column_b", "column_f"]
