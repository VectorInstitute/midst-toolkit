import numpy as np
import pandas as pd
import pytest

from midst_toolkit.attacks.ept.feature_extraction import TaskType, preprocess_train_predict
from midst_toolkit.attacks.ept.feature_extraction import main as run_feature_extraction_main


@pytest.fixture
def sample_column_types() -> dict:
    """Provides a sample column_types dictionary."""
    return {
        "numerical": ["num_col_1", "num_col_2"],
        "categorical": ["cat_col_1", "cat_col_2"],
    }


@pytest.fixture
def sample_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provides sample synthetic and challenge dataframes."""
    synthetic_data = pd.DataFrame(
        {
            "num_col_1": [1.0, 2.5, 3.0, 4.1, 5.8],
            "num_col_2": [10, 20, 30, 40, 50],
            "cat_col_1": ["A", "B", "A", "C", "B"],
            "cat_col_2": ["X", "X", "Y", "Y", "X"],
        }
    )

    challenge_data = pd.DataFrame(
        {
            "num_col_1": [1.2, 2.3, 3.1, 4.0, 5.5],
            "num_col_2": [11, 22, 33, 44, 55],
            "cat_col_1": ["A", "B", "B", "C", "A"],
            "cat_col_2": ["X", "Y", "Y", "X", "Y"],
        }
    )
    return synthetic_data, challenge_data


def test_preprocess_train_predict_classification(sample_dataframes, sample_column_types):
    # Tests the preprocess_train_predict function for a classification task.

    train_df, test_df = sample_dataframes
    target_col = "cat_col_1"

    predictions, y_test, task_type = preprocess_train_predict(
        train_points=train_df,
        test_points=test_df,
        target_col=target_col,
        column_types=sample_column_types,
        random_seed=42,
    )

    assert task_type == TaskType.CLASSIFICATION
    assert len(predictions) == len(test_df)
    assert predictions.dtype == "object"  # RandomForestClassifier predicts original class
    pd.testing.assert_series_equal(y_test, test_df[target_col], check_dtype=False, check_names=False)


def test_preprocess_train_predict_regression(sample_dataframes, sample_column_types):
    # Tests the preprocess_train_predict function for a regression task.

    train_df, test_df = sample_dataframes
    target_col = "num_col_1"

    predictions, y_test, task_type = preprocess_train_predict(
        train_points=train_df,
        test_points=test_df,
        target_col=target_col,
        column_types=sample_column_types,
        random_seed=42,
    )

    assert task_type == TaskType.REGRESSION
    assert len(predictions) == len(test_df)
    assert np.issubdtype(predictions.dtype, np.number)  # Should be numeric
    pd.testing.assert_series_equal(y_test, test_df[target_col], check_dtype=False, check_names=False)


def test_preprocess_train_predict_assertions(sample_dataframes, sample_column_types):
    # Tests that the assertions within preprocess_train_predict fire correctly.

    train_df, test_df = sample_dataframes

    # Test mismatching columns
    test_df_mismatch = test_df.drop(columns=["num_col_1"])
    with pytest.raises(AssertionError, match="Columns in df_train and df_test do not match"):
        preprocess_train_predict(train_df, test_df_mismatch, "cat_col_1", sample_column_types)

    # Test target_col not in column_types
    with pytest.raises(AssertionError, match="must appear exactly once"):
        preprocess_train_predict(train_df, test_df, "missing_col", sample_column_types)

    # Test column_types not matching dataframe columns
    bad_column_types = {"numerical": ["num_col_1"], "categorical": []}
    with pytest.raises(AssertionError, match="must match the columns in the combined dataframe"):
        preprocess_train_predict(train_df, test_df, "num_col_1", bad_column_types)


def test_main_feature_extraction(sample_dataframes, sample_column_types):
    # Tests the main orchestrator function for feature extraction.

    synthetic_data, challenge_data = sample_dataframes

    df_results = run_feature_extraction_main(
        synthetic_data=synthetic_data, challenge_data=challenge_data, column_types=sample_column_types, random_seed=42
    )

    assert isinstance(df_results, pd.DataFrame)
    assert len(df_results) == len(challenge_data)

    # Check for expected columns
    expected_columns = [
        # Numerical 1
        "num_col_1",
        "num_col_1_error",
        "num_col_1_error_ratio",
        "num_col_1_prediction",
        # Numerical 2
        "num_col_2",
        "num_col_2_error",
        "num_col_2_error_ratio",
        "num_col_2_prediction",
        # Categorical 1
        "cat_col_1",
        "cat_col_1_accuracy",
        "cat_col_1_prediction",
        # Categorical 2
        "cat_col_2",
        "cat_col_2_accuracy",
        "cat_col_2_prediction",
    ]

    assert sorted(df_results.columns) == sorted(expected_columns)

    # Check that accuracy is 0 or 1
    assert df_results["cat_col_1_accuracy"].isin([0, 1]).all()
    # Check that error is non-negative
    assert (df_results["num_col_1_error"] >= 0).all()
