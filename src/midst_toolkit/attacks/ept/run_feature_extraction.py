"""
Module to run feature extraction for EPT attack steps 2 and 3.
Overall workflow and decisions are taken with from the BGU team's attack implementation at
https://github.com/eyalgerman/MIA-EPT.

"""

from enum import Enum
from logging import INFO
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from midst_toolkit.common.logger import log


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


def preprocess_train_predict(
    train_points: pd.DataFrame,
    test_points: pd.DataFrame,
    target_col: str,
    column_types: dict[str, Any],
    random_seed: int | None = None,
) -> tuple[np.ndarray, pd.Series, TaskType]:
    """
    Preprocess the data, train an attribute prediction model, and generate predictions.

    Args:
        train_points: Data to train the attribute prediction model on. Must include the target column.
        test_points: Data to test the attribute prediction model on. Must include the target column.
        target_col: Name of the target column to predict.
        column_types: Types of columns in the data. Relevant keys are "numerical", "categorical".
        random_seed: Seed for model reproducibility. Defaults to None.

    Returns:
            predictions: Predicted values for the target column on the test data.
            y_test: True values for the target column on the test data.
            task_type: Whether the attribution prediction model was a classification or regression model.
    """
    df_train = train_points.copy()
    df_test = test_points.copy()

    df_train["is_train"] = 1
    df_test["is_train"] = 0

    assert set(df_train.columns) == set(df_test.columns), "Columns in df_train and df_test do not match"

    # Original code combines the dataframes to ensure consistent preprocessing
    combined = pd.concat([df_train, df_test], axis=0)

    numeric_columns = column_types["numerical"]
    categorical_columns = column_types["categorical"]

    import pdb

    pdb.set_trace()

    # Assert that the target column appears exactly once in numeric_columns + categorical_columns
    assert (numeric_columns + categorical_columns).count(target_col) == 1, (
        f"The target column '{target_col}' must appear exactly once in numeric_columns + categorical_columns"
    )

    # Assert that the union of numeric_columns and categorical_columns matches
    # the columns in the combined dataframe, except for 'is_train'
    assert set(numeric_columns + categorical_columns) == set(combined.columns) - {"is_train"}, (
        "The union of numeric_columns and categorical_columns must match the columns in the combined dataframe, except for 'is_train'"
    )

    # Remove target column from feature columns
    numeric_columns = [col for col in numeric_columns if col != target_col]
    categorical_columns = [col for col in categorical_columns if col != target_col]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    task_type = TaskType.CLASSIFICATION if target_col in categorical_columns else TaskType.REGRESSION

    model = (
        RandomForestClassifier(random_state=random_seed)
        if task_type == TaskType.CLASSIFICATION
        else RandomForestRegressor(random_state=random_seed)
    )

    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Split combined back to train/test
    train_data = combined[combined["is_train"] == 1].drop(["is_train"], axis=1)
    test_data = combined[combined["is_train"] == 0].drop(["is_train"], axis=1)

    x_train = train_data.drop([target_col], axis=1)
    y_train = train_data[target_col]
    x_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]

    model_pipeline.fit(x_train, y_train)

    predictions = model_pipeline.predict(x_test)

    return predictions, y_test, task_type


def main(
    synthetic_data: pd.DataFrame,
    challenge_data: pd.DataFrame,
    column_types: dict[str, Any],
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Run feature extraction for EPT attack steps 2 and 3.


    Args:
        synthetic_data: _description_
        challenge_data: _description_
        column_types: _description_
        random_seed: _description_. Defaults to None.

    Returns:
        _description_
    """
    features = []
    columns = []

    for column in synthetic_data.columns:
        log(INFO, f"Extracting features for column: {column}")

        predictions, y_test, task_type = preprocess_train_predict(
            train_points=synthetic_data,
            test_points=challenge_data,
            target_col=column,
            column_types=column_types,
            random_seed=random_seed,
        )

        # __________ tested till here ___________

        features.append(y_test)
        columns.append(column)

        if task_type == TaskType.CLASSIFICATION:
            # Calculate accuracy
            accuracy = predictions == y_test
            accuracy = accuracy.astype(int)
            features.append(accuracy)
            columns.append(f"{column}_accuracy")
        else:
            # Calculate errors
            errors = np.abs(predictions - y_test)
            # Calculate the ratio of the error
            error_ratio = errors / y_test

            # Save the error and the ratio error
            features.append(errors)
            features.append(error_ratio)
            columns.append(f"{column}_error")
            columns.append(f"{column}_error_ratio")

        # predictions from the model
        features.append(predictions)
        columns.append(f"{column}_prediction")

        # Create a DataFrame with the results
    df_results = pd.DataFrame(features).T
    df_results.columns = columns

    return df_results
