"""
Module to run feature extraction for EPT attack steps 2 and 3.
Overall workflow and decisions are taken with from the Cyber@BGU team's attack implementation at
https://github.com/eyalgerman/MIA-EPT.

"""

from enum import Enum
from logging import INFO

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
    column_types: dict[str, list[str]],
    random_seed: int | None = None,
) -> tuple[np.ndarray, pd.Series, TaskType]:
    """
    An attribute prediction model is trained on `train_points` to predict the `target_col`.

    We determine the nature of the prediction task based on the data type of the target column.
    If the `target_col` is categorical, the model uses a classification approach. Otherwise, if
    the `target_col` is numerical, a regression model is used. This allows the
    model to effectively learn the relationship between the `target_col` and the other attributes
    present in the training data.

    After the model is trained on `train_points`, it is then used to generate predictions for the `target_col`
    on `test_points`.

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
    train_and_test_data = pd.concat([df_train, df_test], axis=0)

    numeric_columns = column_types["numerical"]
    categorical_columns = column_types["categorical"]

    # Assert that the target column appears exactly once in numeric_columns + categorical_columns
    assert (numeric_columns + categorical_columns).count(target_col) == 1, (
        f"The target column '{target_col}' must appear exactly once in numeric_columns + categorical_columns"
    )

    # Assert that the union of numeric_columns and categorical_columns matches
    # the columns in the combined dataframe, except for 'is_train'
    assert set(numeric_columns + categorical_columns) == set(train_and_test_data.columns) - {"is_train"}, (
        "The union of numeric_columns and categorical_columns must match the columns in the combined dataframe, except for 'is_train'"
    )

    task_type = TaskType.CLASSIFICATION if target_col in categorical_columns else TaskType.REGRESSION

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

    model = (
        RandomForestClassifier(random_state=random_seed)
        if task_type == TaskType.CLASSIFICATION
        else RandomForestRegressor(random_state=random_seed)
    )

    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Split combined back to train/test
    train_data = train_and_test_data[train_and_test_data["is_train"] == 1].drop(["is_train"], axis=1)
    test_data = train_and_test_data[train_and_test_data["is_train"] == 0].drop(["is_train"], axis=1)

    x_train = train_data.drop([target_col], axis=1)
    y_train = train_data[target_col]
    x_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]

    model_pipeline.fit(x_train, y_train)

    predictions = model_pipeline.predict(x_test)

    return predictions, y_test, task_type


def extract_features(
    synthetic_data: pd.DataFrame,
    challenge_data: pd.DataFrame,
    column_types: dict[str, list[str]],
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Orchestrator function to run feature extraction for EPT attack:
    1. For each attribute (column) in the synthetic data that is not an ID, train an attribute prediction model
        using the synthetic data.
    2. Use the trained model to predict the values of that attribute in the challenge data, which also doesn't
        contain IDs.
    3. Compute relevant metrics (accuracy for categorical data, error and error ratio for numerical data).
    4. Compile the results into a DataFrame.

    Args:
        synthetic_data: Synthetic data generated by the target model, the data we want to extract features from.
        challenge_data: The data the predictions are compared against, to compute prediction accuracy/errors.
        column_types: A dictionary specifying the types of columns (numerical or categorical) in the data.
        random_seed: Random seed for reproducibility. Defaults to None.

    Returns:
        A DataFrame containing the extracted features for each attribute in the challenge data.
        It includes the following columns:
            - <column_name>: The true values for the attribute.
            - <column_name>_prediction: The predicted values for the attribute.
        If the data is categorical:
            - <column_name>_accuracy: The element-wise accuracy of the predictions. 0 for incorrect prediction,
                1 for correct.
        If the data is numerical:
            - <column_name>_error (if regression): The absolute errors of the predictions.
            - <column_name>_error_ratio (if regression): The ratio of the errors to the true values, which is
                derived by dividing the absolute error by the true value in a zero-safe manner.
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

        features.append(y_test)
        columns.append(column)

        if task_type == TaskType.CLASSIFICATION:
            # Calculate accuracy
            accuracy = predictions == y_test
            accuracy = accuracy.astype(int)
            features.append(accuracy)
            columns.append(f"{column}_accuracy")

        elif task_type == TaskType.REGRESSION:
            # Calculate errors
            errors = pd.Series(np.abs(predictions - y_test), index=y_test.index)

            # Calculate the ratio of the error in a zero-safe manner
            denominator = y_test.replace(0, np.nan)
            error_ratio = errors / denominator

            # Replace infs and NaNs with a large number. If all values are NaN, replace with 1e9.
            finite_max = error_ratio[np.isfinite(error_ratio)].max()
            error_ratio = error_ratio.replace([np.inf, -np.inf], np.nan).fillna(
                finite_max if pd.notna(finite_max) else 1e9
            )

            # Save the error and the ratio error
            features.append(errors)
            features.append(error_ratio)

            columns.append(f"{column}_error")
            columns.append(f"{column}_error_ratio")

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # predictions from the model
        features.append(pd.Series(predictions, index=y_test.index))
        columns.append(f"{column}_prediction")

        # Create a DataFrame with the results
    df_results = pd.DataFrame(features).T
    df_results.columns = columns

    return df_results
