from enum import Enum
from logging import INFO
from midst_toolkit.common.logger import log

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
from typing import Any


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


def preprocess_and_train(train_points: pd.DataFrame, test_points: pd.DataFrame, target_col: str, column_types: dict[str, Any], random_seed: int | None = None) -> tuple[np.ndarray, pd.Series, TaskType]:

    train_points['is_train'] = 1
    test_points['is_train'] = 0

    # Ensure columns in train_points and test_points match
    assert set(train_points.columns) == set(test_points.columns), "Columns in train_points and test_points do not match"
    combined = pd.concat([train_points, test_points], axis=0)

    numeric_columns = column_types["numerical_columns"]
    categorical_columns = column_types["categorical_columns"]

    # Assert that the union of numeric_columns and categorical_columns matches the columns in the combined dataframe, except for 'is_train'
    assert set(numeric_columns + categorical_columns) == set(combined.columns) - {'is_train'}, \
        "The union of numeric_columns and categorical_columns must match the columns in the combined dataframe, except for 'is_train'"

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    task_type = TaskType.CLASSIFICATION if target_col in categorical_columns else TaskType.REGRESSION

    model = RandomForestClassifier(random_state=random_seed) if task_type == TaskType.CLASSIFICATION else RandomForestRegressor(random_state=random_seed)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Split combined back to train/test
    train_data = combined[combined['is_train'] == 1].drop(['is_train'], axis=1)
    test_data = combined[combined['is_train'] == 0].drop(['is_train'], axis=1)

    X_train = train_data.drop([target_col], axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Generate predictions
    predictions = model_pipeline.predict(X_test)

    return predictions, y_test, task_type



def main(synthetic_data: pd.DataFrame, challenge_data: pd.DataFrame, challenge_labels: pd.DataFrame, column_types: dict[str, Any], random_seed: int | None = None) -> pd.DataFrame:


    features = []
    columns = []

    for column in synthetic_data.columns:
        log(INFO, f"Extracting features for column: {column}")

        predictions, y_test, task_type = preprocess_and_train(train_points=synthetic_data, test_points=challenge_data, target_col=column, column_types=column_types)

    # Placeholder for actual feature extraction logic
    # For demonstration, we will just merge the dataframes
    import pdb; pdb.set_trace()
    features = synthetic_data.merge(challenge_data, on='id').merge(challenge_labels, on='id')
    return features
