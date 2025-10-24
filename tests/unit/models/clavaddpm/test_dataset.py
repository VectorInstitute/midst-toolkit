from copy import deepcopy

import numpy as np
import pytest

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.models.clavaddpm.dataset import (
    CAT_MISSING_VALUE,
    CAT_RARE_VALUE,
    Dataset,
    NumericalNaNPolicy,
    TaskType,
    collapse_rare_categories,
    process_nans_in_categorical_features,
    process_nans_in_numerical_features,
)
from midst_toolkit.models.clavaddpm.enumerations import CategoricalNaNPolicy


def _compare_data_splits(test_splits: dict[str, np.ndarray], reference_splits: dict[str, np.ndarray]) -> bool:
    tests = []
    for key, data in test_splits.items():
        tests.append(np.all(data == reference_splits[key]))
    return np.all(tests)


def _get_test_dataset() -> Dataset:
    set_all_random_seeds(42)

    numerical_data_splits = {
        "train": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
        "val": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
        "test": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
    }
    categorical_data_splits = {
        "train": np.random.randint(low=0, high=2, size=(3, 3)).astype(str),
        "val": np.random.randint(low=0, high=2, size=(3, 3)).astype(str),
        "test": np.random.randint(low=0, high=2, size=(3, 3)).astype(str),
    }
    label_splits = {
        "train": np.random.randint(low=0, high=2, size=(3, 3)).astype(int),
        "val": np.random.randint(low=0, high=2, size=(3, 3)).astype(int),
        "test": np.random.randint(low=0, high=2, size=(3, 3)).astype(int),
    }

    dataset = Dataset(
        x_num=numerical_data_splits,
        x_cat=categorical_data_splits,
        y=label_splits,
        y_info={},
        task_type=TaskType.BINCLASS,
        n_classes=2,
    )
    unset_all_random_seeds()
    return dataset


def test_process_nans_in_categorical_features() -> None:
    set_all_random_seeds(42)
    data_splits = {
        "train": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
        "val": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
        "test": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
    }
    # Test when there are no NaNs
    processed_data_splits = process_nans_in_categorical_features(data_splits, CategoricalNaNPolicy.MOST_FREQUENT)
    assert np.all(data_splits["train"] == processed_data_splits["train"])
    assert np.all(data_splits["val"] == processed_data_splits["val"])
    assert np.all(data_splits["test"] == processed_data_splits["test"])

    # Make one of the train data set NaN but no others
    data_splits["train"][0, 1] = float("nan")
    processed_data_splits = process_nans_in_categorical_features(data_splits, CategoricalNaNPolicy.MOST_FREQUENT)
    assert processed_data_splits["train"][0, 1] == 0
    assert np.all(data_splits["val"] == processed_data_splits["val"])
    assert np.all(data_splits["test"] == processed_data_splits["test"])

    # Try when test has a NaN
    data_splits["test"][1, 1] = float("nan")
    processed_data_splits = process_nans_in_categorical_features(data_splits, CategoricalNaNPolicy.MOST_FREQUENT)
    assert processed_data_splits["train"][0, 1] == 0
    assert processed_data_splits["test"][1, 1] == 0
    assert np.all(data_splits["val"] == processed_data_splits["val"])

    # Try when no policy is provided
    processed_data_splits = process_nans_in_categorical_features(data_splits, policy=None)
    # NaNs should be left alone
    assert np.isnan(data_splits["train"][0, 1])
    assert np.isnan(data_splits["test"][1, 1])

    # Try with string values rather than numbers
    data_splits = {k: v.astype(str).astype(object) for k, v in data_splits.items()}
    data_splits["train"][0, 1] = CAT_MISSING_VALUE
    data_splits["test"][1, 1] = CAT_MISSING_VALUE
    processed_data_splits = process_nans_in_categorical_features(data_splits, CategoricalNaNPolicy.MOST_FREQUENT)
    assert processed_data_splits["train"][0, 1] == "0.0"
    assert processed_data_splits["test"][1, 1] == "0.0"

    unset_all_random_seeds()


def test_collapse_rare_values() -> None:
    set_all_random_seeds(42)
    data_splits = {
        "train": np.random.randint(low=0, high=2, size=(10, 10)).astype(str),
        "val": np.random.randint(low=0, high=2, size=(10, 10)).astype(str),
        "test": np.random.randint(low=0, high=2, size=(10, 10)).astype(str),
    }
    # Based on these settings, column index 6 in the train split ends up with 0 being rare (1 entry of 10)
    # So it should be replaced with CAT_RARE_VALUE in all datasets. Otherwise, everywhere else should be equal
    processed_data_splits = collapse_rare_categories(data_splits, 0.2)
    assert processed_data_splits["train"][0, 6] == CAT_RARE_VALUE
    assert processed_data_splits["val"][1, 6] == CAT_RARE_VALUE
    assert processed_data_splits["val"][3, 6] == CAT_RARE_VALUE
    assert processed_data_splits["val"][4, 6] == CAT_RARE_VALUE
    assert processed_data_splits["val"][5, 6] == CAT_RARE_VALUE
    assert processed_data_splits["val"][7, 6] == CAT_RARE_VALUE
    assert processed_data_splits["test"][2, 6] == CAT_RARE_VALUE
    assert processed_data_splits["test"][4, 6] == CAT_RARE_VALUE
    assert processed_data_splits["test"][5, 6] == CAT_RARE_VALUE
    assert processed_data_splits["test"][7, 6] == CAT_RARE_VALUE
    assert processed_data_splits["test"][8, 6] == CAT_RARE_VALUE
    # Make sure there are no other rares
    assert np.sum(processed_data_splits["train"] != data_splits["train"]) == 1
    assert np.sum(processed_data_splits["val"] != data_splits["val"]) == 5
    assert np.sum(processed_data_splits["test"] != data_splits["test"]) == 5

    # Now we create rare ones in both the train and validation sets
    data_splits["train"][0, 1] = "5"
    data_splits["val"][2, 1] = "5"
    processed_data_splits = collapse_rare_categories(data_splits, 0.2)
    assert processed_data_splits["train"][0, 1] == CAT_RARE_VALUE
    assert processed_data_splits["val"][2, 1] == CAT_RARE_VALUE

    unset_all_random_seeds()


def test_process_nans_in_numerical_features_drop() -> None:
    dataset = _get_test_dataset()
    numerical_data_splits = deepcopy(dataset.x_num)
    categorical_data_splits = deepcopy(dataset.x_cat)
    label_splits = deepcopy(dataset.y)

    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.DROP_ROWS)
    assert _compare_data_splits(dataset.x_num, numerical_data_splits)
    assert _compare_data_splits(dataset.x_cat, categorical_data_splits)
    assert _compare_data_splits(dataset.y, label_splits)

    # Now add some NaNs to the train and validation splits
    dataset.x_num["train"][0, 1] = np.NaN
    dataset.x_num["val"][1, 1] = np.NaN
    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.DROP_ROWS)
    # Make sure first row of train in all dataset components is dropped
    assert len(dataset.x_num["train"]) == 2
    assert len(dataset.x_cat["train"]) == 2
    assert len(dataset.y["train"]) == 2
    assert np.all(dataset.x_num["train"] == numerical_data_splits["train"][1:, :])
    assert np.all(dataset.x_cat["train"] == categorical_data_splits["train"][1:, :])
    assert np.all(dataset.y["train"] == label_splits["train"][1:, :])
    # Make sure second row of val in all dataset components is dropped
    assert len(dataset.x_num["val"]) == 2
    assert len(dataset.x_cat["val"]) == 2
    assert len(dataset.y["val"]) == 2
    assert np.all(
        dataset.x_num["val"] == np.vstack((numerical_data_splits["val"][0, :], numerical_data_splits["val"][2, :]))
    )
    assert np.all(
        dataset.x_cat["val"] == np.vstack((categorical_data_splits["val"][0, :], categorical_data_splits["val"][2, :]))
    )
    assert np.all(dataset.y["val"] == np.vstack((label_splits["val"][0, :], label_splits["val"][2, :])))
    assert np.all(dataset.y["test"] == label_splits["test"])

    # Now add NaN to test and make sure we throw.
    dataset.x_num["test"][1, 1] = np.NaN

    with pytest.raises(AssertionError):
        dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.DROP_ROWS)


def test_process_nans_in_numerical_features_mean() -> None:
    dataset = _get_test_dataset()
    numerical_data_splits = deepcopy(dataset.x_num)
    categorical_data_splits = deepcopy(dataset.x_cat)
    label_splits = deepcopy(dataset.y)

    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.MEAN)
    assert _compare_data_splits(dataset.x_num, numerical_data_splits)
    assert _compare_data_splits(dataset.x_cat, categorical_data_splits)
    assert _compare_data_splits(dataset.y, label_splits)

    # Now add some NaNs to the train and validation splits
    dataset.x_num["train"][0, 1] = np.NaN
    dataset.x_num["val"][1, 1] = np.NaN
    # Adding a NaN to a column that doesn't have a NaN in train
    dataset.x_num["val"][1, 2] = np.NaN
    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.MEAN)
    # Nothing should change in the label and cat rows now
    assert _compare_data_splits(dataset.x_cat, categorical_data_splits)
    assert _compare_data_splits(dataset.y, label_splits)
    assert dataset.x_num["train"][0, 1] == 0
    assert dataset.x_num["val"][1, 1] == 0
    assert dataset.x_num["val"][1, 2] == 1.0 / 3.0

    unset_all_random_seeds()
