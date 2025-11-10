import numpy as np
import pytest

from midst_toolkit.common.enumerations import TaskType
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.models.clavaddpm.dataset_transformations import (
    CAT_MISSING_VALUE,
    CAT_RARE_VALUE,
    collapse_rare_categories,
    encode_categorical_features,
    process_nans_in_categorical_features,
    transform_targets,
)
from midst_toolkit.models.clavaddpm.enumerations import CategoricalEncoding, CategoricalNaNPolicy, TargetPolicy


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


def test_encode_categorical_features() -> None:
    set_all_random_seeds(42)
    categorical_features = {
        "train": np.array([["cat", "dog"], ["lion", "wolf"], ["panther", "wolf"]]),
        "val": np.array([["lion", "wolf"], ["lion", "wolf"], ["panther", "wolf"]]),
        "test": np.array([["panther", "dog"], ["panther", "dingo"], ["panther", "coyote"]]),
    }

    encoded_features, is_numeric, encoder = encode_categorical_features(
        categorical_features, CategoricalEncoding.ORDINAL, None, None, True
    )
    assert not is_numeric
    assert encoder is not None
    assert encoder["ordinalencoder"].categories_[0].tolist() == ["cat", "lion", "panther"]
    assert encoder["ordinalencoder"].categories_[1].tolist() == ["dog", "wolf"]

    assert np.all(encoded_features["train"] == np.array([[0, 0], [1, 1], [2, 1]]))
    assert np.all(encoded_features["val"] == np.array([[1, 1], [1, 1], [2, 1]]))
    # Because dingo and coyote are unknown, they should be 1 "higher" than the largest encoded value in train (1)
    assert np.all(encoded_features["test"] == np.array([[2, 0], [2, 2], [2, 2]]))

    encoded_features, is_numeric, encoder = encode_categorical_features(categorical_features, None, None, None, False)
    assert not is_numeric
    assert encoder is None
    # Values should be the same as above, since ordinal is the default
    assert np.all(encoded_features["train"] == np.array([[0, 0], [1, 1], [2, 1]]))
    assert np.all(encoded_features["val"] == np.array([[1, 1], [1, 1], [2, 1]]))
    # Because dingo and coyote are unknown, they should be 1 "higher" than the largest encoded value in train (1)
    assert np.all(encoded_features["test"] == np.array([[2, 0], [2, 2], [2, 2]]))

    encoded_features, is_numeric, encoder = encode_categorical_features(
        categorical_features, CategoricalEncoding.ONE_HOT, None, None, False
    )
    assert is_numeric
    assert encoder is None
    assert np.all(
        encoded_features["train"]
        == np.array([[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0]]).astype("float")
    )
    assert np.all(
        encoded_features["val"]
        == np.array([[0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0]]).astype("float")
    )
    # Note that the two examples with dingo and coyote or full-zero one-hot vectors because they are unknowns from
    # the perspective of train and onehotencoder has handle_unknown as ignore
    assert np.all(
        encoded_features["test"]
        == np.array([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]).astype("float")
    )
    unset_all_random_seeds()


def test_transform_targets() -> None:
    set_all_random_seeds(42)
    target_splits = {
        "train": np.random.randn(3, 1).astype(float),
        "val": np.random.randn(3, 1).astype(float),
        "test": np.random.randn(3, 1).astype(float),
    }
    # Nothing should happen in these settings
    new_targets, _ = transform_targets(target_splits, TargetPolicy.DEFAULT, task_type=TaskType.BINARY_CLASSIFICATION)
    assert np.allclose(target_splits["train"], new_targets["train"], atol=1e-9)
    assert np.allclose(target_splits["val"], new_targets["val"], atol=1e-9)
    assert np.allclose(target_splits["test"], new_targets["test"], atol=1e-9)
    new_targets, _ = transform_targets(target_splits, None, task_type=TaskType.BINARY_CLASSIFICATION)
    assert np.allclose(target_splits["train"], new_targets["train"], atol=1e-9)
    assert np.allclose(target_splits["val"], new_targets["val"], atol=1e-9)
    assert np.allclose(target_splits["test"], new_targets["test"], atol=1e-9)

    new_targets, info = transform_targets(target_splits, TargetPolicy.DEFAULT, task_type=TaskType.REGRESSION)
    assert pytest.approx(info.mean, abs=1e-5) == 0.335379
    assert pytest.approx(info.std, abs=1e-5) == 0.340540

    assert np.allclose(new_targets["train"], np.array([[0.473763, -1.39086, 0.917101]]).T, atol=1e-4)
    assert np.allclose(new_targets["val"], np.array([[3.48755, -1.67244, -1.67239]]).T, atol=1e-4)
    assert np.allclose(new_targets["test"], np.array([[3.65253, 1.26874, -2.36346]]).T, atol=1e-4)

    unset_all_random_seeds()
