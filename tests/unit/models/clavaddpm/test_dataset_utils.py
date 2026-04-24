import numpy as np
import torch

from midst_toolkit.common.dataset_utils import get_category_sizes
from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.models.clavaddpm.dataset_utils import encode_and_merge_features


def test_get_category_sizes() -> None:
    data = [[1, 0, 0], [1, 0, 0], [2, 1, 0], [3, 0, 0]]
    tensor_to_process = torch.Tensor(data)
    array_to_process = np.array(data)

    assert get_category_sizes(tensor_to_process) == [3, 2, 1]
    assert get_category_sizes(array_to_process) == [3, 2, 1]


def test_encode_and_merge_features() -> None:
    set_all_random_seeds(42)

    numerical_features = {
        DataSplit.TRAIN.value: np.random.randn(3, 3),
        DataSplit.VALIDATION.value: np.random.randn(3, 3),
        DataSplit.TEST.value: np.random.randn(3, 3),
    }

    categorical_features = {
        DataSplit.TRAIN.value: np.array([["cat", "dog"], ["lion", "wolf"], ["panther", "wolf"]]),
        DataSplit.VALIDATION.value: np.array([["lion", "wolf"], ["lion", "wolf"], ["panther", "wolf"]]),
        DataSplit.TEST.value: np.array([["panther", "dog"], ["panther", "dingo"], ["panther", "coyote"]]),
    }

    merged_data, encoders = encode_and_merge_features(
        categorical_features=categorical_features, numerical_features=numerical_features, noise_scale=0.1
    )

    assert np.all(merged_data[DataSplit.TRAIN.value][:, 0:3] == numerical_features[DataSplit.TRAIN.value])
    assert np.all(merged_data[DataSplit.VALIDATION.value][:, 0:3] == numerical_features[DataSplit.VALIDATION.value])
    assert np.all(merged_data[DataSplit.TEST.value][:, 0:3] == numerical_features[DataSplit.TEST.value])

    assert np.allclose(
        merged_data[DataSplit.TRAIN.value][:, 3:5],
        np.array([[0.0375698, 2.02088636], [0.93993613, 2.80403299], [1.97083063, 2.8671814]]),
        atol=1e-5,
    )
    assert np.allclose(
        merged_data[DataSplit.VALIDATION.value][:, 3:5],
        np.array([[0.93982934, 3.01968612], [1.18522782, 3.07384666], [1.99865028, 3.01713683]]),
        atol=1e-5,
    )
    assert np.allclose(
        merged_data[DataSplit.TEST.value][:, 3:5],
        np.array([[1.89422891, 1.98843517], [2.08225449, 0.96988963], [1.87791564, -0.1478522]]),
        atol=1e-5,
    )

    assert len(encoders) == 2
    assert encoders[0].classes_.tolist() == ["cat", "lion", "panther"]
    assert encoders[1].classes_.tolist() == ["coyote", "dingo", "dog", "wolf"]

    unset_all_random_seeds()
