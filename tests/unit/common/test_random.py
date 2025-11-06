import random

import numpy as np
import torch

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds


def test_random_seed_setting() -> None:
    set_all_random_seeds(42)

    random_numpy_array = np.random.randint(1, 10, (2, 3))
    random_torch_tensor = torch.randint(1, 10, (2, 3)).float()
    random_number = random.randint(1, 10000)

    assert np.allclose(np.array([[7, 4, 8], [5, 7, 3]]), random_numpy_array, atol=1e-8)
    assert torch.allclose(torch.Tensor([[7, 6, 8], [5, 1, 3]]), random_torch_tensor, atol=1e-8)
    assert random_number == 1825


def test_random_seed_unsetting() -> None:
    # NOTE: This test is unavoidable flaky with an expected failure rate of 1 of every 10,000,000
    set_all_random_seeds(42)
    fixed_random_number = random.randint(1, 10000000)
    set_all_random_seeds(42)
    unset_all_random_seeds()
    # If unset doesn't work, then the number shouldn't be the same (almost) all the time
    random_number = random.randint(1, 10000000)
    assert fixed_random_number != random_number
