"""Unit tests for split_num_cat_target / recover_data column order."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_split_and_recover_regression_column_order(tabdiff_dir: Path) -> None:
    """Hand-worked regression layout: syn_data is [target | nums | cats]."""
    from tabdiff.trainer import recover_data, split_num_cat_target

    info = {
        "task_type": "regression",
        "num_col_idx": [0, 1],
        "cat_col_idx": [2],
        "target_col_idx": [3],
        # Packed order: nums, cats, target (same as get_column_name_mapping).
        "idx_mapping": {0: 0, 1: 1, 2: 2, 3: 3},
    }

    # One sample: target=10, n0=1.5, n1=2.5, cat code=0
    syn_data = np.array([[10.0, 1.5, 2.5, 0.0]], dtype=np.float32)

    def _id(x):
        return x

    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, _id, _id, _id)
    np.testing.assert_allclose(syn_target, np.array([[10.0]], dtype=np.float32))
    np.testing.assert_allclose(syn_num, np.array([[1.5, 2.5]], dtype=np.float32))
    np.testing.assert_allclose(syn_cat, np.array([[0.0]]))

    recovered = recover_data(syn_num, syn_cat, syn_target, info)
    assert list(recovered.columns) == [0, 1, 2, 3]
    np.testing.assert_allclose(recovered[0].to_numpy(), [1.5])
    np.testing.assert_allclose(recovered[1].to_numpy(), [2.5])
    np.testing.assert_allclose(recovered[2].to_numpy(), [0.0])
    np.testing.assert_allclose(recovered[3].to_numpy(), [10.0])


def test_split_and_recover_respects_nontrivial_idx_mapping(tabdiff_dir: Path) -> None:
    """Non-monotonic original indices still unpack via idx_mapping."""
    from tabdiff.trainer import recover_data, split_num_cat_target

    # Original: 0=num, 1=cat, 2=num, 3=target → packed nums(0,1), cat(2), target(3)
    info = {
        "task_type": "regression",
        "num_col_idx": [0, 2],
        "cat_col_idx": [1],
        "target_col_idx": [3],
        "idx_mapping": {0: 0, 2: 1, 1: 2, 3: 3},
    }
    # syn_data: [target, num0, num2, cat1]
    syn_data = np.array([[100.0, 7.0, 8.0, 1.0]], dtype=np.float32)

    def _id(x):
        return x

    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, _id, _id, _id)
    recovered = recover_data(syn_num, syn_cat, syn_target, info)
    np.testing.assert_allclose(recovered[0].to_numpy(), [7.0])
    np.testing.assert_allclose(recovered[2].to_numpy(), [8.0])
    np.testing.assert_allclose(recovered[1].to_numpy(), [1.0])
    np.testing.assert_allclose(recovered[3].to_numpy(), [100.0])
    names = {0: "n0", 1: "c0", 2: "n1", 3: "t0"}
    named = recovered.rename(columns=names)
    assert list(named.columns) == ["n0", "c0", "n1", "t0"]
