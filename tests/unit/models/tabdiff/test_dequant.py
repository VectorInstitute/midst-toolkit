"""Unit tests for integer dequantization round trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_dequant_none_is_identity(tabdiff_dir: Path) -> None:
    import src.data as tabdiff_data

    X = np.array([[1.0, 2.5, 3.0], [4.0, 5.5, 6.0]], dtype=np.float64)
    dq = tabdiff_data.dequantizer("none", int_col_idx_wrt_num=[0, 2], int_dequant_factor=0.0)
    out = dq.transform(X.copy())
    back = dq.inverse_transform(out.copy())
    np.testing.assert_allclose(out, X)
    np.testing.assert_allclose(back, X)


def test_dequant_uniform_round_trip_restores_ints(tabdiff_dir: Path) -> None:
    import src.data as tabdiff_data

    X = np.array([[1.0, 2.5, 3.0], [4.0, 5.5, 6.0]], dtype=np.float64)
    dq = tabdiff_data.dequantizer("uniform", int_col_idx_wrt_num=[0, 2], int_dequant_factor=1.0)
    np.random.seed(0)
    noised = dq.transform(X.copy())
    # Non-integer column untouched; int cols shifted into [n, n+factor).
    np.testing.assert_allclose(noised[:, 1], X[:, 1])
    assert np.all(noised[:, 0] >= X[:, 0]) and np.all(noised[:, 0] < X[:, 0] + 1.0)
    assert np.all(noised[:, 2] >= X[:, 2]) and np.all(noised[:, 2] < X[:, 2] + 1.0)
    restored = dq.inverse_transform(noised.copy())
    np.testing.assert_allclose(restored[:, [0, 2]], X[:, [0, 2]])
    np.testing.assert_allclose(restored[:, 1], X[:, 1])
