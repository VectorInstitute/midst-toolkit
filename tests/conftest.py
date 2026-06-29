"""Conftest."""

import os


os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest


@pytest.fixture
def my_test_number() -> int:
    """My test number.

    Returns:
    -------
        int: A really awesome number.
    """
    return 42
