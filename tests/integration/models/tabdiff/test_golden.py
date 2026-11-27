"""Integration golden tests for TabDiff preprocess / short train / sample."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.integration.models.tabdiff import golden_runner
from tests.integration.utils import is_running_on_ci_environment


FIXTURE_KEYS = tuple(golden_runner.FIXTURES)


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    work = tmp_path / "tabdiff_golden"
    work.mkdir()
    yield work
    shutil.rmtree(work, ignore_errors=True)


def _compare_preprocess(actual: dict, expected_dir: Path) -> None:
    for name in golden_runner.PREPROCESS_FILES:
        expected_path = expected_dir / name
        if name.endswith(".npy"):
            expected = np.load(expected_path, allow_pickle=True)
            np.testing.assert_allclose(actual[name], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
        elif name.endswith(".csv"):
            expected = pd.read_csv(expected_path)
            pd.testing.assert_frame_equal(actual[name], expected, check_dtype=False)
        else:
            with open(expected_path) as file:
                expected = json.load(file)
            assert actual[name] == expected


def _compare_losses(actual: list[dict], expected_path: Path) -> None:
    with open(expected_path) as file:
        expected = json.load(file)
    expected_epochs = [record for record in expected if "epoch" in record]
    actual_epochs = [record for record in actual if "epoch" in record]
    assert len(actual_epochs) == len(expected_epochs)
    for got, want in zip(actual_epochs, expected_epochs, strict=True):
        assert got["epoch"] == want["epoch"]
        for key in ("loss/c_loss", "loss/d_loss", "loss/total_loss", "closs_weight", "dloss_weight", "lr"):
            assert got[key] == pytest.approx(want[key], rel=0, abs=0)
        for key, value in want.items():
            if key.startswith("num_noise/") or key.startswith("cat_noise/"):
                assert got[key] == pytest.approx(value, rel=1e-5, abs=1e-6)


def _compare_samples(actual: pd.DataFrame, expected_path: Path) -> None:
    expected = pd.read_csv(expected_path)
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    assert not actual.isna().any().any()
    if is_running_on_ci_environment():
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    else:
        for column in actual.columns:
            if pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(
                    actual[column].to_numpy(dtype=float),
                    expected[column].to_numpy(dtype=float),
                    rtol=1e-5,
                    atol=1e-5,
                    equal_nan=True,
                )
            else:
                assert actual[column].astype(str).tolist() == expected[column].astype(str).tolist()


@pytest.mark.integration_test()
@pytest.mark.parametrize("fixture_key", FIXTURE_KEYS)
def test_preprocess_matches_golden(scratch_dir: Path, fixture_key: str) -> None:
    fixture = golden_runner.FIXTURES[fixture_key]
    processed_dir = golden_runner.run_preprocess(scratch_dir, fixture)
    outputs = golden_runner.run_preprocess_outputs(processed_dir, fixture)
    _compare_preprocess(outputs, fixture.golden_dir / "preprocess")


@pytest.mark.integration_test()
@pytest.mark.parametrize("fixture_key", FIXTURE_KEYS)
def test_train_and_sample_match_golden(scratch_dir: Path, fixture_key: str) -> None:
    fixture = golden_runner.FIXTURES[fixture_key]
    processed_dir = golden_runner.run_preprocess(scratch_dir, fixture)
    records, samples = golden_runner.run_train_and_sample(processed_dir, scratch_dir / "train", fixture)
    _compare_losses(records, fixture.golden_dir / "training" / "losses.json")
    _compare_samples(samples, fixture.golden_dir / "sampling" / "samples.csv")


@pytest.mark.integration_test()
@pytest.mark.parametrize("fixture_key", FIXTURE_KEYS)
def test_train_and_sample_deterministic_across_runs(scratch_dir: Path, fixture_key: str) -> None:
    fixture = golden_runner.FIXTURES[fixture_key]
    processed_dir = golden_runner.run_preprocess(scratch_dir, fixture)
    records_a, samples_a = golden_runner.run_train_and_sample(processed_dir, scratch_dir / "run_a", fixture)
    records_b, samples_b = golden_runner.run_train_and_sample(processed_dir, scratch_dir / "run_b", fixture)

    epochs_a = [record for record in records_a if "epoch" in record]
    epochs_b = [record for record in records_b if "epoch" in record]
    assert epochs_a == epochs_b
    pd.testing.assert_frame_equal(samples_a, samples_b, check_dtype=False)
