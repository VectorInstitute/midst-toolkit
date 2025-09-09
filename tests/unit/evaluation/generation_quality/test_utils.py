import os
from pathlib import Path

import pytest

from midst_toolkit.evaluation.generation.utils import (
    create_quality_metrics_directory,
    dump_metrics_dict,
)


def test_create_quality_metrics_directory_new(tmp_path: Path) -> None:
    create_quality_metrics_directory(tmp_path / "metrics")
    assert os.path.exists(tmp_path / "metrics")


def test_create_quality_metrics_directory_exist(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    checkpoint_dir = tmp_path.joinpath("metrics")
    checkpoint_dir.mkdir()
    create_quality_metrics_directory(tmp_path / "metrics")
    assert "already exists. Make sure this is intended." in caplog.text


def test_dump_metrics_dict_new(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    checkpoint_dir = tmp_path.joinpath("metrics")
    checkpoint_dir.mkdir()

    metrics_path = tmp_path / "metrics" / "metrics_test.txt"

    metrics_dict = {"metric_1": 2.5}
    dump_metrics_dict(metrics_dict, metrics_path)

    with open(metrics_path, "r") as f:
        metric_lines = f.readlines()

    assert metric_lines[0] == "Metric Name: metric_1\t Metric Value: 2.5\n"

    # Dump the metrics, but then overwrite them. Assert we get our warning log.

    metrics_dict = {"metric_1": 1.23, "metric_2": 3.45}
    dump_metrics_dict(metrics_dict, metrics_path)
    with open(metrics_path, "r") as f:
        metric_lines = f.readlines()

    assert metric_lines[0] == "Metric Name: metric_1\t Metric Value: 1.23\n"
    assert metric_lines[1] == "Metric Name: metric_2\t Metric Value: 3.45\n"

    assert f"File at path {metrics_path} already exists." in caplog.text
