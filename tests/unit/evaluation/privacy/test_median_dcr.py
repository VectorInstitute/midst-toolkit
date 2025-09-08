import math

import pandas as pd
import pytest
import torch

from midst_toolkit.data_processing.midst_data_processing import load_midst_data
from midst_toolkit.evaluation.privacy.distance_closest_record import (
    MedianDistanceToClosestRecordScore,
    NormType,
    minimum_distances,
    preprocess_for_distance_to_closest_record_score,
)


SYNTHETIC_DATA = torch.Tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0], [3.4, 1.0, 0.3]])
REAL_DATA = torch.Tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 1.0, 2.0], [1.2, 2.3, 0.5]])

SYNTHETIC_DATA_PATH = "tests/assets/synthetic_data_dcr.csv"
REAL_DATA_TRAIN_PATH = "tests/assets/real_data_dcr.csv"
META_INFO_PATH = "tests/assets/meta_info.json"


def test_minimum_distance_no_batch_l1() -> None:
    test_distance = minimum_distances(REAL_DATA, REAL_DATA, skip_diagonal=True)
    target_distance = torch.Tensor([2.6, 3.0, 4.0, 2.6])
    assert torch.allclose(test_distance, target_distance)


def test_minimum_distance_no_batch_l2() -> None:
    test_distance = minimum_distances(REAL_DATA, REAL_DATA, norm=NormType.L2, skip_diagonal=True)
    target_distance = torch.Tensor(
        [
            math.sqrt(0.8 * 0.8 + 1.3 * 1.3 + 0.5 * 0.5),
            math.sqrt(6),
            math.sqrt(6),
            math.sqrt(0.8 * 0.8 + 1.3 * 1.3 + 0.5 * 0.5),
        ]
    )
    assert torch.allclose(test_distance, target_distance)


def test_minimum_distance_l2_no_skip_diagonal() -> None:
    test_distance = minimum_distances(REAL_DATA, REAL_DATA, norm=NormType.L1)
    target_distance = torch.Tensor([0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(test_distance, target_distance)


def test_median_dcr_score() -> None:
    real_data, synthetic_data, meta_info = load_midst_data(REAL_DATA_TRAIN_PATH, SYNTHETIC_DATA_PATH, META_INFO_PATH)

    real_data, _, synthetic_data = preprocess_for_distance_to_closest_record_score(
        synthetic_data, real_data, real_data, meta_info
    )
    dcr_metric = MedianDistanceToClosestRecordScore()
    dcr_score = dcr_metric.compute(real_data, synthetic_data)
    assert pytest.approx(dcr_score["median_dcr_score"], abs=1e-8) == 6.540543187576836


def test_median_dcr_score_dummy() -> None:
    real_data_df = pd.DataFrame(REAL_DATA).astype(float)
    synthetic_df = pd.DataFrame(SYNTHETIC_DATA).astype(float)

    dcr_metric = MedianDistanceToClosestRecordScore()
    dcr_score = dcr_metric.compute(real_data_df, synthetic_df)

    assert pytest.approx(dcr_score["median_dcr_score"], abs=1e-6) == 1.0 / 2.6
