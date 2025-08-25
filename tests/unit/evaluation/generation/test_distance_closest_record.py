import math

import pandas as pd
import pytest
import torch

from midst_toolkit.data_processing.midst_data_processing import load_midst_data_with_test
from midst_toolkit.evaluation.generation_quality.distance_closest_record import (
    NormType,
    distance_to_closest_record_score,
    minimum_distances,
    preprocess_for_distance_to_closest_record_score,
)


SYNTHETIC_DATA = torch.Tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
REAL_DATA = torch.Tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 1.0, 2.0]])
REAL_DATA_TEST = torch.Tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 4.0], [1.0, 2.0, 1.1]])

SYNTHETIC_DATA_PATH = "tests/assets/synthetic_data_dcr.csv"
REAL_DATA_TRAIN_PATH = "tests/assets/real_data_dcr.csv"
REAL_DATA_TEST_PATH = "tests/assets/real_data_test.csv"
META_INFO_PATH = "tests/assets/meta_info.json"


def test_minimum_distance_no_batch_l1() -> None:
    test_distance = minimum_distances(SYNTHETIC_DATA, REAL_DATA)
    target_distance = torch.Tensor([2.0, 0.0])
    torch.allclose(test_distance, target_distance)


def test_minimum_distance_no_batch_l2() -> None:
    test_distance = minimum_distances(SYNTHETIC_DATA, REAL_DATA, norm=NormType.L2)
    target_distance = torch.Tensor([math.sqrt(3.0), 0.0])
    torch.allclose(test_distance, target_distance)


def test_minimum_distance_batch_l1() -> None:
    test_distance = minimum_distances(SYNTHETIC_DATA, REAL_DATA, batch_size=2, norm=NormType.L2)
    target_distance = torch.Tensor([2.0, 0.0])
    torch.allclose(test_distance, target_distance)


def test_minimum_distance_batch_l2() -> None:
    test_distance = minimum_distances(SYNTHETIC_DATA, REAL_DATA, batch_size=2, norm=NormType.L2)
    target_distance = torch.Tensor([math.sqrt(3.0), 0.0])
    torch.allclose(test_distance, target_distance)


def test_dcr_score() -> None:
    real_data_train, synthetic_data, real_data_test, meta_info = load_midst_data_with_test(
        REAL_DATA_TRAIN_PATH, SYNTHETIC_DATA_PATH, META_INFO_PATH, REAL_DATA_TEST_PATH
    )

    real_data_train, real_data_test, synthetic_data = preprocess_for_distance_to_closest_record_score(
        synthetic_data, real_data_train, real_data_test, meta_info
    )
    dcr_score = distance_to_closest_record_score(synthetic_data, real_data_train, real_data_test)
    assert pytest.approx(dcr_score, abs=1e-8) == 0.4715718924999237


def test_dcr_score_dummy() -> None:
    real_data_train_df = pd.DataFrame(REAL_DATA).astype(float)
    real_data_test_df = pd.DataFrame(REAL_DATA_TEST).astype(float)
    synthetic_df = pd.DataFrame(SYNTHETIC_DATA).astype(float)

    dcr_score = distance_to_closest_record_score(synthetic_df, real_data_train_df, real_data_test_df)

    assert dcr_score == 0.5
