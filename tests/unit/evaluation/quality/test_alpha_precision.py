import pytest
import torch

from midst_toolkit.data_processing.midst_data_processing import (
    load_midst_data,
    process_midst_data_for_alpha_precision_evaluation,
)
from midst_toolkit.evaluation.quality.alpha_precision import AlphaPrecision
from midst_toolkit.evaluation.utils import (
    extract_columns_based_on_meta_info,
    one_hot_encode_categoricals_and_merge_with_numerical,
)
from tests.utils.architecture import is_apple_silicon


SYNTHETIC_DATA_PATH = "tests/assets/synthetic_data.csv"
REAL_DATA_PATH = "tests/assets/real_data.csv"
META_INFO_PATH = "tests/assets/meta_info.json"


def test_alpha_precision_evaluation() -> None:
    # TODO: Change to set all random seeds when merged.
    torch.manual_seed(1)

    real_data, synthetic_data, meta_info = load_midst_data(REAL_DATA_PATH, SYNTHETIC_DATA_PATH, META_INFO_PATH)

    numerical_real_data, categorical_real_data = extract_columns_based_on_meta_info(real_data, meta_info)
    numerical_synthetic_data, categorical_synthetic_data = extract_columns_based_on_meta_info(
        synthetic_data, meta_info
    )

    numerical_real_numpy, categorical_real_numpy, numerical_synthetic_numpy, categorical_synthetic_numpy = (
        process_midst_data_for_alpha_precision_evaluation(
            numerical_real_data,
            categorical_real_data,
            numerical_synthetic_data,
            categorical_synthetic_data,
            "default",
            "tabddpm",
        )
    )

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_numpy, categorical_synthetic_numpy, numerical_real_numpy, numerical_synthetic_numpy
    )

    alpha_precision_metric = AlphaPrecision(naive_only=False)

    quality_results = alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)
    if is_apple_silicon():
        assert pytest.approx(0.972538441890166, abs=1e-8) == quality_results["delta_precision_alpha_OC"]
        assert pytest.approx(0.4709851851851852, abs=1e-8) == quality_results["delta_coverage_beta_OC"]
        assert pytest.approx(0.512, abs=1e-8) == quality_results["authenticity_OC"]
        assert pytest.approx(0.05994074074074074, abs=1e-8) == quality_results["delta_precision_alpha_naive"]
        assert pytest.approx(0.005229629629629584, abs=1e-8) == quality_results["delta_coverage_beta_naive"]
        assert pytest.approx(0.9905185185185185, abs=1e-8) == quality_results["authenticity_naive"]
    else:
        assert pytest.approx(0.9732668369518944, abs=1e-8) == quality_results["delta_precision_alpha_OC"]
        assert pytest.approx(0.47238271604938276, abs=1e-8) == quality_results["delta_coverage_beta_OC"]
        assert pytest.approx(0.5102592592592593, abs=1e-8) == quality_results["authenticity_OC"]
        assert pytest.approx(0.05994074074074074, abs=1e-8) == quality_results["delta_precision_alpha_naive"]
        assert pytest.approx(0.005229629629629584, abs=1e-8) == quality_results["delta_coverage_beta_naive"]
        assert pytest.approx(0.9905185185185185, abs=1e-8) == quality_results["authenticity_naive"]
