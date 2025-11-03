import pytest

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
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


def test_alpha_precision_evaluation() -> None:
    synthetic_data_path = "tests/assets/synthetic_data.csv"
    real_data_path = "tests/assets/real_data.csv"
    meta_info_path = "tests/assets/meta_info.json"

    set_all_random_seeds(1)

    real_data, synthetic_data, meta_info = load_midst_data(real_data_path, synthetic_data_path, meta_info_path)

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
        assert pytest.approx(0.9717975308641975, abs=1e-8) == quality_results["delta_precision_alpha_OC"]
        assert pytest.approx(0.472162962962963, abs=1e-8) == quality_results["delta_coverage_beta_OC"]
        assert pytest.approx(0.5063333333333333, abs=1e-8) == quality_results["authenticity_OC"]
        assert pytest.approx(0.05994074074074074, abs=1e-8) == quality_results["delta_precision_alpha_naive"]
        assert pytest.approx(0.005229629629629584, abs=1e-8) == quality_results["delta_coverage_beta_naive"]
        assert pytest.approx(0.9905185185185185, abs=1e-8) == quality_results["authenticity_naive"]

    # Unset seed for safety
    unset_all_random_seeds()


def test_alpha_precision_computation_handles_real_and_synthetic_length_mismatch_smaller_synthetic() -> None:
    synthetic_data_path = (
        "/projects/midst-experiments/all_tabddpms/tabddpm_trained_with_20k/train/tabddpm_1/synthetic_data/2k/2k.csv"
    )
    real_data_path = (
        "/projects/midst-experiments/all_tabddpms/tabddpm_trained_with_20k/train/tabddpm_1/train_with_id.csv"
    )
    meta_info_path = "/projects/midst-experiments/trans_temp.json"
    set_all_random_seeds(1)

    real_data, synthetic_data, meta_info = load_midst_data(real_data_path, synthetic_data_path, meta_info_path)

    # --- Clean ---
    for df in [real_data, synthetic_data]:
        df.drop(columns=["trans_id", "account_id"], inplace=True, errors="ignore")

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

    alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)

    unset_all_random_seeds()


def test_alpha_precision_computation_handles_real_and_synthetic_length_mismatch_larger_synthetic() -> None:
    synthetic_data_path = "/projects/midst-experiments/all_tabddpms/tabddpm_trained_with_20k/train/tabddpm_1/synthetic_data/200k/200k.csv"
    real_data_path = (
        "/projects/midst-experiments/all_tabddpms/tabddpm_trained_with_20k/train/tabddpm_1/train_with_id.csv"
    )
    meta_info_path = "/projects/midst-experiments/trans_temp.json"
    set_all_random_seeds(1)

    real_data, synthetic_data, meta_info = load_midst_data(real_data_path, synthetic_data_path, meta_info_path)

    # --- Clean ---
    for df in [real_data, synthetic_data]:
        df.drop(columns=["trans_id", "account_id"], inplace=True, errors="ignore")

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

    alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)

    unset_all_random_seeds()
