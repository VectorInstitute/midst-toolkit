from collections import namedtuple
from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from midst_toolkit.attacks.ensemble.rmia.rmia_calculation import (
    Key,
    calculate_rmia_signals,
    conditional_average,
    get_rmia_gower,
)


MockTrainingResult = namedtuple("TrainingResult", ["synthetic_data"])


@pytest.fixture
def base_data() -> dict[str, Any]:
    """Provides base data for testing."""
    df_input = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "city": ["A", "B", "C"],
            "score": [100.0, 200.0, 300.0],
        }
    )

    df_syn1 = pd.DataFrame(
        {
            "id": [101, 102],
            "age": [30, 55],
            "city": ["A", "C"],
            "score": [100.0, 350.0],
        }
    )
    df_syn2 = pd.DataFrame(
        {
            "id": [201, 202, 203],
            "age": [42, 45, 48],
            "city": ["B", "B", "C"],
            "score": [210.0, 220.0, 280.0],
        }
    )

    model_data = {
        "trained_results": [
            MockTrainingResult(synthetic_data=df_syn1),
            MockTrainingResult(synthetic_data=df_syn2),
        ],
        "fine_tuned_results": [
            MockTrainingResult(synthetic_data=df_syn1),
        ],
    }

    return {
        "df_input": df_input,
        "model_data": model_data,
        "categorical_column_names": ["city"],
        "id_column_name": "id",
        "random_seed": 42,
    }


@pytest.fixture
def rmia_signal_data() -> dict[str, Any]:
    """Provides complex mock data for the main calculate_rmia_signals function."""
    k = 2
    df_input = pd.DataFrame({"age": [30, 40, 50], "city": ["A", "B", "C"], "score": [100, 200, 300]})
    id_column_data = pd.Series([1, 2, 3], name="id")

    train_set_0 = pd.DataFrame({"id": [1, 101], "age": [30, 31], "city": ["A", "A"], "score": [100, 110]})
    train_set_1 = pd.DataFrame({"id": [2, 202], "age": [40, 41], "city": ["B", "B"], "score": [200, 210]})
    train_set_2 = pd.DataFrame({"id": [1, 303], "age": [30, 32], "city": ["A", "A"], "score": [100, 120]})

    syn_data_5 = pd.DataFrame(np.random.rand(5, 3), columns=["age", "city", "score"])

    shadow_data_collection = [
        {
            "fine_tuning_sets": [train_set_0["id"].tolist()],
            "fine_tuned_results": [MockTrainingResult(syn_data_5.copy())],
        },
        {
            "fine_tuning_sets": [train_set_1["id"].tolist()],
            "fine_tuned_results": [MockTrainingResult(syn_data_5.copy())],
        },
        {
            "selected_sets": [train_set_2["id"].tolist()],
            "trained_results": [MockTrainingResult(syn_data_5.copy())],
        },
    ]

    target_synthetic_data = MockTrainingResult(syn_data_5.copy()).synthetic_data

    return {
        "df_input": df_input,
        "id_column_data": id_column_data,
        "shadow_data_collection": shadow_data_collection,
        "target_synthetic_data": target_synthetic_data,
        "categorical_column_names": ["city"],
        "id_column_name": "id",
        "k": k,
        "random_seed": 42,
    }


class TestConditionalAverage:
    def test_conditional_average_basic(self):
        """Tests standard column-wise conditional averaging."""
        values = np.array([[10, 20, 30], [2, 4, 6], [100, 200, 300]])
        mask = np.array([[True, False, True], [True, True, False], [False, True, True]])
        expected = np.array([6.0, 102.0, 165.0])
        result = conditional_average(values, mask)
        npt.assert_allclose(result, expected)

    def test_conditional_average_all_false_column(self):
        """Tests that a column with no True in mask results in NaN."""
        values = np.array([[10, 20], [2, 4]])
        mask = np.array([[True, False], [True, False]])
        expected = np.array([6.0, np.nan])
        result = conditional_average(values, mask)
        npt.assert_allclose(result, expected, equal_nan=True)

    def test_conditional_average_all_false_mask(self):
        """Tests that an all-False mask results in all NaNs."""
        values = np.array([[10, 20], [2, 4]])
        mask = np.array([[False, False], [False, False]])
        expected = np.array([np.nan, np.nan])
        result = conditional_average(values, mask)
        npt.assert_allclose(result, expected, equal_nan=True)

    def test_conditional_average_shape_mismatch(self):
        """Tests that mismatched shapes raise an AssertionError."""
        values = np.array([[1, 2], [3, 4]])
        mask = np.array([True, False])
        with pytest.raises(AssertionError, match="condition_mask must have the same shape as values"):
            conditional_average(values, mask)


class TestGetRmiaGower:
    def test_get_rmia_gower_basic_run(self, base_data, mocker):
        """Tests a basic run without sampling, checking mock calls."""
        mock_gower_matrix = mocker.patch(
            "midst_toolkit.attacks.ensemble.rmia.rmia_calculation.gower.gower_matrix",
            side_effect=[
                np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                np.array([[0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            ],
        )

        min_length = 3
        shadow_synthetic_list = [
            train_result.synthetic_data for train_result in base_data["model_data"][Key.TRAINED_RESULTS.value]
        ]
        results = get_rmia_gower(
            df_input=base_data["df_input"],
            model_data=shadow_synthetic_list,
            min_length=min_length,
            categorical_column_names=base_data["categorical_column_names"],
            id_column_name=base_data["id_column_name"],
            random_seed=base_data["random_seed"],
        )

        assert len(results) == 2
        npt.assert_array_equal(results[0], np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        npt.assert_array_equal(
            results[1],
            np.array([[0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        )

        assert mock_gower_matrix.call_count == 2

        call_args_1 = mock_gower_matrix.call_args_list[0].kwargs
        pdt.assert_frame_equal(call_args_1["data_x"], base_data["df_input"], check_dtype=False)
        syn_data_1_dropped = base_data["model_data"]["trained_results"][0].synthetic_data.drop(columns=["id"])
        pdt.assert_frame_equal(call_args_1["data_y"], syn_data_1_dropped, check_dtype=False)
        assert call_args_1["cat_features"] == [False, True, False]

        call_args_2 = mock_gower_matrix.call_args_list[1].kwargs
        syn_data_2_dropped = base_data["model_data"]["trained_results"][1].synthetic_data.drop(columns=["id"])
        pdt.assert_frame_equal(call_args_2["data_y"], syn_data_2_dropped, check_dtype=False)

    def test_get_rmia_gower_with_sampling(self, base_data, mocker):
        """Tests that sampling is triggered and random_state is used."""
        mock_gower_matrix = mocker.patch(
            "midst_toolkit.attacks.ensemble.rmia.rmia_calculation.gower.gower_matrix",
            return_value=np.array([[0.1], [0.2], [0.3]]),
        )

        original_syn_data = base_data["model_data"]["trained_results"][1].synthetic_data

        mock_sample = mocker.patch("pandas.DataFrame.sample", wraps=original_syn_data.sample)

        min_length = 2
        synthetic_data_list = [data.synthetic_data for data in base_data["model_data"][Key.TRAINED_RESULTS.value]]
        get_rmia_gower(
            df_input=base_data["df_input"],
            model_data=synthetic_data_list,
            min_length=min_length,
            categorical_column_names=base_data["categorical_column_names"],
            id_column_name=base_data["id_column_name"],
            random_seed=base_data["random_seed"],
        )

        assert mock_gower_matrix.call_count == 2
        mock_sample.assert_called_once_with(n=min_length, random_state=base_data["random_seed"])

        call_args_2 = mock_gower_matrix.call_args_list[1].kwargs
        expected_sampled_data = original_syn_data.sample(n=min_length, random_state=base_data["random_seed"]).drop(
            columns=[base_data["id_column_name"]]
        )
        pdt.assert_frame_equal(call_args_2["data_y"], expected_sampled_data, check_dtype=False)

    def test_get_rmia_gower_missing_categorical_column(self, base_data, mocker, caplog):
        """Tests that a warning is logged for missing categorical columns."""
        mocker.patch(
            "midst_toolkit.attacks.ensemble.rmia.rmia_calculation.gower.gower_matrix", return_value=np.array([[0.1]])
        )

        missing_cat_cols = ["city", "non_existent_column"]

        with caplog.at_level("INFO"):
            synthetic_data_list = [data.synthetic_data for data in base_data["model_data"][Key.FINE_TUNED_RESULTS.value]]
            get_rmia_gower(
                df_input=base_data["df_input"],
                model_data=synthetic_data_list,
                min_length=1,
                categorical_column_names=missing_cat_cols,
                id_column_name=base_data["id_column_name"],
            )

        assert "Warning: The following categorical columns are missing" in caplog.text
        assert "{'non_existent_column'}" in caplog.text


class TestCalculateRmiaSignals:
    @pytest.fixture
    def mock_dependencies(self, mocker, rmia_signal_data):
        """Mocks dependencies for calculate_rmia_signals."""
        gower_shadow_0 = [np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.5, 0.4, 0.3, 0.2], [0.9, 0.9, 0.8, 0.7, 0.6]])]
        gower_shadow_1 = [np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.5, 0.4, 0.3, 0.2, 0.1], [0.8, 0.7, 0.6, 0.9, 1.0]])]
        gower_shadow_2 = [np.array([[0.3, 0.4, 0.5, 0.6, 0.7], [0.4, 0.3, 0.2, 0.1, 0.0], [0.7, 0.6, 0.8, 0.9, 0.5]])]
        gower_target = [
            np.array([[0.05, 0.1, 0.15, 0.2, 0.25], [0.01, 0.02, 0.03, 0.04, 0.05], [0.8, 0.82, 0.84, 0.86, 0.88]])
        ]

        mock_get_gower = mocker.patch(
            "midst_toolkit.attacks.ensemble.rmia.rmia_calculation.get_rmia_gower",
            side_effect=[gower_shadow_0, gower_shadow_1, gower_shadow_2, gower_target],
            autospec=True,
        )

        return mock_get_gower, gower_shadow_0, gower_shadow_1, gower_shadow_2, gower_target

    def test_calculate_rmia_signals_main_logic(self, rmia_signal_data, mock_dependencies):
        """Tests the main orchestration and calculation logic of the function."""
        k = rmia_signal_data["k"]

        result_df = calculate_rmia_signals(**rmia_signal_data)

        signal_target_k_2 = np.array([0.075, 0.015, 0.81])
        signal_target_k_1 = np.array([0.05, 0.01, 0.8])

        signal_shadows_k_1 = np.array([0.2, 0.1, 0.56666667])
        signal_shadows_k_2 = np.array([0.25, 0.15, 0.61666667])

        signal_shadows_in_k_1 = np.array([0.2, 0.1, np.nan])
        signal_shadows_in_k_2 = np.array([0.25, 0.15, np.nan])

        signal_shadows_out_k_1 = np.array([0.2, 0.1, 0.56666667])
        signal_shadows_out_k_2 = np.array([0.25, 0.15, 0.61666667])

        rmia_k_1 = np.array([0.25, 0.1, 1.4117647])
        rmia_k_2 = np.array([0.3, 0.1, 1.3135593])
        rmia_out_k_1 = np.array([0.25, 0.1, 1.4117647])
        rmia_out_k_2 = np.array([0.3, 0.1, 1.3135593])

        expected_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "signal_shadow_k_1": signal_shadows_k_1,
                f"signal_shadow_k_{k}": signal_shadows_k_2,
                "signal_shadows_in_k_1": signal_shadows_in_k_1,
                f"signal_shadows_in_k_{k}": signal_shadows_in_k_2,
                "signal_shadows_out_k_1": signal_shadows_out_k_1,
                f"signal_shadows_out_k_{k}": signal_shadows_out_k_2,
                "signal_target_k_1": signal_target_k_1,
                f"signal_target_k_{k}": signal_target_k_2,
                "rmia_k_1": rmia_k_1,
                f"rmia_k_{k}": rmia_k_2,
                "rmia_out_k_1": rmia_out_k_1,
                f"rmia_out_k_{k}": rmia_out_k_2,
            }
        )

        pdt.assert_frame_equal(result_df, expected_df, check_dtype=False, atol=0.005)

    def test_calculate_rmia_signals_value_errors(self, rmia_signal_data):
        """Tests that ValueErrors are raised for invalid k or empty data."""
        data_k0 = rmia_signal_data.copy()
        data_k0["k"] = 0
        with pytest.raises(ValueError):
            calculate_rmia_signals(**data_k0)

        data_empty = rmia_signal_data.copy()
        data_empty["shadow_data_collection"][0]["fine_tuning_sets"] = []
        with pytest.raises(ValueError, match="contain empty sets"):
            calculate_rmia_signals(**data_empty)

    def test_calculate_rmia_signals_division_by_zero(self, rmia_signal_data, mocker):
        """Tests that division by zero in RMIA score results in NaN."""
        k = rmia_signal_data["k"]

        gower_zeros = [np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])]
        gower_target = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])]

        mocker.patch(
            "midst_toolkit.attacks.ensemble.rmia.rmia_calculation.get_rmia_gower",
            side_effect=[gower_zeros, gower_zeros, gower_zeros, gower_target],
            autospec=True,
        )

        mocker.patch(
            "midst_toolkit.attacks.ensemble.rmia.rmia_calculation.conditional_average",
            return_value=np.array([0.0, 0.0, 0.0]),
            autospec=True,
        )

        result_df = calculate_rmia_signals(**rmia_signal_data)

        assert (result_df["signal_shadow_k_1"] == 0.0).all()
        assert (result_df[f"signal_shadow_k_{k}"] == 0.0).all()
        assert (result_df["signal_shadows_out_k_1"] == 0.0).all()
        assert (result_df[f"signal_shadows_out_k_{k}"] == 0.0).all()

        assert (result_df["signal_target_k_1"] > 0.0).all()
        assert (result_df[f"signal_target_k_{k}"] > 0.0).all()

        assert result_df["rmia_k_1"].isna().all()
        assert result_df[f"rmia_k_{k}"].isna().all()
        assert result_df["rmia_out_k_1"].isna().all()
        assert result_df[f"rmia_out_k_{k}"].isna().all()
