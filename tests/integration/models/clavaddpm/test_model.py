import json
import os
import pickle
import platform
from pathlib import Path

import numpy as np
import pytest

from midst_toolkit.core.data_loaders import load_multi_table
from midst_toolkit.models.clavaddpm.model import clava_clustering, clava_training
from tests.utils.random import set_all_random_seeds, unset_all_random_seeds


CLUSTERING_CONFIG = {
    "parent_scale": 1.0,
    "num_clusters": 3,
    "clustering_method": "both",
}

DIFFUSION_CONFIG = {
    "d_layers": [512, 1024, 1024, 1024, 1024, 512],
    "dropout": 0.0,
    "num_timesteps": 100,
    "model_type": "mlp",
    "iterations": 1000,
    "batch_size": 24,
    "lr": 0.0006,
    "gaussian_loss_type": "mse",
    "weight_decay": 1e-05,
    "scheduler": "cosine",
}

CLASSIFIER_CONFIG = {
    "d_layers": [128, 256, 512, 1024, 512, 256, 128],
    "lr": 0.0001,
    "dim_t": 128,
    "batch_size": 24,
    "iterations": 1000,
}


@pytest.mark.integration_test()
def test_load_single_table():
    tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/single_table/")

    assert list(tables.keys()) == ["trans"]

    assert tables["trans"]["df"].columns.tolist() == [
        "trans_date",
        "trans_type",
        "operation",
        "amount",
        "balance",
        "k_symbol",
        "bank",
        "account",
    ]
    assert tables["trans"]["df"].shape == (99, 8)
    assert tables["trans"]["df"].equals(tables["trans"]["original_df"])
    assert tables["trans"]["df"].columns.tolist() == tables["trans"]["original_cols"]
    with open("tests/integration/data/single_table/trans_domain.json", "r") as f:
        assert tables["trans"]["domain"] == json.load(f)
    assert tables["trans"]["children"] == []
    assert tables["trans"]["parents"] == []
    assert tables["trans"]["info"] == {
        "num_col_idx": [0, 3, 4, 7],
        "cat_col_idx": [1, 2, 5, 6],
        "target_col_idx": [],
        "task_type": "None",
        "column_names": ["trans_date", "trans_type", "operation", "amount", "balance", "k_symbol", "bank", "account"],
        "column_info": {
            0: {},
            1: {},
            2: {},
            3: {},
            4: {},
            5: {},
            6: {},
            7: {},
            "type": "categorical",
            "max": 92881422.0,
            "min": 0.0,
            "categorizes": [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        },
        "train_num": 99,
        "idx_mapping": {0: 0, 1: 4, 2: 5, 3: 1, 4: 2, 5: 6, 6: 7, 7: 3},
        "inverse_idx_mapping": {0: 0, 1: 3, 2: 4, 3: 7, 4: 1, 5: 2, 6: 5, 7: 6},
        "idx_name_mapping": {
            0: "trans_date",
            1: "trans_type",
            2: "operation",
            3: "amount",
            4: "balance",
            5: "k_symbol",
            6: "bank",
            7: "account",
        },
        "metadata": {
            "columns": {
                0: {"sdtype": "numerical", "computer_representation": "Float"},
                1: {"sdtype": "categorical"},
                2: {"sdtype": "categorical"},
                3: {"sdtype": "numerical", "computer_representation": "Float"},
                4: {"sdtype": "numerical", "computer_representation": "Float"},
                5: {"sdtype": "categorical"},
                6: {"sdtype": "categorical"},
                7: {"sdtype": "numerical", "computer_representation": "Float"},
            },
        },
    }

    assert relation_order == [[None, "trans"]]
    assert dataset_meta["relation_order"] == [[None, "trans"]]
    assert dataset_meta["tables"] == {"trans": {"children": [], "parents": []}}


@pytest.mark.integration_test()
def test_load_multi_table():
    tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/multi_table/")

    assert list(tables.keys()) == ["account", "trans"]

    assert tables["account"]["df"].columns.tolist() == ["account_id", "district_id", "frequency", "account_date"]
    assert tables["account"]["df"].shape == (9, 4)
    assert tables["account"]["df"].equals(tables["account"]["original_df"])
    assert tables["account"]["df"].columns.tolist() == tables["account"]["original_cols"]
    with open("tests/integration/data/multi_table/account_domain.json", "r") as f:
        assert tables["account"]["domain"] == json.load(f)
    assert tables["account"]["children"] == ["trans"]
    assert tables["account"]["parents"] == []
    assert tables["account"]["info"] == {
        "num_col_idx": [1],
        "cat_col_idx": [0],
        "target_col_idx": [],
        "task_type": "None",
        "column_names": ["frequency", "account_date"],
        "column_info": {
            0: {},
            1: {},
            "type": "categorical",
            "max": 36.0,
            "min": 2.0,
            "categorizes": [0, 1],
        },
        "train_num": 9,
        "idx_mapping": {0: 1, 1: 0},
        "inverse_idx_mapping": {1: 0, 0: 1},
        "idx_name_mapping": {0: "frequency", 1: "account_date"},
        "metadata": {
            "columns": {
                0: {"sdtype": "categorical"},
                1: {"sdtype": "numerical", "computer_representation": "Float"},
            },
        },
    }

    assert tables["trans"]["df"].columns.tolist() == [
        "trans_id",
        "account_id",
        "trans_date",
        "trans_type",
        "operation",
        "amount",
        "balance",
        "k_symbol",
        "bank",
        "account",
    ]
    assert tables["trans"]["df"].shape == (143, 10)
    assert tables["trans"]["df"].equals(tables["trans"]["original_df"])
    assert tables["trans"]["df"].columns.tolist() == tables["trans"]["original_cols"]
    with open("tests/integration/data/multi_table/trans_domain.json", "r") as f:
        assert tables["trans"]["domain"] == json.load(f)
    assert tables["trans"]["children"] == []
    assert tables["trans"]["parents"] == ["account"]
    assert tables["trans"]["original_cols"] == [
        "trans_id",
        "account_id",
        "trans_date",
        "trans_type",
        "operation",
        "amount",
        "balance",
        "k_symbol",
        "bank",
        "account",
    ]
    assert tables["trans"]["info"] == {
        "num_col_idx": [0, 3, 4, 7],
        "cat_col_idx": [1, 2, 5, 6],
        "target_col_idx": [],
        "task_type": "None",
        "column_names": ["trans_date", "trans_type", "operation", "amount", "balance", "k_symbol", "bank", "account"],
        "column_info": {
            0: {},
            1: {},
            2: {},
            3: {},
            4: {},
            5: {},
            6: {},
            7: {},
            "type": "categorical",
            "max": 95059883.0,
            "min": 0.0,
            "categorizes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        },
        "train_num": 143,
        "idx_mapping": {0: 0, 1: 4, 2: 5, 3: 1, 4: 2, 5: 6, 6: 7, 7: 3},
        "inverse_idx_mapping": {0: 0, 1: 3, 2: 4, 3: 7, 4: 1, 5: 2, 6: 5, 7: 6},
        "idx_name_mapping": {
            0: "trans_date",
            1: "trans_type",
            2: "operation",
            3: "amount",
            4: "balance",
            5: "k_symbol",
            6: "bank",
            7: "account",
        },
        "metadata": {
            "columns": {
                0: {"sdtype": "numerical", "computer_representation": "Float"},
                1: {"sdtype": "categorical"},
                2: {"sdtype": "categorical"},
                3: {"sdtype": "numerical", "computer_representation": "Float"},
                4: {"sdtype": "numerical", "computer_representation": "Float"},
                5: {"sdtype": "categorical"},
                6: {"sdtype": "categorical"},
                7: {"sdtype": "numerical", "computer_representation": "Float"},
            },
        },
    }

    assert relation_order == [["account", "trans"]]
    assert dataset_meta["relation_order"] == [["account", "trans"]]
    assert dataset_meta["tables"] == {
        "account": {"children": ["trans"], "parents": []},
        "trans": {"children": [], "parents": ["account"]},
    }


@pytest.mark.integration_test()
def test_train_single_table(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    os.makedirs(tmp_path / "models")
    configs = {"clustering": CLUSTERING_CONFIG, "diffusion": DIFFUSION_CONFIG}

    # Act
    tables, relation_order, _ = load_multi_table("tests/integration/data/single_table/")
    tables, models = clava_training(tables, relation_order, tmp_path, configs, device="cpu")

    # Assert
    with open(tmp_path / "models" / "None_trans_ckpt.pkl", "rb") as f:
        table_info = pickle.load(f)["table_info"]

    sample_size = 5
    key = (None, "trans")
    x_gen_tensor, y_gen_tensor = models[key]["diffusion"].sample_all(
        sample_size,
        DIFFUSION_CONFIG["batch_size"],
        table_info[key]["empirical_class_dist"].float(),
        ddim=False,
    )
    X_gen, y_gen = x_gen_tensor.numpy(), y_gen_tensor.numpy()

    with open("tests/integration/data/single_table/assertion_data/syntetic_data.json", "r") as f:
        expected_results = json.load(f)

    print(X_gen)
    print(expected_results["X_gen"])
    model_data = dict(models[key]["diffusion"].named_parameters())
    print(model_data)

    Path("tests/integration/data/single_table/assertion_data/diffusion_parameters.pkl").write_bytes(
        pickle.dumps(model_data)
    )

    if _is_apple_silicon():
        # TODO: Figure out if there is a good way of testing the synthetic data results
        # on multiple platforms. https://app.clickup.com/t/868f43wp0
        assert np.allclose(X_gen, expected_results["X_gen"])
        assert np.allclose(y_gen, expected_results["y_gen"])

    unset_all_random_seeds()


@pytest.mark.integration_test()
def test_train_multi_table(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    # Act
    os.makedirs(tmp_path / "models")
    configs = {"clustering": CLUSTERING_CONFIG, "diffusion": DIFFUSION_CONFIG, "classifier": CLASSIFIER_CONFIG}

    tables, relation_order, _ = load_multi_table("tests/integration/data/multi_table/")
    tables, _ = clava_clustering(tables, relation_order, tmp_path, configs)
    models = clava_training(tables, relation_order, tmp_path, configs, device="cpu")

    # Assert
    with open(tmp_path / "models" / "account_trans_ckpt.pkl", "rb") as f:
        table_info = pickle.load(f)["table_info"]

    sample_size = 5
    key = ("account", "trans")
    x_gen_tensor, y_gen_tensor = models[1][key]["diffusion"].sample_all(
        sample_size,
        DIFFUSION_CONFIG["batch_size"],
        table_info[key]["empirical_class_dist"].float(),
        ddim=False,
    )
    X_gen, y_gen = x_gen_tensor.numpy(), y_gen_tensor.numpy()

    with open("tests/integration/data/multi_table/assertion_data/syntetic_data.json", "r") as f:
        expected_results = json.load(f)

    print(X_gen)
    print(expected_results["X_gen"])

    if _is_apple_silicon():
        # TODO: Figure out if there is a good way of testing the synthetic data results
        # on multiple platforms. https://app.clickup.com/t/868f43wp0
        assert np.allclose(X_gen, expected_results["X_gen"])
        assert np.allclose(y_gen, expected_results["y_gen"])

    unset_all_random_seeds()


@pytest.mark.integration_test()
def test_clustering_reload(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    os.makedirs(tmp_path / "models")
    configs = {"clustering": CLUSTERING_CONFIG}

    # Act
    tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/multi_table/")
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, configs)

    # Assert
    account_df_no_clustering = tables["account"]["df"].drop(columns=["account_trans_cluster"])
    account_original_df_as_float = tables["account"]["original_df"].astype(float)
    assert account_df_no_clustering.equals(account_original_df_as_float)

    if _is_apple_silicon():
        # TODO: Figure out if there is a good way of testing the clustering results
        # on multiple platforms. https://app.clickup.com/t/868f43wp0
        with open("tests/integration/data/multi_table/assertion_data/expected_account_clustering.json", "r") as f:
            expected_account_clustering = json.load(f)
        assert tables["account"]["df"]["account_trans_cluster"].tolist() == expected_account_clustering

    trans_df_no_clustering = tables["trans"]["df"].drop(columns=["account_trans_cluster"])
    trans_original_df_as_float = tables["trans"]["original_df"].astype(float)
    trans_original_df_as_float["trans_id"] = trans_original_df_as_float["trans_id"].astype(int)
    assert trans_df_no_clustering.equals(trans_original_df_as_float)

    if _is_apple_silicon():
        with open("tests/integration/data/multi_table/assertion_data/expected_trans_clustering.json", "r") as f:
            expected_trans_clustering = json.load(f)
        assert tables["trans"]["df"]["account_trans_cluster"].tolist() == expected_trans_clustering

    # loading from previously saved clustering
    tables_saved, all_group_lengths_prob_dicts_saved = clava_clustering(tables, relation_order, tmp_path, configs)

    assert all_group_lengths_prob_dicts_saved == all_group_lengths_prob_dicts

    assert tables_saved["account"]["df"].equals(tables["account"]["df"])
    assert tables_saved["account"]["original_df"].equals(tables["account"]["original_df"])
    assert tables_saved["account"]["original_cols"] == tables["account"]["original_cols"]
    assert tables_saved["account"]["domain"] == tables["account"]["domain"]
    assert tables_saved["account"]["children"] == tables["account"]["children"]
    assert tables_saved["account"]["parents"] == tables["account"]["parents"]
    assert tables_saved["account"]["info"] == tables["account"]["info"]

    assert tables_saved["trans"]["df"].equals(tables["trans"]["df"])
    assert tables_saved["trans"]["original_df"].equals(tables["trans"]["original_df"])
    assert tables_saved["trans"]["original_cols"] == tables["trans"]["original_cols"]
    assert tables_saved["trans"]["domain"] == tables["trans"]["domain"]
    assert tables_saved["trans"]["children"] == tables["trans"]["children"]
    assert tables_saved["trans"]["parents"] == tables["trans"]["parents"]
    assert tables_saved["trans"]["info"] == tables["trans"]["info"]

    unset_all_random_seeds()


def _is_apple_silicon():
    """Check if running on macOS with Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"
