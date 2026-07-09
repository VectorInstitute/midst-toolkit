import json
import pickle
import random
from collections.abc import Callable
from logging import WARNING
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional

from midst_toolkit.common.config import ClavaDDPMClassifierConfig, ClavaDDPMClusteringConfig, ClavaDDPMDiffusionConfig
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import (
    NO_PARENT_COLUMN_NAME,
    CategoricalColumnInfo,
    ColumnInfo,
    ColumnMetadata,
    ColumnType,
    ComputerRepresentation,
    DomainInfo,
    NumericalColumnInfo,
    load_tables,
)
from midst_toolkit.models.clavaddpm.enumerations import ClusteringMethod
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import GaussianLossType, SchedulerType
from midst_toolkit.models.clavaddpm.model import Classifier, ModelType
from midst_toolkit.models.clavaddpm.train import clava_training
from tests.integration.utils import is_running_on_ci_environment


CLUSTERING_CONFIG = ClavaDDPMClusteringConfig(
    parent_scale=1.0,
    num_clusters=3,
    clustering_method=ClusteringMethod.KMEANS_AND_GMM,
)

DIFFUSION_CONFIG = ClavaDDPMDiffusionConfig(
    d_layers=[512, 1024, 1024, 1024, 1024, 512],
    dropout=0.0,
    num_timesteps=100,
    model_type=ModelType.MLP,
    iterations=1000,
    batch_size=24,
    lr=0.0006,
    gaussian_loss_type=GaussianLossType.MSE,
    weight_decay=1e-05,
    scheduler=SchedulerType.COSINE,
    data_split_ratios=[0.99, 0.005, 0.005],
)

CLASSIFIER_CONFIG = ClavaDDPMClassifierConfig(
    d_layers=[128, 256, 512, 1024, 512, 256, 128],
    lr=0.0001,
    dim_t=128,
    batch_size=24,
    iterations=1000,
    data_split_ratios=[0.99, 0.005, 0.005],
)


@pytest.mark.integration_test()
def test_load_single_table():
    tables, relation_order, dataset_meta = load_tables(Path("tests/integration/assets/single_table/"))

    assert list(tables.keys()) == ["trans"]

    assert tables["trans"].data.columns.tolist() == [
        "trans_date",
        "trans_type",
        "operation",
        "amount",
        "balance",
        "k_symbol",
        "bank",
        "account",
    ]
    assert tables["trans"].data.shape == (99, 8)
    assert tables["trans"].data.equals(tables["trans"].original_data)
    assert tables["trans"].data.columns.tolist() == tables["trans"].original_column_names
    with open("tests/integration/assets/single_table/trans_domain.json", "r") as f:
        assert tables["trans"].domain == json.load(f)
    assert tables["trans"].children == []
    assert tables["trans"].parents == []
    assert tables["trans"].info == DomainInfo(
        numerical_column_indices=[0, 3, 4, 7],
        categorical_column_indices=[1, 2, 5, 6],
        target_column_indices=[],
        task_type=None,
        column_names=["trans_date", "trans_type", "operation", "amount", "balance", "k_symbol", "bank", "account"],
        columns_info={
            "trans_date": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=2166.0, min=280.0)),
            "trans_type": ColumnInfo(type=ColumnType.CATEGORICAL, info=CategoricalColumnInfo(categorizes=[0, 1, 2])),
            "operation": ColumnInfo(
                type=ColumnType.CATEGORICAL, info=CategoricalColumnInfo(categorizes=[0, 1, 2, 3, 4])
            ),
            "amount": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=45715.0, min=14.6)),
            "balance": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=140228.7, min=7704.0)),
            "k_symbol": ColumnInfo(
                type=ColumnType.CATEGORICAL, info=CategoricalColumnInfo(categorizes=[0, 1, 2, 3, 5, 6, 7, 8])
            ),
            "bank": ColumnInfo(
                type=ColumnType.CATEGORICAL,
                info=CategoricalColumnInfo(categorizes=[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            ),
            "account": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=92881422.0, min=0.0)),
        },
        train_num=99,
        metadata={
            0: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
            1: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            2: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            3: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
            4: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
            5: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            6: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            7: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
        },
    )

    assert relation_order == [(None, "trans")]
    assert dataset_meta["relation_order"] == [[None, "trans"]]
    assert dataset_meta["tables"] == {"trans": {"children": [], "parents": []}}


@pytest.mark.integration_test()
def test_load_tables():
    tables, relation_order, dataset_meta = load_tables(Path("tests/integration/assets/multi_table/"))

    assert list(tables.keys()) == ["account", "trans"]

    assert tables["account"].data.columns.tolist() == ["account_id", "district_id", "frequency", "account_date"]
    assert tables["account"].data.shape == (9, 4)
    assert tables["account"].data.equals(tables["account"].original_data)
    assert tables["account"].data.columns.tolist() == tables["account"].original_column_names
    with open("tests/integration/assets/multi_table/account_domain.json", "r") as f:
        assert tables["account"].domain == json.load(f)
    assert tables["account"].children == ["trans"]
    assert tables["account"].parents == []
    assert tables["account"].info == DomainInfo(
        numerical_column_indices=[1],
        categorical_column_indices=[0],
        target_column_indices=[],
        task_type=None,
        column_names=["frequency", "account_date"],
        columns_info={
            "frequency": ColumnInfo(type=ColumnType.CATEGORICAL, info=CategoricalColumnInfo(categorizes=[0, 1])),
            "account_date": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=36.0, min=2.0)),
        },
        train_num=9,
        metadata={
            0: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            1: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
        },
    )

    assert tables["trans"].data.columns.tolist() == [
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
    assert tables["trans"].data.shape == (143, 10)
    assert tables["trans"].data.equals(tables["trans"].original_data)
    assert tables["trans"].data.columns.tolist() == tables["trans"].original_column_names
    with open("tests/integration/assets/multi_table/trans_domain.json", "r") as f:
        assert tables["trans"].domain == json.load(f)
    assert tables["trans"].children == []
    assert tables["trans"].parents == ["account"]
    assert tables["trans"].original_column_names == [
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
    assert tables["trans"].info == DomainInfo(
        numerical_column_indices=[0, 3, 4, 7],
        categorical_column_indices=[1, 2, 5, 6],
        target_column_indices=[],
        task_type=None,
        column_names=["trans_date", "trans_type", "operation", "amount", "balance", "k_symbol", "bank", "account"],
        columns_info={
            "trans_date": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=2169.0, min=58.0)),
            "trans_type": ColumnInfo(type=ColumnType.CATEGORICAL, info=CategoricalColumnInfo(categorizes=[0, 1, 2])),
            "operation": ColumnInfo(
                type=ColumnType.CATEGORICAL, info=CategoricalColumnInfo(categorizes=[0, 1, 2, 3, 4])
            ),
            "amount": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=49764.0, min=14.6)),
            "balance": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=98605.1, min=14750.5)),
            "k_symbol": ColumnInfo(
                type=ColumnType.CATEGORICAL, info=CategoricalColumnInfo(categorizes=[0, 1, 3, 5, 6, 7])
            ),
            "bank": ColumnInfo(
                type=ColumnType.CATEGORICAL,
                info=CategoricalColumnInfo(categorizes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            ),
            "account": ColumnInfo(type=ColumnType.NUMERICAL, info=NumericalColumnInfo(max=95059883.0, min=0.0)),
        },
        train_num=143,
        metadata={
            0: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
            1: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            2: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            3: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
            4: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
            5: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            6: ColumnMetadata(sdtype=ColumnType.CATEGORICAL),
            7: ColumnMetadata(sdtype=ColumnType.NUMERICAL, computer_representation=ComputerRepresentation.FLOAT),
        },
    )

    assert relation_order == [(None, "account"), ("account", "trans")]
    assert dataset_meta["relation_order"] == [[None, "account"], ["account", "trans"]]
    assert dataset_meta["tables"] == {
        "account": {"children": ["trans"], "parents": []},
        "trans": {"children": [], "parents": ["account"]},
    }


@pytest.mark.integration_test()
def test_train_single_table(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    # Act
    tables, relation_order, _ = load_tables(Path("tests/integration/assets/single_table/"))
    tables, models = clava_training(tables, relation_order, tmp_path, DIFFUSION_CONFIG, device=DEVICE)

    # Assert
    with open(tmp_path / "models" / "None_trans_ckpt.pkl", "rb") as f:
        table_info = pickle.load(f).table_info

    sample_size = 5
    key = (None, "trans")
    x_gen_tensor, y_gen_tensor = models[key].diffusion.sample_all(
        sample_size,
        DIFFUSION_CONFIG.batch_size,
        table_info[key]["empirical_class_dist"].float(),
        ddim=False,
    )
    x_gen, y_gen = x_gen_tensor.numpy(), y_gen_tensor.numpy()

    with open("tests/integration/assets/single_table/assertion_data/synthetic_data.json", "r") as f:
        expected_results = json.load(f)

    model_data = dict(models[key].diffusion.named_parameters())

    expected_model_data = pickle.loads(
        Path("tests/integration/assets/single_table/assertion_data/diffusion_parameters.pkl").read_bytes(),
    )
    # Making sure the expected model data is loaded on the correct device
    expected_model_data = {layer: data.to(DEVICE) for layer, data in expected_model_data.items()}

    model_layers = list(model_data.keys())
    # Adding those asserts under an if condition because they only pass on github.
    # In the else block, we set a tolerance that would work across platforms
    # however, it is way too high of a tolerance.
    if is_running_on_ci_environment():
        # if the first layer is equal with minimal tolerance, all others should be equal as well
        assert all(torch.allclose(model_data[layer], expected_model_data[layer]) for layer in model_layers)

        # TODO: Figure out if there is a good way of testing the synthetic data results
        # on multiple platforms. https://app.clickup.com/t/868f43wp0
        assert np.allclose(x_gen, expected_results["X_gen"])
        assert np.allclose(y_gen, expected_results["y_gen"])

    else:
        # Otherwise, set a tolerance that would work across platforms
        # TODO: Figure out a way to set a lower tolerance
        # https://app.clickup.com/t/868f43wp0
        log(WARNING, "Not running on CI, assertions are made with a higher tolerance.")
        assert all(torch.allclose(model_data[layer], expected_model_data[layer], atol=0.1) for layer in model_layers)

    unset_all_random_seeds()


@pytest.mark.integration_test()
def test_train_multi_table(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    # Act
    tables, relation_order, _ = load_tables(Path("tests/integration/assets/multi_table/"))
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, CLUSTERING_CONFIG)
    models = clava_training(tables, relation_order, tmp_path, DIFFUSION_CONFIG, CLASSIFIER_CONFIG, device=DEVICE)

    # Assert
    with open(tmp_path / "models" / "account_trans_ckpt.pkl", "rb") as f:
        table_info = pickle.load(f).table_info

    sample_size = 5
    key = ("account", "trans")
    x_gen_tensor, y_gen_tensor = models[1][key].diffusion.sample_all(
        sample_size,
        DIFFUSION_CONFIG.batch_size,
        table_info[key]["empirical_class_dist"].float(),
        ddim=False,
    )
    x_gen, y_gen = x_gen_tensor.numpy(), y_gen_tensor.numpy()

    with open("tests/integration/assets/multi_table/assertion_data/synthetic_data.json", "r") as f:
        expected_results = json.load(f)

    model_data = dict(models[1][key].diffusion.named_parameters())

    expected_model_data = pickle.loads(
        Path("tests/integration/assets/multi_table/assertion_data/diffusion_parameters.pkl").read_bytes(),
    )
    # Making sure the expected model data is loaded on the correct device
    expected_model_data = {layer: data.to(DEVICE) for layer, data in expected_model_data.items()}

    # Adding those asserts under an if condition because they only pass on github.
    # In the else block, we set a tolerance that would work across platforms
    # however, it is way too high of a tolerance.
    model_layers = list(model_data.keys())
    if is_running_on_ci_environment():
        # if the first layer is equal with minimal tolerance, all others should be equal as well
        assert all(torch.allclose(model_data[layer], expected_model_data[layer]) for layer in model_layers)

        # TODO: Figure out if there is a good way of testing the synthetic data results
        # on multiple platforms. https://app.clickup.com/t/868f43wp0
        assert np.allclose(x_gen, expected_results["X_gen"])
        assert np.allclose(y_gen, expected_results["y_gen"])

    else:
        # Otherwise, set a tolerance that would work across platforms
        # TODO: Figure out a way to set a lower tolerance
        # https://app.clickup.com/t/868f43wp0
        log(WARNING, "Not running on CI, assertions are made with a higher tolerance.")
        assert all(torch.allclose(model_data[layer], expected_model_data[layer], atol=0.1) for layer in model_layers)

    classifier_scale = 1.0
    classifier_batch_size = 5
    # Generating some random data to test the classifier
    groups = list(all_group_lengths_prob_dicts[key].keys())
    ys = [[y] for y in random.choices(groups, k=classifier_batch_size)]

    ys_tensor = torch.tensor(np.array(ys).reshape(-1, 1), requires_grad=False)
    conditional_sample, _ = models[1][key].diffusion.conditional_sample(
        targets=ys_tensor,
        model_kwargs={"y": ys_tensor},
        conditioning_function=get_conditioning_function_for_diffusion(models[1][key].classifier, classifier_scale),
    )

    expected_conditional_sample = torch.load(
        "tests/integration/assets/multi_table/assertion_data/conditional_samples.pt"
    ).to(DEVICE)

    # Adding those asserts under an if condition because they only pass on github.
    if is_running_on_ci_environment():
        # if the first values are equal with minimal tolerance, all others should be equal as well
        assert torch.allclose(conditional_sample, expected_conditional_sample)
    else:
        log(WARNING, "Not running on CI, skipping detailed assertions.")

    unset_all_random_seeds()


@pytest.mark.integration_test()
def test_clustering_reload(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    # Act
    tables, relation_order, _ = load_tables(Path("tests/integration/assets/multi_table/"))
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, CLUSTERING_CONFIG)

    # Assert
    account_df_no_clustering = tables["account"].data.drop(columns=["account_trans_cluster", NO_PARENT_COLUMN_NAME])
    account_original_df_as_float = tables["account"].original_data.astype(float)
    assert account_df_no_clustering.equals(account_original_df_as_float)

    account_assertion_file_name = "expected_account_clustering.json"
    trans_assertion_file_name = "expected_trans_clustering.json"
    if is_running_on_ci_environment():
        # TODO: Figure out if there is a good way of testing the synthetic data results
        # on multiple platforms. https://app.clickup.com/t/868f43wp0
        account_assertion_file_name = "expected_account_clustering_remote.json"
        trans_assertion_file_name = "expected_trans_clustering_remote.json"

    with open(f"tests/integration/assets/multi_table/assertion_data/{account_assertion_file_name}", "r") as f:
        expected_account_clustering = json.load(f)
    assert tables["account"].data["account_trans_cluster"].tolist() == expected_account_clustering

    trans_df_no_clustering = tables["trans"].data.drop(columns=["account_trans_cluster"])
    trans_original_df_as_float = tables["trans"].original_data.astype(float)
    trans_original_df_as_float["trans_id"] = trans_original_df_as_float["trans_id"].astype(int)
    assert trans_df_no_clustering.equals(trans_original_df_as_float)

    with open(f"tests/integration/assets/multi_table/assertion_data/{trans_assertion_file_name}", "r") as f:
        expected_trans_clustering = json.load(f)
    assert tables["trans"].data["account_trans_cluster"].tolist() == expected_trans_clustering

    # loading from previously saved clustering
    tables_saved, all_group_lengths_prob_dicts_saved = clava_clustering(
        tables, relation_order, tmp_path, CLUSTERING_CONFIG
    )

    assert all_group_lengths_prob_dicts_saved == all_group_lengths_prob_dicts

    assert tables_saved["account"].data.equals(tables["account"].data)
    assert tables_saved["account"].original_data.equals(tables["account"].original_data)
    assert tables_saved["account"].original_column_names == tables["account"].original_column_names
    assert tables_saved["account"].domain == tables["account"].domain
    assert tables_saved["account"].children == tables["account"].children
    assert tables_saved["account"].parents == tables["account"].parents
    assert tables_saved["account"].info == tables["account"].info

    assert tables_saved["trans"].data.equals(tables["trans"].data)
    assert tables_saved["trans"].original_data.equals(tables["trans"].original_data)
    assert tables_saved["trans"].original_column_names == tables["trans"].original_column_names
    assert tables_saved["trans"].domain == tables["trans"].domain
    assert tables_saved["trans"].children == tables["trans"].children
    assert tables_saved["trans"].parents == tables["trans"].parents
    assert tables_saved["trans"].info == tables["trans"].info

    unset_all_random_seeds()


def get_conditioning_function_for_diffusion(classifier: Classifier, classifier_scale: float) -> Callable:
    def conditioning_function(
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        remove_first_col: bool = False,
    ) -> torch.Tensor:
        assert y is not None
        with torch.enable_grad():
            if remove_first_col:
                x_in = x[:, 1:].detach().requires_grad_(True).float()
            else:
                x_in = x.detach().requires_grad_(True).float()
            logits = classifier(x_in, t)
            log_probs = functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    return conditioning_function
