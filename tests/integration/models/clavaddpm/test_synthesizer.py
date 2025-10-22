from copy import deepcopy
from pathlib import Path

import pytest

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import load_multi_table
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing
from midst_toolkit.models.clavaddpm.train import clava_training


CLUSTERING_CONFIG = {
    "parent_scale": 1.0,
    "num_clusters": 3,
    "clustering_method": "kmeans_and_gmm",
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
    "data_split_ratios": [0.99, 0.005, 0.005],
}

CLASSIFIER_CONFIG = {
    "d_layers": [128, 256, 512, 1024, 512, 256, 128],
    "lr": 0.0001,
    "dim_t": 128,
    "batch_size": 24,
    "iterations": 1000,
    "data_split_ratios": [0.99, 0.005, 0.005],
}

SYNTHESIZING_CONFIG = {
    "general": {
        "exp_name": "ensemble_attack",
        "workspace_dir": None,
        "sample_prefix": "",
    },
    "sampling": {
        "batch_size": 2,
        "classifier_scale": 1.0,
    },
    "matching": {
        "num_matching_clusters": 1,
        "matching_batch_size": 1,
        "unique_matching": True,
        "no_matching": False,
    },
}


@pytest.mark.integration_test()
def test_clava_syntheesize_multi_table(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    # Act
    tables, relation_order, _ = load_multi_table(Path("tests/integration/assets/multi_table/"))
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, CLUSTERING_CONFIG)
    models = clava_training(tables, relation_order, tmp_path, DIFFUSION_CONFIG, CLASSIFIER_CONFIG, device=DEVICE)

    # TODO: Temporary, we should refactor those configs
    configs = deepcopy(SYNTHESIZING_CONFIG)
    configs["general"]["workspace_dir"] = str(tmp_path)

    cleaned_tables, _, _ = clava_synthesizing(
        tables,
        relation_order,
        tmp_path,
        all_group_lengths_prob_dicts,
        models[1],
        configs,
    )

    unset_all_random_seeds()
