from pathlib import Path

import pytest

from midst_toolkit.core.data_loaders import load_multi_table
from midst_toolkit.models.clavaddpm.train import clava_clustering, clava_training


CLUSTERING_CONFIG = {
    "parent_scale": 1.0,
    "num_clusters": 50,
    "clustering_method": "both",
}

DIFFUSION_CONFIG = {
    "d_layers": [512, 1024, 1024, 1024, 1024, 512],
    "dropout": 0.0,
    "num_timesteps": 2000,
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
def test_train_single_table(tmp_path: Path):
    tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/single_table/")
    models = clava_training(tables, relation_order, tmp_path, DIFFUSION_CONFIG, {}, device="cpu")

    assert models


@pytest.mark.integration_test()
def test_train_multi_table(tmp_path: Path):
    tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/multi_table/")
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, CLUSTERING_CONFIG)
    models = clava_training(tables, relation_order, tmp_path, DIFFUSION_CONFIG, None, device="cpu")

    assert models


@pytest.mark.integration_test()
def test_clustering_reload(tmp_path: Path):
    tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/multi_table/")
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, CLUSTERING_CONFIG)

    assert all_group_lengths_prob_dicts

    # loading from previously saved clustering
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, CLUSTERING_CONFIG)

    assert all_group_lengths_prob_dicts
