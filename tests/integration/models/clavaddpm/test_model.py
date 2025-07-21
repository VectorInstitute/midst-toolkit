import os
import tempfile

import pytest

from midst_toolkit.core.data_loaders import load_multi_table
from midst_toolkit.models.clavaddpm.model import clava_clustering, clava_training


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
def test_train_single_table():
    with tempfile.TemporaryDirectory() as save_dir:
        os.makedirs(os.path.join(save_dir, "models"))

        configs = {"clustering": CLUSTERING_CONFIG, "diffusion": DIFFUSION_CONFIG}

        tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/single_table/")
        models = clava_training(tables, relation_order, save_dir, configs, device="cpu")

        assert models


@pytest.mark.integration_test()
def test_train_multi_table():
    with tempfile.TemporaryDirectory() as save_dir:
        os.makedirs(os.path.join(save_dir, "models"))

        configs = {"clustering": CLUSTERING_CONFIG, "diffusion": DIFFUSION_CONFIG, "classifier": CLASSIFIER_CONFIG}

        tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/multi_table/")
        tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)
        models = clava_training(tables, relation_order, save_dir, configs, device="cpu")

        assert models


@pytest.mark.integration_test()
def test_clustering_reload():
    with tempfile.TemporaryDirectory() as save_dir:
        configs = {"clustering": CLUSTERING_CONFIG}

        tables, relation_order, dataset_meta = load_multi_table("tests/integration/data/multi_table/")
        tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)

        assert all_group_lengths_prob_dicts

        # loading from previously saved clustering
        tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)

        assert all_group_lengths_prob_dicts
