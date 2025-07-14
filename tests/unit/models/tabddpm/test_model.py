import os
import shutil

from midst_toolkit.core.data_loaders import load_multi_table
from midst_toolkit.models.tabddpm.model import clava_clustering, clava_training


def test_train():
    save_dir = "tests/unit/models/tabddpm/results/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    configs = {
        "clustering": {
            "parent_scale": 1.0,
            "num_clusters": 50,
            "clustering_method": "both",
        },
        "diffusion": {
            "d_layers": [
                512,
                1024,
                1024,
                1024,
                1024,
                512,
            ],
            "dropout": 0.0,
            "num_timesteps": 2000,
            "model_type": "mlp",
            "iterations": 200000,
            "batch_size": 4096,
            "lr": 0.0006,
            "gaussian_loss_type": "mse",
            "weight_decay": 1e-05,
            "scheduler": "cosine",
        },
    }

    tables, relation_order, dataset_meta = load_multi_table("tests/unit/data/tabddpm/")

    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)

    models = clava_training(tables, relation_order, save_dir, configs, device="cpu")

    assert models
