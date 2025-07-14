import os
import shutil

from midst_toolkit.core.data_loaders import load_multi_table
from midst_toolkit.models.tabddpm.model import clava_clustering


def test_train():
    save_dir = "tests/unit/models/tabddpm/results/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    tables, relation_order, dataset_meta = load_multi_table("tests/unit/data/tabddpm/")

    configs = {
        "clustering": {
            "parent_scale": 1.0,
            "num_clusters": 50,
            "clustering_method": "both",
        },
    }

    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)

    assert tables
