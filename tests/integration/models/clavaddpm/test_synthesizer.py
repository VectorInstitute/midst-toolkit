import pickle
from logging import WARNING
from pathlib import Path

import pytest

from midst_toolkit.common.config import (
    ClavaDDPMClassifierConfig,
    ClavaDDPMClusteringConfig,
    ClavaDDPMDiffusionConfig,
    ClavaDDPMMatchingConfig,
    ClavaDDPMSamplingConfig,
    GeneralConfig,
)
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import load_tables
from midst_toolkit.models.clavaddpm.enumerations import ClusteringMethod
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import GaussianLossType, SchedulerType
from midst_toolkit.models.clavaddpm.model import ModelType
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing
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

GENERAL_CONFIG = GeneralConfig(
    data_dir=Path("tests/integration/assets/multi_table/"),
    test_data_dir=Path("tests/integration/assets/multi_table/"),
    exp_name="ensemble_attack",
    workspace_dir=Path("temp/workspace/dir"),
    sample_prefix="",
)

SAMPLING_CONFIG = ClavaDDPMSamplingConfig(
    batch_size=2,
    classifier_scale=1.0,
)

MATCHING_CONFIG = ClavaDDPMMatchingConfig(
    num_matching_clusters=1,
    matching_batch_size=1,
    unique_matching=True,
    no_matching=False,
)


@pytest.mark.integration_test()
def test_clava_synthesize_multi_table(tmp_path: Path):
    # Setup
    set_all_random_seeds(seed=133742, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    # Act
    tables, relation_order, _ = load_tables(Path("tests/integration/assets/multi_table/"))
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, tmp_path, CLUSTERING_CONFIG)
    models = clava_training(tables, relation_order, tmp_path, DIFFUSION_CONFIG, CLASSIFIER_CONFIG, device=DEVICE)

    synthesizing_config = GENERAL_CONFIG.model_copy()
    synthesizing_config.workspace_dir = tmp_path

    cleaned_tables, _, _ = clava_synthesizing(
        tables,
        relation_order,
        tmp_path,
        models[1],
        synthesizing_config,
        SAMPLING_CONFIG,
        MATCHING_CONFIG,
        all_group_lengths_prob_dicts,
    )

    # Assert
    assert cleaned_tables["account"].shape == (9, 2)
    assert cleaned_tables["trans"].shape == (145, 8)

    if is_running_on_ci_environment():
        expected_cleaned_tables = pickle.loads(
            Path("tests/integration/assets/multi_table/assertion_data/cleaned_tables.pkl").read_bytes(),
        )
        assert cleaned_tables["account"].equals(expected_cleaned_tables["account"])
        assert cleaned_tables["trans"].equals(expected_cleaned_tables["trans"])

    else:
        log(WARNING, "Not running on CI, skipping detailed assertions.")

    unset_all_random_seeds()
