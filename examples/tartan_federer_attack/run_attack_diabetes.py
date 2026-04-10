"""
run_attack_diabetes.py
======================
Diabetes-specific version of run_attack.py.

Patches make_dataset_from_df_with_loaded() with global vocabulary expansion
so there are zero unseen label warnings for diag_1/2/3 and other
split-inconsistent columns.
"""

from __future__ import annotations

import json
import os
from logging import INFO
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder, StandardScaler

from midst_toolkit.attacks.tartan_federer.tartan_federer_attack import tartan_federer_attack
import midst_toolkit.attacks.tartan_federer.tartan_federer_attack as _tfa_module
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.models.clavaddpm.dataset import (
    Dataset,
    TableMetadata,
    Transformations,
    TargetInfo,
    transform_dataset,
)
from midst_toolkit.models.clavaddpm.dataset import get_categorical_and_numerical_column_names
from midst_toolkit.models.clavaddpm.enumerations import IsTargetConditioned
from dataclasses import replace as dc_replace


# ─────────────────────────────────────────────────────────────────────────────
# Global vocab sizes — true global sizes across the full 101k dataset
# Used to expand label encoders that were fitted on a single 9975-row split
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_VOCAB_SIZES = {
    # col_name: global_size (max integer value = global_size - 1)
    "diag_1": 916,
    "diag_2": 916,
    "diag_3": 916,
    "admission_type": 8,
    "discharge_disposition": 26,
    "admission_source": 17,
    "nateglinide": 4,
    "chlorpropamide": 4,
    "tolbutamide": 2,
    "acarbose": 4,
    "miglitol": 4,
    "tolazamide": 3,
    "glyburide-metformin": 4,
    "gender": 2,
}


def _expand_encoder(encoder: LabelEncoder, col_name: str) -> bool:
    """
    Expand a LabelEncoder's classes_ to cover the full global vocabulary.
    Returns True if expansion was needed.
    """
    if col_name not in GLOBAL_VOCAB_SIZES:
        return False
    global_size = GLOBAL_VOCAB_SIZES[col_name]
    current_size = len(encoder.classes_)
    if current_size >= global_size:
        return False
    existing = set(encoder.classes_.astype(int))
    missing = [i for i in range(global_size) if i not in existing]
    if missing:
        new_classes = np.sort(
            np.concatenate([encoder.classes_.astype(int), np.array(missing)])
        )
        encoder.classes_ = new_classes.astype(encoder.classes_.dtype)
    return True


def _safe_label_encode(encoder: LabelEncoder, values: np.ndarray) -> np.ndarray:
    known = set(encoder.classes_)
    fallback = encoder.classes_[0]
    safe_values = np.where(np.isin(values, list(known)), values, fallback)
    return encoder.transform(safe_values).astype(float)


def make_dataset_from_df_with_loaded_diabetes(
    data: pd.DataFrame,
    transformation: Transformations,
    is_target_conditioned: IsTargetConditioned,
    table_metadata: TableMetadata,
    label_encoders: dict[int, LabelEncoder],
    numerical_transform: StandardScaler | None = None,
    noise_scale: float = 0,
) -> Dataset:
    categorical_column_names, numerical_column_names = get_categorical_and_numerical_column_names(
        table_metadata,
        is_target_conditioned,
    )
    numerical_features = {DataSplit.TRAIN.value: data[numerical_column_names].values.astype(np.float32)}
    categorical_features = {DataSplit.TRAIN.value: data[categorical_column_names].to_numpy(dtype=np.str_)}
    targets = {DataSplit.TRAIN.value: data[[table_metadata.target_column_name]].values.astype(np.float32)}

    if len(categorical_column_names) > 0:
        all_categorical_features = categorical_features[DataSplit.TRAIN.value]
        encoded_categorical_features = []
        for column_index in range(all_categorical_features.shape[1]):
            col_values = all_categorical_features[:, column_index]
            encoder = label_encoders[column_index]
            col_name = categorical_column_names[column_index]

            # Expand encoder to global vocabulary if needed
            _expand_encoder(encoder, col_name)

            # Check for any remaining unseen values
            known = set(encoder.classes_)
            unseen = set(col_values) - known
            if unseen:
                log(INFO, f"  [WARNING] col {column_index} ({col_name}): "
                    f"{len(unseen)} unseen labels still remain after expansion")

            encoded_column = _safe_label_encode(encoder, col_values)

            if noise_scale > 0:
                encoded_column += np.random.normal(0, noise_scale, encoded_column.shape)
            encoded_categorical_features.append(encoded_column)

        categorical_features[DataSplit.TRAIN.value] = np.vstack(encoded_categorical_features).T

    if len(numerical_column_names) >= 0:
        numerical_features[DataSplit.TRAIN.value] = np.concatenate(
            (numerical_features[DataSplit.TRAIN.value], categorical_features[DataSplit.TRAIN.value]), axis=1
        )
    else:
        numerical_features = categorical_features

    target_info = TargetInfo(policy=None, mean=None, std=None)

    if numerical_transform is not None:
        numerical_features = {k: numerical_transform.transform(v) for k, v in numerical_features.items()}

    dataset = Dataset(
        numerical_features=numerical_features,
        categorical_features=None,
        target=targets,
        target_info=target_info,
        task_type=table_metadata.task_type,
        n_classes=table_metadata.n_classes,
        categorical_transform=None,
        numerical_transform=numerical_transform,
    )

    transformation_no_norm = dc_replace(transformation, normalization=None)
    return transform_dataset(dataset, transformation_no_norm, None)


_tfa_module.make_dataset_from_df_with_loaded = make_dataset_from_df_with_loaded_diabetes


# ─────────────────────────────────────────────────────────────────────────────
# Population data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_population_dataset_for_attack(
    model_indices: list[int],
    model_type: str,
    models_base_dir: Path,
    columns_for_deduplication: list[str],
    all_model_indices: list[int] | None = None,
) -> pd.DataFrame:
    if len(model_indices) == 0:
        raise ValueError("The 'indices' list is empty.")

    df_merge_list = []
    for model_index in model_indices:
        base_path = models_base_dir / f"{model_type}_{model_index}"
        df_merge_list.append(pd.read_csv(os.path.join(base_path, "train_with_id.csv")))

    challenge_source_indices = all_model_indices if all_model_indices is not None else model_indices
    df_challenge_list = []
    for model_index in challenge_source_indices:
        base_path = models_base_dir / f"{model_type}_{model_index}"
        challenge_path = os.path.join(base_path, "challenge_with_id.csv")
        if os.path.exists(challenge_path):
            df_challenge_list.append(pd.read_csv(challenge_path))

    df_merge = pd.concat(df_merge_list, ignore_index=True)
    df_challenge = pd.concat(df_challenge_list, ignore_index=True)
    df_merge = df_merge.drop_duplicates(subset=columns_for_deduplication)
    df_challenge = df_challenge.drop_duplicates(subset=columns_for_deduplication)

    return df_merge[
        ~df_merge.set_index(columns_for_deduplication).index.isin(
            df_challenge.set_index(columns_for_deduplication).index
        )
    ]


def run_data_processing(config: dict[str, Any]) -> None:
    log(INFO, "Running data processing pipeline...")

    population_data_path = Path(config["data_paths"]["population_data_path"])
    midst_data_path = Path(config["data_paths"]["midst_data_path"])
    population_data_path.mkdir(parents=True, exist_ok=True)

    attack_cfg = config["attack_config"]
    all_model_indices = list(
        set(attack_cfg["train_indices"])
        | set(attack_cfg["val_indices"])
        | set(attack_cfg["test_indices"])
    )

    population_data_for_training_attack = prepare_population_dataset_for_attack(
        model_indices=config["data_processing_config"]["population_attack_indices_to_collect_for_training"],
        model_type=config["data_processing_config"]["model_type"],
        models_base_dir=midst_data_path,
        columns_for_deduplication=config["data_processing_config"]["columns_for_deduplication"],
        all_model_indices=all_model_indices,
    )
    population_data_for_training_attack.to_csv(
        population_data_path / "population_dataset_for_training_attack.csv", index=False,
    )

    population_data_for_validating_attack = prepare_population_dataset_for_attack(
        model_indices=config["data_processing_config"]["population_attack_indices_to_collect_for_validation"],
        model_type=config["data_processing_config"]["model_type"],
        models_base_dir=midst_data_path,
        columns_for_deduplication=config["data_processing_config"]["columns_for_deduplication"],
        all_model_indices=all_model_indices,
    )
    population_data_for_validating_attack.to_csv(
        population_data_path / "population_dataset_for_validating_attack.csv", index=False,
    )

    log(INFO, "Data processing pipeline finished.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="experiment_config_diabetes", version_base=None)
def run_attack(config: DictConfig) -> None:
    log(INFO, "Running Tartan-Federer attack (diabetes)...")

    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    cfg = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))

    if config["pipeline"]["run_data_processing"]:
        run_data_processing(cfg)

    data_cfg = cfg["data_paths"]
    attack_cfg = cfg["attack_config"]
    classifier_cfg = cfg["classifier_config"]

    _mia_performance_train, _mia_performance_val, _mia_performance_test = tartan_federer_attack(
        train_indices=attack_cfg["train_indices"],
        val_indices=attack_cfg["val_indices"],
        test_indices=attack_cfg["test_indices"],
        columns_for_deduplication=attack_cfg["columns_for_deduplication"],
        timesteps=attack_cfg["timesteps"],
        additional_timesteps=attack_cfg["additional_timesteps"],
        num_noise_per_time_step=attack_cfg["num_noise_per_time_step"],
        samples_per_train_model=attack_cfg["samples_per_train_model"],
        samples_per_val_model=attack_cfg["samples_per_val_model"],
        classifier_num_epochs=classifier_cfg["num_epochs"],
        classifier_hidden_dim=classifier_cfg["hidden_dim"],
        classifier_learning_rate=classifier_cfg["learning_rate"],
        model_type=attack_cfg["model_type"],
        predictions_file_name=attack_cfg["predictions_file_name"],
        population_data_dir=Path(data_cfg["population_data_path"]),
        model_data_dir=Path(config["data_paths"]["midst_data_path"]),
        meta_dir=Path(config["data_paths"]["metadata_dir"]),
        target_model_subdir=Path(attack_cfg["target_shadow_model_subdir"]),
        results_path=Path(attack_cfg["results_path"]),
    )

    unset_all_random_seeds()
    log(INFO, "Diabetes attack finished successfully.")


if __name__ == "__main__":
    run_attack()