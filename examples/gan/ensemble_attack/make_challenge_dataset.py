from logging import INFO
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from examples.gan.utils import get_table_name
from midst_toolkit.common.logger import log


@hydra.main(config_path="./", config_name="config", version_base=None)
def make_challenge_dataset(config: DictConfig) -> None:
    """Main function to make the challenge dataset."""
    log(INFO, "Making challenge dataset...")

    if config.training.data_path is None:
        dataset_name = get_table_name(config.base_data_dir)
        real_data = pd.read_csv(Path(config.base_data_dir) / f"{dataset_name}.csv")
    else:
        dataset_name = Path(config.training.data_path).stem
        real_data = pd.read_csv(config.training.data_path)

    random_seed = config.ensemble_attack.random_seed

    training_data = pd.read_csv(Path(config.results_dir) / f"{dataset_name}_sampled.csv")
    id_column = config.ensemble_attack.table_id_column_name
    untrained_data = real_data[~real_data[id_column].isin(training_data[id_column])]
    sampled_untrained_data = untrained_data.sample(len(training_data), random_state=random_seed)

    challenge_data = pd.concat([training_data, sampled_untrained_data])
    challenge_data_labels = np.concatenate([np.ones(len(training_data)), np.zeros(len(sampled_untrained_data))])

    processed_attack_data_path = Path(config.ensemble_attack.data_paths.processed_attack_data_path)
    processed_attack_data_path.mkdir(parents=True, exist_ok=True)

    challenge_data_path = processed_attack_data_path / f"{dataset_name}_challenge_data.csv"
    challenge_label_path = processed_attack_data_path / f"{dataset_name}_challenge_labels.npy"
    log(INFO, f"Saving challenge data to {challenge_data_path}")
    challenge_data.to_csv(challenge_data_path, index=False)
    log(INFO, f"Saving challenge labels to {challenge_label_path}")
    np.save(challenge_label_path, challenge_data_labels)

    log(INFO, "Done!")


if __name__ == "__main__":
    make_challenge_dataset()
