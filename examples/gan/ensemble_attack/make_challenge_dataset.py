from logging import INFO
from pathlib import Path

import hydra
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

    training_data = pd.read_csv(Path(config.results_dir) / f"{dataset_name}_sampled.csv")
    untrained_data = real_data[~real_data["trans_id"].isin(training_data["trans_id"])].sample(len(training_data))

    challenge_data = pd.concat([training_data, untrained_data])
    challenge_data["label"] = [1] * len(training_data) + [0] * len(untrained_data)

    challenge_data_path = (
        Path(config.ensemble_attack.data_paths.processed_attack_data_path) / f"{dataset_name}_challenge.csv"
    )
    log(INFO, f"Saving challenge data to {challenge_data_path}")
    challenge_data.to_csv(challenge_data_path, index=False)


if __name__ == "__main__":
    make_challenge_dataset()
