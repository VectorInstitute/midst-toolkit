from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from examples.ensemble_attack.compute_attack_success import compute_attack_success_for_given_targets
from midst_toolkit.common.logger import log


@hydra.main(config_path="./", config_name="config", version_base=None)
def compute_attack_success(config: DictConfig) -> None:
    """Main function to compute the attack success."""
    log(
        INFO,
        f"Computing attack success for target synthetic data at {config.ensemble_attack.target_model.target_synthetic_data_path}...",
    )

    compute_attack_success_for_given_targets(
        target_model_config=config.ensemble_attack.target_model,
        # TODO: refactor this to work better outside of the challenge context (i.e. no target ID)
        # No target ID needed for CTGAN, but it needs at least one element in this array. The value does not matter.
        target_ids=[0],
        experiment_directory=Path(config.results_dir),
        metaclassifier_model_name=config.ensemble_attack.metaclassifier.meta_classifier_model_name,
    )


if __name__ == "__main__":
    compute_attack_success()
