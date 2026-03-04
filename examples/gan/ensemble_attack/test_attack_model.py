from logging import INFO

import hydra
from omegaconf import DictConfig

from examples.ensemble_attack.test_attack_model import run_metaclassifier_testing
from midst_toolkit.common.logger import log


@hydra.main(config_path="./", config_name="config", version_base=None)
def test_attack_model(config: DictConfig) -> None:
    """Main function to test the attack model."""
    log(INFO, f"Testing attack model at {config.ensemble_attack.target_model.target_model_directory}...")
    run_metaclassifier_testing(config.ensemble_attack)


if __name__ == "__main__":
    test_attack_model()
