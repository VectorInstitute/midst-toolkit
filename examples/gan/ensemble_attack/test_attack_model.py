from logging import INFO

import hydra
from omegaconf import DictConfig

from examples.ensemble_attack.test_attack_model import run_metaclassifier_testing
from examples.gan.ensemble_attack.utils import make_training_config
from midst_toolkit.attacks.ensemble.model import EnsembleAttackCTGANModelRunner
from midst_toolkit.common.logger import log


@hydra.main(config_path="./", config_name="config", version_base=None)
def attack_model_test(config: DictConfig) -> None:
    """
    Main function to test the attack model.
    
    Args:
        config: The configuration object from the config.yaml file.
    """
    log(
        INFO,
        f"Testing attack model against synthetic data at {config.ensemble_attack.target_model.target_synthetic_data_path}...",
    )

    training_config = make_training_config(config)
    model_runner = EnsembleAttackCTGANModelRunner(training_config=training_config)

    run_metaclassifier_testing(model_runner, config.ensemble_attack)


if __name__ == "__main__":
    attack_model_test()
