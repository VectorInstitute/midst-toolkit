from pathlib import Path
from omegaconf import DictConfig
import pytest
from hydra import initialize, compose
from omegaconf import DictConfig
from src.midst_toolkit.attacks.ensemble.utils import load_dataframe
from src.midst_toolkit.attacks.ensemble.process_split_data import process_split_data


@pytest.fixture(scope="session")
def cfg() -> DictConfig:
    # Adjust path to point to conf/ folder
    with initialize(config_path="."):
        return compose(config_name="test_config")


def test_process_split_data(cfg: DictConfig, tmp_path: Path) -> None:
    # Comment the next line to update processed attack data stored at processed_attack_data_path.
    cfg.data_paths.processed_attack_data_path = tmp_path
    output_dir = Path(cfg.data_paths.processed_attack_data_path)

    process_split_data(
        all_population_data=load_dataframe(
            Path(cfg.data_paths.population_path),
            "all_population.csv",
        ),
        processed_attack_data_path=Path(cfg.data_paths.processed_attack_data_path),
        column_to_stratify=cfg.data_processing_config.column_to_stratify,
        num_total_samples=cfg.data_processing_config.population_sample_size,
        random_seed=cfg.random_seed,
    )

    # Assert that the split real data files are saved in the provided path
    assert (output_dir / "real_train.csv").exists()
    assert (output_dir / "real_val.csv").exists()
    assert (output_dir / "real_test.csv").exists()
    assert (output_dir / "real_test.csv").exists()

    # Assert that the master challenge data files are saved in the provided path
    assert (output_dir / "master_challenge_train.csv").exists()
    assert (output_dir / "master_challenge_train_labels.npy").exists()
    assert (output_dir / "master_challenge_test.csv").exists()
    assert (output_dir / "master_challenge_test_labels.npy").exists()

    # Assert that the collected data has the expected number of rows and columns
    real_train = load_dataframe(output_dir, "real_train.csv")
    # The whole real data in this test is 80 samples, half of which is assigned to training set.
    assert real_train.shape == (40, 10), f"Shape is {real_train.shape}"

    # Real val and test sets each contain a quarter of the whole real data (80 samples).
    real_val = load_dataframe(output_dir, "real_val.csv")
    assert real_val.shape == (20, 10), f"Shape is {real_val.shape}"
    real_test = load_dataframe(output_dir, "real_test.csv")
    assert real_test.shape == (20, 10), f" Shape is {real_test.shape}"

    # Recall that `master_challenge_train`` consists of two halves: one half (20 samples) from `real_val`` data
    # with their "is_train" column set to 0, and the other half (20 samples) from the real train data (`real_train``)
    # with their "is_train" column set to 1. Note that ["is_train"] column is dropped in the final dataframes.
    master_challenge_train = load_dataframe(output_dir, "master_challenge_train.csv")
    assert master_challenge_train.shape == (40, 10), f" Shape is {master_challenge_train.shape}"

    # Recall that `master_challenge_test`` consists of two halves: one half (20 samples) from `real_test`` data
    # with their "is_train" column set to 0, and the other half (20 samples) from the real train data (`real_train``)
    # with their "is_train" column set to 1. Note that ["is_train"] column is dropped in the final dataframes.
    master_challenge_test = load_dataframe(output_dir, "master_challenge_test.csv")
    assert master_challenge_test.shape == (40, 10), f" Shape is {master_challenge_test.shape}"
