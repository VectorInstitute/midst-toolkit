from pathlib import Path

from src.midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.real_data_collection import (
    collect_midst_data,
    collect_population_data_ensemble_mia,
)
from tests.unit.attacks.ensemble_mia.config import DATA_CONFIG


def test_collect_population_data_ensemble_mia(tmp_path: Path) -> None:
    # Comment the next line to update population data stored in DATA_CONFIG["population_path"].
    DATA_CONFIG["population_path"] = tmp_path
    _ = collect_population_data_ensemble_mia(
        data_config=DATA_CONFIG,
        attack_types=["tabddpm_black_box", "clavaddpm_black_box"],
    )
    # Assert that the population data is saved in the provided path
    assert (DATA_CONFIG["population_path"] / "population_all.csv").exists()
    assert (DATA_CONFIG["population_path"] / "population_all_no_id.csv").exists()

    # Assert that challenge set is saved in the provided path
    assert (DATA_CONFIG["population_path"] / "challenge_points_all.csv").exists()

    assert (DATA_CONFIG["population_path"] / "population_all_no_challenge.csv").exists()

    assert (DATA_CONFIG["population_path"] / "population_all_with_challenge.csv").exists()

    assert (DATA_CONFIG["population_path"] / "population_all_with_challenge_no_id.csv").exists()


def test_collect_midst_data() -> None:
    df_population = collect_midst_data(
        attack_types=["tabddpm_black_box", "clavaddpm_black_box"],
        data_splits=["train"],
        dataset="train",
        data_config=DATA_CONFIG,
    )
    # Assert that the collected data has the expected number of rows and columns
    assert df_population.shape == (77787, 10), f"Shape is {df_population.shape}"

    # Collect a portion of challenge data
    df_challenge = collect_midst_data(
        attack_types=["tabddpm_black_box"],
        data_splits=["train", "dev"],
        dataset="challenge",
        data_config=DATA_CONFIG,
    )
    # Assert that the collected data has the expected number of rows and columns
    assert df_challenge.shape == (400, 10), f"Shape is {df_challenge.shape}"
