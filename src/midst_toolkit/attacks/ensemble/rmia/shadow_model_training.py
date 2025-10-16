import pickle
import random
import shutil
from logging import INFO
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.shadow_model_utils import (
    fine_tune_tabddpm_and_synthesize,
    save_additional_tabddpm_config,
    train_tabddpm_and_synthesize,
)
from midst_toolkit.common.logger import log


# TODO: This function and the next one can be unified later.
def train_fine_tuned_shadow_models(
    n_models: int,
    n_reps: int,
    population_data: pd.DataFrame,
    master_challenge_data: pd.DataFrame,
    shadow_models_output_path: Path,
    training_json_config_paths: DictConfig,
    fine_tuning_config: DictConfig,
    init_model_id: int,
    table_name: str,
    id_column_name: str,
    pre_training_data_size: int = 60000,
    init_data_seed: int | None = None,
    random_seed: int | None = None,
) -> Path:
    """
    Train ``n_models`` shadow models that start from a pre-trained TabDDPM model and are fine-tuned on
    a portion of the challenge data.
    The attack's shadow training design is described as follows:
    1. Initial training set includes ``pre_training_data_size`` observations (60,000 in the original attack code),
        but NONE of the observations included in the challenge lists of that repo.
    2. One TabDDPM is trained on that initial training set. This is then used as the pre-trained model for all
        shadow models.
    3. A new "fine-tuning" set is selected with exactly half of the observations included in the challenge lists
        for each of the shadow models. Each observation is included in the fine-tuning set of exactly half of
        the models.
        Each observation is repeated ``n_reps`` times. Each set is shuffled.
    4. The pre-trained model is fine-tuned independently based on the new "fine-tuning" set to obtain each
        shadow model.
    5. A synthetic dataset of 20K observations is generated for each model.

    Note: All the numbers such as 60K and 20K mentioned above are the used values by the attack designers,
    but are nto enforced in this code and can be changed according to the user preference. ``pre_training_data_size``
    specified the size of the pre-training set, and the number of synthetic samples generated will be equal to the
    size of fine-tuning set.

    Args:
            n_models: Number of shadow models to train, must be even.
            n_reps: Number of repetitions for each challenge point in the fine-tuning set.
            population_data: The total population data that the attacker has access to.
            master_challenge_data: The master challenge training dataset.
            shadow_models_output_path: Path where the all datasets and information necessary to train shadow models
                will be saved. Model artifacts and synthetic data will be saved under this directory as well.
            training_json_config_paths: Configuration dictionary containing paths to the data JSON config files.
                An example of this config is provided in ``examples/ensemble_attack/config.yaml``. Required keys are:
                - table_domain_file_path (str): Path to the table domain json file.
                - dataset_meta_file_path (str): Path to dataset meta json file.
                - tabddpm_training_config_path (str): Path to table's training config json file.
            fine_tuning_config: Configuration dictionary containing shadow model fine-tuning specific information.
            init_model_id: An ID to assign to the pre-trained initial models. This can be used to save multiple
                pre-trained models with different IDs.
            table_name: Name of the main table to be used for training the TabDDPM model.
            id_column_name: Name of the ID column in the data.
            pre_training_data_size: Size of the initial training set, defaults to 60,000.
            init_data_seed: Random seed for the initial training set.
            random_seed: Random seed used for reproducibility, defaults to None.

    Returns:
            The path where the shadow models and their artifacts are saved.

    """
    # Pre-training set should not contain any sample that is in challenge points
    unique_ids = master_challenge_data[id_column_name].unique().tolist()
    train_pop = population_data[~population_data[id_column_name].isin(unique_ids)]
    # Checking if there are enough samples in the pre-training set once the
    # samples that are in master_challenge_data are removed.
    assert len(train_pop) >= pre_training_data_size, (
        f"Not enough population data to create the pre-training set of size {pre_training_data_size} "
        f"without challenge points. Available non-challenge points: {len(train_pop)}"
    )

    # Create the necessary folders and config files
    shadow_model_data_folder = shadow_models_output_path / f"initial_model_rmia_{init_model_id}"
    # Create the new folder if it doesn't exist
    shadow_model_data_folder.mkdir(exist_ok=True)

    # Create the initial training set (train data)
    # Randomly sample ``pre_train_data_size`` points from all the population data.
    train = train_pop.sample(n=pre_training_data_size, random_state=init_data_seed)
    train.to_csv(shadow_model_data_folder / "initial_train_set.csv")

    # Copy the json config files to the data folder
    shutil.copyfile(
        training_json_config_paths.table_domain_file_path,
        shadow_model_data_folder / f"{table_name}_domain.json",
    )
    shutil.copyfile(
        training_json_config_paths.dataset_meta_file_path,
        shadow_model_data_folder / "dataset_meta.json",
    )

    # Train initial model with 60K data without any challenge points
    # ``save_additional_tabddpm_config`` makes a personalized copy of the training config for each
    # tabddpm model (here the base model).
    # All the shadow models will be saved under the base model data directory.
    configs, save_dir = save_additional_tabddpm_config(
        data_dir=shadow_model_data_folder,
        training_config_json_path=Path(training_json_config_paths.tabddpm_training_config_path),
        final_config_json_path=shadow_model_data_folder / f"{table_name}.json",  # Path to the new json
        experiment_name="pre_trained_model",
    )

    # Train the initial model if it is not already trained and saved.
<<<<<<< HEAD
    if not (save_dir / f"initial_model_rmia_{init_model_id}.pkl").exists():
=======
    initial_model_path = save_dir / f"initial_model_rmia_{init_model_id}.pkl"
    if not initial_model_path.exists():
>>>>>>> main
        log(INFO, f"Training initial model with ID {init_model_id}...")
        initial_model_training_results = train_tabddpm_and_synthesize(train, configs, save_dir, synthesize=False)

        # Save the initial model
        # Pickle dump the results
<<<<<<< HEAD
        with open(save_dir / f"initial_model_rmia_{init_model_id}.pkl", "wb") as file:
            pickle.dump(initial_model_training_results, file)
        log(
            INFO,
            f"Initial model with ID {init_model_id} trained and saved at {save_dir / f'initial_model_rmia_{init_model_id}.pkl'}.",
        )
    else:
        log(INFO, f"Initial model with ID {init_model_id} already exists, loading it from disk.")
        with open(save_dir / f"initial_model_rmia_{init_model_id}.pkl", "rb") as f:
            initial_model_training_results = pickle.load(f)

    # assert initial_model_training_results.models[("", table_name)]["diffusion"] is not None

=======
        with open(initial_model_path, "wb") as file:
            pickle.dump(initial_model_training_results, file)
        log(
            INFO,
            f"Initial model with ID {init_model_id} trained and saved at {initial_model_path}.",
        )
    else:
        log(INFO, f"Initial model with ID {init_model_id} already exists, loading it from disk.")
        with open(initial_model_path, "rb") as f:
            initial_model_training_results = pickle.load(f)

>>>>>>> main
    # Then create 4 random list of challenge points for each shadow model
    # to be used for fine-tuning.
    random.shuffle(unique_ids)  # Shuffle to randomize order
    half_models = n_models // 2
    selected_id_lists: list[list[int]] = [[] for _ in range(n_models)]

    # Assign each unique_id to half of the random lists (used to train shadow models)
    for uid in unique_ids:
        selected_lists = random.sample(range(n_models), half_models)  # Select 2 random list indices
        for idx in selected_lists:
            selected_id_lists[idx].append(uid)

    attack_data: dict[str, Any] = {
        "fine_tuning_sets": selected_id_lists,
        "fine_tuned_results": [],
    }

    for model_id, ref_list in enumerate(selected_id_lists):
        log(INFO, f"Reference model number: {model_id}")
        selected_challenges = master_challenge_data[master_challenge_data[id_column_name].isin(ref_list)]
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        train_result = fine_tune_tabddpm_and_synthesize(
            trained_models=initial_model_training_results.models,
            fine_tune_set=selected_challenges,
            configs=configs,
            save_dir=save_dir,
            fine_tuning_diffusion_iterations=fine_tuning_config.fine_tune_diffusion_iterations,
            fine_tuning_classifier_iterations=fine_tuning_config.fine_tune_classifier_iterations,
            synthesize=True,
        )
        assert train_result.synthetic_data is not None, "Fine-tuned models should generate synthetic data."
        attack_data["fine_tuned_results"].append(train_result)

    # Pickle dump the results
    result_path = Path(save_dir / "rmia_shadows.pkl")
    with open(result_path, "wb") as file:
        pickle.dump(attack_data, file)

    return result_path


def train_shadow_on_half_challenge_data(
    n_models: int,
    n_reps: int,
    master_challenge_data: pd.DataFrame,
    shadow_models_output_path: Path,
    training_json_config_paths: DictConfig,
    table_name: str,
    id_column_name: str,
    random_seed: int | None = None,
) -> Path:
    """
    1. Create eight training sets with exactly half of the observations included in the challenge lists
        of that repo for each of the n`_models` (eight in the original attack) models.
        Each observation is included in the training set of exactly half of the models (four in the original attack).
        Each observation is repeated 12 times. Each set is shuffled.
    2. Train a new TabDDPM model for the `n_models` shadow models (eight models are trained in the original attack).
    3. A synthetic dataset of 20K observations is generated for each model.

    Args:
            n_models: number of shadow models to train, must be even.
            n_reps: number of repetitions for each challenge point in the fine-tuning set.
            master_challenge_data: The master challenge training dataset.
            shadow_models_output_path: Path where the all datasets and information necessary to train shadow models
                will be saved.
            training_json_config_paths: Configuration dictionary containing paths to the data JSON config files.
                An example of this config is provided in ``examples/ensemble_attack/config.yaml``. Required keys are:
                - table_domain_file_path (str): Path to the table domain json file.
                - dataset_meta_file_path (str): Path to dataset meta json file.
                - tabddpm_training_config_path (str): Path to table's training config json file.
            table_name: Name of the main table to be used for training the TabDDPM model.
            id_column_name: Name of the ID column in the data.
            random_seed: Random seed used for reproducibility, defaults to None.

    Returns:
            The path where the shadow models and their artifacts are saved.

    """
    # Extract unique id values of the master challenge points
    unique_ids = master_challenge_data[id_column_name].unique().tolist()

    # Create 4 random list of challenge points for each shadow model training..
    random.shuffle(unique_ids)  # Shuffle to randomize order
    half_models = n_models // 2
    selected_id_lists: list[list[int]] = [[] for _ in range(n_models)]
    # Assign each unique_id to half of the random lists
    for uid in unique_ids:
        selected_lists = random.sample(range(n_models), half_models)  # Select 2 random list indices
        for idx in selected_lists:
            selected_id_lists[idx].append(uid)

    # Create the necessary folders and config files
    shadow_folder = shadow_models_output_path / "shadow_model_rmia_third_set"
    shadow_folder.mkdir(exist_ok=True)
    shutil.copyfile(
        training_json_config_paths.table_domain_file_path,
        shadow_folder / f"{table_name}_domain.json",
    )
    shutil.copyfile(
        training_json_config_paths.dataset_meta_file_path,
        shadow_folder / "dataset_meta.json",
    )
    configs, save_dir = save_additional_tabddpm_config(
        data_dir=shadow_folder,
        training_config_json_path=Path(training_json_config_paths.tabddpm_training_config_path),
        final_config_json_path=shadow_folder / f"{table_name}.json",  # Path to the new json
        experiment_name="trained_model",
    )
    attack_data: dict[str, Any] = {
        "selected_sets": selected_id_lists,
        "trained_results": [],
    }

    for model_id, ref_list in enumerate(selected_id_lists):
        log(INFO, f"Reference model number: {model_id}")

        selected_challenges = master_challenge_data[master_challenge_data[id_column_name].isin(ref_list)]
        log(
            INFO,
            f"Number of selected challenges to train the shadow model: {len(selected_challenges)}",
        )
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        train_result = train_tabddpm_and_synthesize(
            selected_challenges,
            configs,
            save_dir,
            synthesize=True,
        )

        attack_data["trained_results"].append(train_result)

    # Pickle dump the results
    result_path = Path(save_dir, "rmia_shadows_third_set.pkl")
    with open(result_path, "wb") as file:
        pickle.dump(attack_data, file)

    return result_path


def train_three_sets_of_shadow_models(
    population_data: pd.DataFrame,
    master_challenge_data: pd.DataFrame,
    shadow_models_output_path: Path,
    training_json_config_paths: DictConfig,
    fine_tuning_config: DictConfig,
    table_name: str,
    id_column_name: str,
    n_models_per_set: int = 4,
    n_reps: int = 12,
    random_seed: int | None = None,
) -> tuple[Path, Path, Path]:
    """
    Runs the shadow model training pipeline of the ensemble attack. This pipeline trains three sets of shadow models.
    In the first step, one-fourth of the shadow models are pre-trained on a random subset of the ``population_data``
    and then fine-tuned on half of the challenge points. In the second step, another one-fourth of the shadow models
    are pre-trained on a different random subset of the ``population_data`` and then fine-tuned on half of the
    challenge points.
    In the third step, the remaining half of the shadow models are trained from scratch on half of the challenge
    points.
    Each observation in the challenge points is included in the training set of exactly half of the shadow models
    in each set.
    Each observation in the challenge points is repeated ``n_reps`` times in the training set of each shadow model.

    This attack by default trains 16 shadow models in total, 8 of which are fine-tuned from a pre-traine model and
    8 of which are trained from scratch on the challenge points. Pre-training is done on a number of samples from the
    population data as stated in the ``fine_tuning_config`` or `60K` samples by default.
    We keep track of and save the challenge id's used to train each shadow model to be used by RMIA.

    ``population_data`` is all the data that the attacker has access to, including the challenge points.
    ``master_challenge_data`` is the master challenge data that will be used to train the meta classifier.
    It includes challenge points that we have the labels for
    (i.e., whether they were used to train the target model or not). Please refer to the attack example README
    at ``examples/ensemble_attack/README.md`` for more details.


    Args:
        population_data: The total population data used for pre-training some of the shadow models.
        master_challenge_data: The master challenge training dataset.
        shadow_models_output_path: Path where the all datasets and information (configs) necessary to
            train shadow models will be saved. Model artifacts and synthetic data will be saved under
            this directory as well. This path will be created if it does not exist, and all the relevant
            configs will be copied here automatically.
        training_json_config_paths: Configuration dictionary containing paths to the data JSON config files.
            An example of this config is provided in ``examples/ensemble_attack/config.yaml``. Required keys are:
                - table_domain_file_path (str): Path to the table domain json file.
                - dataset_meta_file_path (str): Path to dataset meta json file.
                - tabddpm_training_config_path (str): Path to table's training config json file.
        fine_tuning_config: Configuration dictionary containing shadow model fine-tuning specific information.
            An example of this config is provided in ``examples/ensemble_attack/config.yaml``. Required keys are:
                - fine_tune_diffusion_iterations (int): Number of diffusion fine-tuning iterations.
                - fine_tune_classifier_iterations (int): Number of classifier fine-tuning iterations.
                - pre_train_data_size (int): Size of the data used for pre-training the initial TabDDPM model.
        table_name: Name of the main table to be used for training the TabDDPM model.
        id_column_name: Name of the ID column in the data.
        n_models_per_set: Number of shadow models to train by each approach. Must be an even number. Defaults to 4.
        n_reps: Number of repetitions for each challenge point in the fine-tuning or training sets, defaults to 12.
        random_seed: Random seed used for reproducibility, defaults to None.

    Returns:
        Paths where the shadow models and their artifacts including synthetic data are saved for each of
            the three sets of shadows.
    """
    # Number of shadow models to train, must be even
    assert n_models_per_set % 2 == 0, "n_models_per_set must be even."
    # Create the folder including their parent directories if they don't exist
    shadow_models_output_path.mkdir(parents=True, exist_ok=True)

    first_set_result_path = train_fine_tuned_shadow_models(
        n_models=n_models_per_set,
        n_reps=n_reps,
        population_data=population_data,
        master_challenge_data=master_challenge_data,
        shadow_models_output_path=shadow_models_output_path,
        training_json_config_paths=training_json_config_paths,
        fine_tuning_config=fine_tuning_config,
        init_model_id=1,  # To distinguish these shadow models from the next ones
        table_name=table_name,
        id_column_name=id_column_name,
        pre_training_data_size=fine_tuning_config.pre_train_data_size,
        init_data_seed=random_seed,
        random_seed=random_seed,
    )
    log(
        INFO,
        f"First set of shadow model training completed and saved at {first_set_result_path}",
    )
    # Original codebase comment: "The following four models are trained in the same way,
    # with a new initial training set
    # in the hopes of increased performance (gain was minimal based on the submission comments).""
    second_set_result_path = train_fine_tuned_shadow_models(
        n_models=n_models_per_set,
        n_reps=n_reps,
        population_data=population_data,
        master_challenge_data=master_challenge_data,
        shadow_models_output_path=shadow_models_output_path,
        training_json_config_paths=training_json_config_paths,
        fine_tuning_config=fine_tuning_config,
        init_model_id=2,  # To distinguish these shadow models from the previous ones
        table_name=table_name,
        id_column_name=id_column_name,
        pre_training_data_size=fine_tuning_config.pre_train_data_size,
        # Setting a different seed for the second train set
        init_data_seed=random_seed + 1 if random_seed is not None else None,
        random_seed=random_seed,
    )
    log(
        INFO,
        f"Second set of shadow model training completed and saved at {second_set_result_path}.",
    )
    # Original codebase comment: "The following eight models are trained from scratch on the challenge points,
    # still in the hopes of increased performance (again the gain was minimal).""
    third_set_result_path = train_shadow_on_half_challenge_data(
        n_models=n_models_per_set * 2,
        n_reps=n_reps,
        master_challenge_data=master_challenge_data,
        shadow_models_output_path=shadow_models_output_path,
        training_json_config_paths=training_json_config_paths,
        table_name=table_name,
        id_column_name=id_column_name,
        random_seed=random_seed,
    )
    log(
        INFO,
        f"Third set of shadow model training completed and saved at: {third_set_result_path}",
    )
    return first_set_result_path, second_set_result_path, third_set_result_path
