import pickle
import random
import shutil
from pathlib import Path
from logging import INFO
import pandas as pd
from omegaconf import DictConfig
from midst_toolkit.common.logger import log
from midst_toolkit.attacks.ensemble.tabddpm_training import (
    config_tabddpm,
    fine_tune_tabddpm_and_synthesize,
    train_tabddpm_and_synthesize,
)

def train_fine_tuning_shadows(
    n_models: int,
    n_reps: int,
    population_data: pd.DataFrame,
    master_challenge_data: pd.DataFrame,
    shadow_models_data_path: Path,
    training_json_config_paths: DictConfig,
    shadow_training_config: DictConfig,
    init_model_id: int,
    init_data_seed: int,
    pre_training_data_size: int = 60000,
    random_seed: int = 42,
) -> None:
    """
    Train ``n_models`` shadow models that start from a pre-trained TabDDPM model and are fine-tuned on a portion of
    the challenge data.
    1. Initial training set includes 60,000 observations, but NONE of the observations included in
        the challenge lists of that repo.
    2. One TabDDPM is trained on that initial training set. This is then used as the pre-trained model for all shadow models.
    3. A new "fine-tuning" set is selected with exactly half of the observations included in the challenge lists
        for each of the shadow models. Each observation is included in the fine-tuning set of exactly half of the models.
        Each observation is repeated ``n_reps`` times. Each set is shuffled.
    4. The pre-trained model is fine-tuned independently based on the new "fine-tuning" set to obtain each shadow model.
    5. A synthetic dataset of 20K observations is generated for each model.

    Args:
            n_models: Number of shadow models to train, must be even.
            n_reps: Number of repetitions for each challenge point in the fine-tuning set.
            population_data: The total population data that the attacker has access to.
            master_challenge_data: The master challenge training dataset.
            shadow_models_data_path: Path where the all datasets and information necessary to train shadow models
                will be saved. Model artifacts and synthetic data will be saved under this directory as well.
            training_json_config_paths: Configuration dictionary containing paths to the data JSON config files.
            shadow_training_config: Configuration dictionary containing shadow model training specific information.
            init_model_id: Distinguishes the pre-trained initial models.
            init_data_seed: Random seed for the initial training set.
            pre_training_data_size: Size of the initial training set, defaults to 60,000.
            random_seed: Random seed used for reproducibility, defaults to 42.

    Returns:
            None

    """

    # Pre-training set should not contain any sample that is in challenge points
    unique_ids = master_challenge_data["trans_id"].unique().tolist()
    train_pop = population_data[~population_data["trans_id"].isin(unique_ids)]

    # Create the necessary folders and config files
    shadow_model_data_folder = Path(shadow_models_data_path / f"initial_model_rmia_{init_model_id}")
    # Create the new folder if it doesn't exist
    shadow_model_data_folder.mkdir(exist_ok=True)

    # Create the initial training set (train data)
    # Randomly sample ``pre_train_data_size`` points from all the population data.
    train = train_pop.sample(n=pre_training_data_size, random_state=init_data_seed)
    train.to_csv(Path(shadow_model_data_folder / "initial_train_set.csv"))

    # Copy the json config files to the data folder
    shutil.copyfile(
        training_json_config_paths.trans_domain_file_path,
        Path(shadow_model_data_folder / "trans_domain.json"),
    )
    shutil.copyfile(
        training_json_config_paths.dataset_meta_file_path,
        Path(shadow_model_data_folder, "dataset_meta.json"),
    )

    # Train initial model with 60K data without any challenge points
    # Note: in the ``config_tabddpm`` implementation, only string typed addresses (not Path) can be saved in JSON files.
    # ``config_tabddpm`` makes a personalized copy of the training config for each tabddpm model (here the base model).
    # All the shadow models will be saved under the base model data directory.
    configs, save_dir = config_tabddpm(
        data_dir=shadow_model_data_folder,
        training_json_path=Path(training_json_config_paths.tabddpm_training_config_path),
        final_json_path=Path(shadow_model_data_folder, "trans.json"),  # Path to the new json
        experiment_name="pre_trained_model",
    )

    # Train the initial model if it is not already trained and saved.
    if not Path(save_dir / f"rmia_initial_model_{init_model_id}.pkl").exists():
        initial_model = train_tabddpm_and_synthesize(train, configs, save_dir, n_synth=0)

        # Save the initial model
        # Pickle dump the results
        with open(
            Path(save_dir / f"rmia_initial_model_{init_model_id}.pkl"), "wb"
        ) as file:
            pickle.dump(initial_model, file)
    else:
        initial_model = pickle.load(
            open(Path(save_dir / f"rmia_initial_model_{init_model_id}.pkl"), "rb")
        )
    assert initial_model["models"][(None, "trans")]["diffusion"] is not None

    # Then create 4 random list of challenge points for each shadow model
    # to be used for fine-tuning.
    random.shuffle(unique_ids)  # Shuffle to randomize order
    half_models = n_models // 2
    lists: list = [[] for _ in range(n_models)]

    # Assign each unique_id to half of the random lists (used to train shadow models)
    for uid in unique_ids:
        selected_lists = random.sample(range(n_models), half_models)  # Select 2 random list indices
        for idx in selected_lists:
            lists[idx].append(uid)

    attack_data = {"fine_tuning_sets": lists, "fine_tuned_results": []}

    for model_id, ref_list in enumerate(lists):
        log(INFO, f"Reference model number: {model_id}")
        selected_challenges = master_challenge_data[
            master_challenge_data["trans_id"].isin(ref_list)
        ]
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        train_result = fine_tune_tabddpm_and_synthesize(
            trained_models=initial_model["models"],
            new_train_set=selected_challenges,
            configs=configs,
            save_dir=save_dir,
            fine_tuning_diffusion_iterations=shadow_training_config.fine_tune_diffusion_iterations,
            fine_tuning_classifier_iterations=shadow_training_config.fine_tune_classifier_iterations,
            n_synth=shadow_training_config.num_synth_samples,
        )

        attack_data["fine_tuned_results"].append(train_result)

    # Pickle dump the results
    with open(Path(save_dir / "rmia_shadows.pkl"), "wb") as file:
        pickle.dump(attack_data, file)


def train_shadow_on_half_challenge_data(
    n_models: int,
    n_reps: int,
    master_challenge_data: Path,
    shadow_models_data_path: Path,
    training_json_config_paths: DictConfig,
    shadow_training_config: DictConfig,
    random_seed: int = 42,
) -> None:
    """
    1. Create eight training sets with exactly half of the observations included in the challenge lists
        of that repo for each of the n`_models` (eight in the original attack) models.
        Each observation is included in the training set of exactly half of the models (four in the original attack).
        Each observation is repeated 12 times. Each set is shuffled.
    2. Train a new TabDDPM model for the `n_models` shadow models (eight models are trained in the original attack).
    3. A synthetic dataset of 20K observations is generated for each model.

    Args:
            n_models: number of shadow models to train, must be even.
            n_reps : = number of repetitions for each challenge point in the fine-tuning set.
            master_challenge_df: The master challenge training dataset.
            shadow_models_data_path: Path where the all datasets and information necessary to train shadow models
                will be saved.
            shadow_models_artifacts_path: Path where the trained shadow models and synthetic data will be saved.
            data_json_config_paths: Configuration dictionary containing paths to the data JSON config files.
            shadow_training_config: Configuration dictionary containing shadow model training specific information.
            random_seed: Random seed used for reproducibility, defaults to 42.
    """
    # Extract unique trans_id values of the master challenge points
    unique_ids = master_challenge_data["trans_id"].unique().tolist()

    # Create 4 random list of challenge points for each shadow model training..
    random.shuffle(unique_ids)  # Shuffle to randomize order
    half_models = n_models // 2
    lists: list = [[] for _ in range(n_models)]
    # Assign each unique_id to half of the random lists
    for uid in unique_ids:
        selected_lists = random.sample(range(n_models), half_models)  # Select 2 random list indices
        for idx in selected_lists:
            lists[idx].append(uid)

    # Create the necessary folders and config files
    # TODO: do not change the model name and folder names for now to be consistent for RMIA implementation.
    shadow_folder = Path(shadow_models_data_path / "shadow_model_rmia_m8")
    shadow_folder.mkdir(exist_ok=True)
    shutil.copyfile(
        training_json_config_paths.trans_domain_file_path,
        Path(shadow_folder / "trans_domain.json"),
    )
    shutil.copyfile(
        training_json_config_paths.dataset_meta_file_path,
        Path(shadow_folder / "dataset_meta.json"),
    )
    configs, save_dir = config_tabddpm(
        data_dir=shadow_folder,
        training_json_path=Path(training_json_config_paths.tabddpm_training_config_path),
        final_json_path=Path(shadow_folder, "trans.json"),  # Path to the new json
        experiment_name="trained_model",
    )
    attack_data = {"selected_sets": lists, "trained_results": []}

    for model_id, ref_list in enumerate(lists):
        log(INFO, f"Reference model number: {model_id}")

        selected_challenges = master_challenge_data[
            master_challenge_data["trans_id"].isin(ref_list)
        ]
        log(INFO, f"Number of selected challenges to train the shadow model: {len(selected_challenges)}")
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        train_result = train_tabddpm_and_synthesize(
            selected_challenges,
            configs,
            save_dir,
            n_synth=shadow_training_config.num_synth_samples,
        )

        attack_data["trained_results"].append(train_result)

    # Pickle dump the results
    with open(Path(save_dir, "rmia_shadows_m8.pkl"), "wb") as file:
        pickle.dump(attack_data, file)


def run_shadow_model_training(
    population_data: pd.DataFrame,
    master_challenge_data: pd.DataFrame,
    shadow_models_data_path: Path,
    training_json_config_paths: DictConfig,
    shadow_training_config: DictConfig,
    n_models: int = 4,
    n_reps: int = 12,
    random_seed: int = 42,
) -> None:
    """
    Runs the shadow model training pipeline of the ensemble attack.

    Args:
        population_data: The total population data used for pre-training some of the shadow models.
        master_challenge_data: The master challenge training dataset.
        shadow_models_data_path: Path where the all datasets and information (configs) necessary to train shadow models
            will be saved. Model artifacts and synthetic data will be saved under this directory as well. This path
            will be created if it does not exist, and all the relevant configs will be copied here automatically.
        training_json_config_paths: Configuration dictionary containing paths to the data JSON config files.
        shadow_training_config: Configuration dictionary containing shadow model training specific information.
        n_models: Number of shadow models to train, must be even, defaults to 4.
        n_reps: Number of repetitions for each challenge point in the fine-tuning or training sets, defaults to 12.
        random_seed: Random seed used for reproducibility, defaults to 42.
    """
    # Number of shadow models to train, must be even
    assert n_models % 2 == 0, "n_models must be even."
    # Create the folder including their parent directories if they don't exist
    shadow_models_data_path.mkdir(parents=True, exist_ok=True)

    train_fine_tuning_shadows(
        n_models=n_models,
        n_reps=n_reps,
        population_data=population_data,
        master_challenge_data=master_challenge_data,
        shadow_models_data_path=shadow_models_data_path,
        training_json_config_paths=training_json_config_paths,
        shadow_training_config=shadow_training_config,
        init_model_id=1, # To distinguish these shadow models from the next ones
        init_data_seed=random_seed,
        pre_training_data_size=shadow_training_config.pre_train_data_size,
        random_seed=random_seed,
    )
    log(INFO, "First set of shadow model training completed ")
    # The following four models are trained in the same way, with a new initial training set
    # in the hopes of increased performance (gain was minimal based on the submission comments).
    train_fine_tuning_shadows(
        n_models=n_models,
        n_reps=n_reps,
        population_data=population_data,
        master_challenge_data=master_challenge_data,
        shadow_models_data_path=shadow_models_data_path,
        training_json_config_paths=training_json_config_paths,
        shadow_training_config=shadow_training_config,
        init_model_id=2,  # To distinguish these shadow models from the previous ones
        init_data_seed=random_seed + 1,
        pre_training_data_size=shadow_training_config.pre_train_data_size,
        random_seed=random_seed,
    )
    log(INFO, "Second set of shadow model training completed.")
    # The following eight models are trained as from scratch on the challenge points,
    # still in the hopes of increased performance (again the gain was minimal).
    train_shadow_on_half_challenge_data(
        n_models=n_models * 2,
        n_reps=n_reps,
        master_challenge_data=master_challenge_data,
        shadow_models_data_path=shadow_models_data_path,
        training_json_config_paths=training_json_config_paths,
        shadow_training_config=shadow_training_config,
        random_seed=random_seed,
    )
    log(INFO, "Third set of shadow model training completed")
