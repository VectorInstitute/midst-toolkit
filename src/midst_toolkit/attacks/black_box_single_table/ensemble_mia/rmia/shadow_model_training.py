import logging
import pickle
import random
import shutil
from pathlib import Path

import pandas as pd

from midst_toolkit.attacks.black_box_single_table.ensemble_mia.config import (
    DATASET_META_FILE_PATH,
    POPULATION_PATH,
    PROCESSED_ATTACK_DATA_PATH,
    SHADOW_MODELS_ARTIFACTS_PATH,
    SHADOW_MODELS_DATA_PATH,
    TRANS_DOMAIN_FILE_PATH,
    TRANS_JSON_FILE_PATH,
    seed,
)
from midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.data_utils import (
    load_dataframe,
)
from midst_toolkit.attacks.black_box_single_table.ensemble_mia.tabddpm_training import (
    config_tabddpm,
    fine_tune_tabddpm,
    train_tabddpm,
)


def train_fine_tuning_shadows(n_models: int, n_reps: int, init_model_id: int, init_data_seed: int) -> None:
    """
    Train shadow models that start from a pre-trained TabDDPM model and are fine-tuned on a portion of
    the challenge data.
    1. Initial training set includes 60,000 observations, but NONE of the observations included in
        the challenge lists of that repo. This is common to the four models.
    2. TabDDPM is trained on that initial training set. This is common to the four models.
    3. A new "fine-tuning" set is selected with exactly half of the observations included in the
        challenge lists of that repo for each of the four shadow models. Each observation is included in the
        fine-tuning set of exactly half of the models (two models in original submission).
        Each observation is repeated 12 times. Each set is shuffled.
    4. The initial model is fine-tuned independently based on the new set to obtain each shadow model.
    5. A synthetic dataset of 20K observations is generated for each model.

    Args:
            n_models (int): Number of shadow models to train, must be even.
            n_reps (int): Number of repetitions for each challenge point in the fine-tuning set.
            init_model_id (int): Distinguishes the pre-trained initial models.
            init_data_seed (int): Random seed for the initial training set.
    """
    # Randomly sample 60k points from real_all.csv
    # It should not contain any sample that is in challenge points
    master_challenge_df = load_dataframe(PROCESSED_ATTACK_DATA_PATH, "master_challenge_train.csv")
    df_real_all = load_dataframe(POPULATION_PATH, "population_all_with_challenge.csv")

    # Extract unique trans_id values
    unique_ids = master_challenge_df["trans_id"].unique().tolist()
    train_pop = df_real_all[~df_real_all["trans_id"].isin(unique_ids)]

    # Create the necessary folders and config files
    shadow_model_folder = Path(SHADOW_MODELS_DATA_PATH / f"initial_model_rmia_{init_model_id}")
    # Create the new folder if it doesn't exist
    shadow_model_folder.mkdir(exist_ok=True)
    shutil.copyfile(TRANS_DOMAIN_FILE_PATH, Path(shadow_model_folder / "trans_domain.json"))
    shutil.copyfile(DATASET_META_FILE_PATH, Path(shadow_model_folder, "dataset_meta.json"))
    # Train initial model with 60K data without any challenge points
    configs, save_dir = config_tabddpm(
        data_dir=shadow_model_folder,
        json_path=TRANS_JSON_FILE_PATH,
        final_json_path=Path(SHADOW_MODELS_DATA_PATH, "trans.json"),
        diffusion_layers=[512, 1024, 1024, 1024, 1024, 512],
        diffusion_iterations=200000,
        classifier_layers=[128, 256, 512, 1024, 512, 256, 128],
        classifier_dim_t=128,
        classifier_iterations=20000,
    )

    # Create the initial training set
    train = train_pop.sample(n=60000, seed=init_data_seed)
    train.to_csv(Path(shadow_model_folder / "initial_train_set.csv"))

    # Train the initial model
    initial_model = train_tabddpm(train, configs, save_dir)

    # Save the initial model
    # Pickle dump the results
    with open(Path(SHADOW_MODELS_ARTIFACTS_PATH / "rmia_initial_model_2.pkl"), "wb") as file:
        pickle.dump(initial_model, file)

    # Then create 4 random list of challenge points for each shadow model
    # to be used for fine-tuning.
    # Create the random lists, each with half the size of unique_ids
    random.shuffle(unique_ids)  # Shuffle to randomize order
    half_models = n_models // 2
    lists: list = [[] for _ in range(n_models)]

    # Assign each unique_id to half of the random lists
    for uid in unique_ids:
        selected_lists = random.sample(range(n_models), half_models)  # Select 2 random list indices
        for idx in selected_lists:
            lists[idx].append(uid)

    attack_data = {"fine_tuning_sets": lists, "fine_tuned_results": []}

    for idx, ref_list in enumerate(lists):
        logging.info(f"Reference model number: {idx}")
        selected_challenges = master_challenge_df[master_challenge_df["trans_id"].isin(ref_list)]
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=seed).reset_index(drop=True)

        train_result = fine_tune_tabddpm(
            trained_models=initial_model["models"],
            new_train_set=selected_challenges,
            configs=configs,
            save_dir=save_dir,
            new_diffusion_iterations=200000,
            new_classifier_iterations=20000,
            n_synth=20000,
        )

        attack_data["fine_tuned_results"].append(train_result)

    # Pickle dump the results
    with open(Path(SHADOW_MODELS_ARTIFACTS_PATH / "rmia_shadows.pkl"), "wb") as file:
        pickle.dump(attack_data, file)


def train_shadow_on_half_challenge_data(n_models: int, n_reps: int) -> None:
    """
    1. Create eight training sets with exactly half of the observations included in the challenge lists
        of that repo for each of the n`_models` (eight in the original attack) models.
        Each observation is included in the training set of exactly half of the models (four in the original attack).
        Each observation is repeated 12 times. Each set is shuffled.
    2. Train a new TabDDPM model for the `n_models` shadow models (eight models are trained in the original attack).
    3. A synthetic dataset of 20K observations is generated for each model.

    Args:
            n_models (int): number of shadow models to train, must be even.
            n_reps (int): = number of repetitions for each challenge point in the fine-tuning set.
    """
    master_challenge_df = load_dataframe(PROCESSED_ATTACK_DATA_PATH, "master_challenge_train.csv")
    # Extract unique trans_id values
    unique_ids = master_challenge_df["trans_id"].unique().tolist()

    # create the random lists, each with half the size of unique_ids
    random.shuffle(unique_ids)  # Shuffle to randomize order
    half_models = n_models // 2
    lists: list = [[] for _ in range(n_models)]

    # Assign each unique_id to half of the random lists
    for uid in unique_ids:
        selected_lists = random.sample(range(n_models), half_models)  # Select 2 random list indices
        for idx in selected_lists:
            lists[idx].append(uid)

    attack_data = {"selected_sets": lists, "trained_results": []}

    for idx, ref_list in enumerate(lists):
        logging.info(f"Reference model number: {idx}")

        # Create the necessary folders and config files
        folder_name = "shadow_model_rmia_" + str(idx)
        shadow_folder = Path(SHADOW_MODELS_DATA_PATH / folder_name)
        # create the new folder if i doesn't exist
        shadow_folder.mkdir(exist_ok=True)
        shutil.copyfile(TRANS_DOMAIN_FILE_PATH, Path(shadow_folder / "trans_domain.json"))
        shutil.copyfile(DATASET_META_FILE_PATH, Path(shadow_folder / "dataset_meta.json"))
        configs, save_dir = config_tabddpm(
            data_dir=shadow_folder,
            json_path=TRANS_JSON_FILE_PATH,
            final_json_path=Path(SHADOW_MODELS_DATA_PATH, "trans.json"),
            diffusion_layers=[512, 1024, 1024, 1024, 1024, 512],
            diffusion_iterations=200000,
            classifier_layers=[128, 256, 512, 1024, 512, 256, 128],
            classifier_dim_t=128,
            classifier_iterations=20000,
        )

        selected_challenges = master_challenge_df[master_challenge_df["trans_id"].isin(ref_list)]
        logging.info(f"Number of selected challenges to train the shadow model: {len(selected_challenges)}")
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=seed).reset_index(drop=True)

        train_result = train_tabddpm(selected_challenges, configs, save_dir)

        attack_data["trained_results"].append(train_result)

    # Pickle dump the results
    with open(Path(SHADOW_MODELS_ARTIFACTS_PATH, "rmia_shadows_m8.pkl"), "wb") as file:
        pickle.dump(attack_data, file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Shadow model training started...")
    n_models = 4  # number of shadow models to train, must be even
    n_reps = 12  # number of repetitions for each challenge point in the fine-tuning set
    train_fine_tuning_shadows(n_models=n_models, n_reps=n_reps, init_model_id=1, init_data_seed=seed)
    logging.info("First set of shadow model training completed.")
    # The following four models were trained in the same way, with a new initial training set
    # in the hopes of increased performance (gain was minimal).
    train_fine_tuning_shadows(n_models=n_models, n_reps=n_reps, init_model_id=2, init_data_seed=seed + 1)
    logging.info("Second set of shadow model training completed.")
    # The following eight models were trained as follows, still in the hopes of
    # increased performance (again the gain was minimal).
    train_shadow_on_half_challenge_data(n_models=8, n_reps=n_reps)
    logging.info("Third set of shadow model training completed.")
