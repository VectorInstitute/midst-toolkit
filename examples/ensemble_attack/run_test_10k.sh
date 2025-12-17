#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --job-name=test10_all_targets
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=1:00:00
#SBATCH --array=0-18

# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.

# TARGET_IDS=(21)

TARGET_IDS=( 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)  # List of target IDs

TARGET_ID=${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Running test for target_model_id: $TARGET_ID using all target models' challenges (same shadows)"

echo "Experiments Launched"
pwd

python -m examples.ensemble_attack.test_attack_model --config-name=experiment_config_10k target_model.target_model_id=$TARGET_ID

echo "Experiments Completed"