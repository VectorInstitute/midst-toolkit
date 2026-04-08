#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --job-name=10k_test_error_commented_diabetes
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sara.kodeiri@vectorinstitute.ai

# This script sets up the environment and runs the ensemble attack test script.
source /h/skodeiri/second/midst-toolkit/.venv/bin/activate || true

echo "Active Environment:"
which python


# Map SLURM_ARRAY_TASK_ID to target_model_id.

# TARGET_IDS=(8)
TARGET_IDS=(9 10)  # List of target IDs


TARGET_ID=${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Running test for target_model_id: $TARGET_ID using all target models' challenges (same shadows)"

echo "Experiments Launched"

# run with the explicit venv python to avoid any PATH/module interference
PYTHONPATH=./src /h/skodeiri/second/midst-toolkit/.venv/bin/python -m examples.ensemble_attack.diabetes.test_attack_model --config-name=diabetes_experiment_config_10k target_model.target_model_id=$TARGET_ID


echo "Experiments Completed"