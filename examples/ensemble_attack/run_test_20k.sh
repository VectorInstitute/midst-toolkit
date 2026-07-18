#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=68G
#SBATCH --job-name=test_20k_44
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=2:00:00
#SBATCH --array=0-18

# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.

# # TARGET_IDS=(61 62 63 64 65 66 67 68 69 70)
# TARGET_IDS=(61)
TARGET_IDS=( 62 63 64 65 66 67 68 69 70 101 102 103 104 105 106 107 108 109 110)  # List of target IDs


TARGET_ID=${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Running test for target_model_id: $TARGET_ID using all target models' challenges (same shadows)"

echo "Experiments Launched"

python -m examples.ensemble_attack.test_attack_model --config-name=experiment_config_20k_44 target_model.target_model_id=$TARGET_ID

echo "Experiments Completed"