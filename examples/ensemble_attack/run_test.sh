#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --qos=m
#SBATCH --job-name=ensemble_attack_test
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=9:00:00
#SBATCH --array=0-2  # For 3 target_model_ids (adjust range as needed)

# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.
TARGET_IDS=(21 22 23)  # List of target IDs
TARGET_ID=${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Running test for target_model_id: $TARGET_ID"

echo "Experiments Launched"

python -m examples.ensemble_attack.test_attack_model target_model.target_model_id=$TARGET_ID # Overrides the target_model_id in config.

echo "Experiments Completed"
