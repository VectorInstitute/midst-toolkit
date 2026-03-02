#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --job-name=final_lower
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=2:00:00

# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.
TARGET_IDS=(61 62 63 64 65 66 67 68 69 70 101 102 103 104 105 106 107 108 109 110)  # List of target IDs

# python -m examples.ensemble_attack.test_attack_model target_model.target_model_id=$TARGET_ID --config-name=experiment_config_20k
PYTHONPATH=./src python -m examples.ensemble_attack.compute_attack_success --config-name=experiment_config_20k_size_small

