#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=160G
#SBATCH --job-name=berka_test_8192_bs
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=3:00:00
#SBATCH --mail-user=fatemeh.tavakoli@vectorinstitute.ai
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-18

# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.

# TARGET_IDS=(112)
TARGET_IDS=( 111 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130)  # List of target IDs


TARGET_ID=${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Running test for target_model_id: $TARGET_ID using all target models' challenges (same shadows)"

echo "Experiments Launched"

python -m examples.ensemble_attack.berka.test_attack_model --config-name=experiment_config_20k_8192bs target_model.target_model_id=$TARGET_ID

echo "Experiments Completed"