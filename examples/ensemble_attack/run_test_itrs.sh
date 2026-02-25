#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --job-name=test_4e5_itrs
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=23:00:00
#SBATCH --array=0-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fatemeh.tavakoli@vectorinstitute.ai
# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.

TARGET_IDS=(111)
# TARGET_IDS=( 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130)  # List of target IDs


TARGET_ID=${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Running test for target_model_id: $TARGET_ID using all target models' challenges (same shadows)"

echo "Experiments Launched"

python -m examples.ensemble_attack.test_attack_model --config-name=experiment_config_4e5_itr target_model.target_model_id=$TARGET_ID

echo "Experiments Completed"