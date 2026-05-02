#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=success_S6
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=00:20:00

# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.
TARGET_IDS=(8 9 10)  # List of target IDs



PYTHONPATH=./src /h/skodeiri/second/midst-toolkit/.venv/bin/python -m examples.ensemble_attack.diabetes.compute_attack_success --config-name=diabetes_experiment_config_S6
