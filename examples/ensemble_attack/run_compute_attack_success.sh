#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=set_5_attack_success
#SBATCH --output=%j_%x_%a.out
#SBATCH --error=%j_%x_%a.err
#SBATCH --time=00:20:00

# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.
TARGET_IDS=(111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130)  # List of target IDs
# TARGET_IDS=(61 62 63 64 65 66 67 68 69 70 101 102 103 104 105 106 107 108 109 110)  # List of target IDs


PYTHONPATH=./src /h/skodeiri/midst-toolkit/.venv/bin/python -m examples.ensemble_attack.compute_attack_success --config-name=experiment_config_size_5
