#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=success_44
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=00:20:00


# This script sets up the environment and runs the ensemble attack test script.
source .venv/bin/activate

echo "Active Environment:"
which python

# Map SLURM_ARRAY_TASK_ID to target_model_id.
TARGET_IDS=(61 62 63 64 65 67 68 70 101 102 103 104 105 106 107 108 109 110) # List of target_model_ids to test.
TARGET_ID=${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Experiments Launched"

python -m examples.ensemble_attack.compute_attack_success --config-name=experiment_config_batch_8192_44

echo "Experiments Completed"
