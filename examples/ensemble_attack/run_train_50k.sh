#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=290G
#SBATCH --job-name=50k_RMIA_train_save_synth_only
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=12:00:00


echo "Total memory allocated: $(($SLURM_MEM_PER_NODE / 1024)) GB"

# This script sets up the environment and runs the ensemble attack example.
source .venv/bin/activate

echo "Active Environment:"
which python

echo "Experiments Launched"

python -m examples.ensemble_attack.run_attack  --config-name=experiment_config_50k

echo "Experiments Completed"
