#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=160G
#SBATCH --job-name=new_exp_10k_train
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=16:00:00


echo "Total memory allocated: $(($SLURM_MEM_PER_NODE / 1024)) GB"

# This script sets up the environment and runs the ensemble attack example.
source .venv/bin/activate

echo "Active Environment:"
which python

echo "Experiments Launched"

python -m examples.ensemble_attack.run_attack  --config-name=experiment_config_10k

echo "Experiments Completed"
