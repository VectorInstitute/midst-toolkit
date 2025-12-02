#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --qos=normal
#SBATCH --job-name=new_100k_meta_ensemble_attack_train
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=14:00:00


# This script sets up the environment and runs the ensemble attack example.
source .venv/bin/activate

echo "Active Environment:"
which python

echo "Experiments Launched"

python -m examples.ensemble_attack.run_attack --config-name=experiment_config_100k

echo "Experiments Completed"
