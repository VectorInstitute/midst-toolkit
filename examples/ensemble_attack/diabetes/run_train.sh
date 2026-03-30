#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --job-name=diabetes_v2_train
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=48:00:00


echo "Total memory allocated: $(($SLURM_MEM_PER_NODE / 1024)) GB"
# This script sets up the environment and runs the ensemble attack example.
source /h/skodeiri/second/midst-toolkit/.venv/bin/activate || true

echo "Active Environment:"
which python

echo "Experiments Launched"

# python -m examples.ensemble_attack.diabetes.run_attack_diabetes  --config-name=diabetes_experiment_config
PYTHONPATH=./src /h/skodeiri/second/midst-toolkit/.venv/bin/python -m examples.ensemble_attack.diabetes.run_attack_diabetes --config-name=diabetes_v2_experiment_config

echo "Experiments Completed"
