#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=190G
#SBATCH --job-name=uniform_train_diabetes
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fatemeh.tavakoli@vectorinstitute.ai

echo "Total memory allocated: $(($SLURM_MEM_PER_NODE / 1024)) GB"
# This script sets up the environment and runs the ensemble attack example.
source .venv/bin/activate

echo "Active Environment:"
which python

echo "Experiments Launched"

python -m examples.ensemble_attack.diabetes.run_attack_diabetes --config-name=diabetes_experiment_config_10k_uniform   

echo "Experiments Completed"
