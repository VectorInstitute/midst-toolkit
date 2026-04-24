#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=bn083
#SBATCH --mem=128G
#SBATCH --job-name=train_S6_meta
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=3:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sara.kodeiri@vectorinstitute.ai

echo "Total memory allocated: $(($SLURM_MEM_PER_NODE / 1024)) GB"
# This script sets up the environment and runs the ensemble attack example.
source .venv/bin/activate

echo "Active Environment:"
which python

echo "Experiments Launched"

python -m examples.ensemble_attack.diabetes.run_attack_diabetes --config-name=diabetes_experiment_config_S6.yaml

echo "Experiments Completed"
