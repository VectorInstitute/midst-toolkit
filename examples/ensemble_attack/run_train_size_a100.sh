#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu183
#SBATCH --mem=200G
#SBATCH --job-name=set_6_train
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sara.kodeiri@vectorinstitute.ai


echo "Total memory allocated: $(($SLURM_MEM_PER_NODE / 1024)) GB"
# This script sets up the environment and runs the ensemble attack example.
source .venv/bin/activate

echo "Active Environment:"
which python

echo "Experiments Launched"

python -m examples.ensemble_attack.run_attack  --config-name=experiment_config_size_6

echo "Experiments Completed"
