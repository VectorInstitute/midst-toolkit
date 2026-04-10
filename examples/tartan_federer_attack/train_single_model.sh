#!/bin/bash
#SBATCH --job-name=train_tabddpm
#SBATCH --output=/scratch/elabass/elahe_midst/github_clone/midst-toolkit/examples/tartan_federer_attack/logs/train_%a.out
#SBATCH --error=/scratch/elabass/elahe_midst/github_clone/midst-toolkit/examples/tartan_federer_attack/logs/train_%a.err
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=111-140

# Activate conda
source /pkgs/anaconda2023.03/etc/profile.d/conda.sh
conda activate midst312

export PYTHONNOUSERSITE=1

# Move to project root
cd /scratch/elabass/elahe_midst/github_clone/midst-toolkit

MODEL_IDX=${SLURM_ARRAY_TASK_ID}
MODEL_DIR=examples/tartan_federer_attack/data_try_20K_again/tabddpm_white_box/tabddpm_${MODEL_IDX}

echo "Training model tabddpm_${MODEL_IDX}..."

HYDRA_FULL_ERROR=1 python examples/training/single_table/run_training.py \
    --config-path /scratch/elabass/elahe_midst/github_clone/midst-toolkit/${MODEL_DIR} \
    --config-name config

echo "Done training tabddpm_${MODEL_IDX}"
