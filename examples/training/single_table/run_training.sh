#!/bin/bash
#SBATCH --job-name=midst_training
#SBATCH --output=logs/training_%A_%a.out
#SBATCH --error=logs/training_%A_%a.err
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=1-30

echo "========================================="
echo "Starting Training Job, instance: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

source ~/.bashrc
conda activate midst312
export PYTHONNOUSERSITE=1

cd /scratch/elabass/elahe_midst/github_clone/midst-toolkit

python examples/training/single_table/run_training.py \
    base_data_dir=whitebox_single_table_70/tabddpm_${SLURM_ARRAY_TASK_ID} \
    results_dir=whitebox_single_table_70/tabddpm_${SLURM_ARRAY_TASK_ID}

echo "Training instance $SLURM_ARRAY_TASK_ID finished!"
