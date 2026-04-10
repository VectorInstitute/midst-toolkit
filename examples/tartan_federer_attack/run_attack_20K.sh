#!/bin/bash
#SBATCH --job-name=tartan_federer_20K
#SBATCH --output=/scratch/elabass/elahe_midst/github_clone/midst-toolkit/examples/tartan_federer_attack/logs/tartan_20K_%j.out
#SBATCH --error=/scratch/elabass/elahe_midst/github_clone/midst-toolkit/examples/tartan_federer_attack/logs/tartan_20K_%j.err
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# Activate conda properly in SLURM
source /pkgs/anaconda2023.03/etc/profile.d/conda.sh
conda activate midst312

# Set environment variable
export PYTHONNOUSERSITE=1

# Move to project root
cd /scratch/elabass/elahe_midst/github_clone/midst-toolkit

# Run the attack with absolute config path to avoid Hydra path doubling
HYDRA_FULL_ERROR=1 python examples/tartan_federer_attack/run_attack.py \
    --config-path /scratch/elabass/elahe_midst/github_clone/midst-toolkit/examples/tartan_federer_attack/configs \
    --config-name experiment_config_20K

