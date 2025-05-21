#!/bin/bash
#SBATCH --job-name=update_siglip_model_pos_embeds.py
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=20G
#SBATCH --output=/fsx/m4/experiments/general_logs/update_siglip_model_pos_embeds/%x_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=hopper-prod
#SBATCH --gpus=1
#SBATCH --qos high


set -x -e

source /fsx/m4/start-m4-user
conda activate base
conda activate shared-m4


WORKING_DIR=/fsx/hugo/repos/m4_36
pushd $WORKING_DIR

python m4/scripts/update_siglip_model_pos_embeds.py
