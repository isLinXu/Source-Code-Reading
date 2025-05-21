#!/bin/bash
#SBATCH --job-name=merge_on_image_individual_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --mem-per-cpu=21G
#SBATCH --output=/fsx/m4/experiments/general_logs/merge_on_image_individual_dataset/res%A_%a
#SBATCH --partition=hopper-prod

source /fsx/m4/start-m4-user
conda activate /fsx/m4/conda/hugo_3


python /fsx/hugo/repos/m4_15/datasets_processing_scripts/build_concatenation_datasets_sft/merge_on_image_individual_dataset.py ${SLURM_ARRAY_TASK_ID}
