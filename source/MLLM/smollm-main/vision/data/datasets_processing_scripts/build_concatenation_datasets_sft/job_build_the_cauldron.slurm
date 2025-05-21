#!/bin/bash
#SBATCH --job-name=build_the_cauldron
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --mem-per-cpu=21G
#SBATCH --output=/fsx/m4/experiments/general_logs/build_the_cauldron/res%A_%a
#SBATCH --partition=hopper-prod
#SBATCH --qos high

source /fsx/m4/start-m4-user
conda activate shared-m4


python /fsx/hugo/repos/m4_28/datasets_processing_scripts/build_concatenation_datasets_sft/build_the_cauldron.py ${SLURM_ARRAY_TASK_ID}
