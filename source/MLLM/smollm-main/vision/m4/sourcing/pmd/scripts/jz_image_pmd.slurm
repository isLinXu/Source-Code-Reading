#!/bin/bash
#SBATCH --job-name=jz_image_pmd
#SBATCH --qos=qos_cpu-t4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread
#SBATCH --array=0-1%2
#SBATCH --time=100:00:00
#SBATCH --output=output-%x-%A_%a.out
#SBATCH --error=output-%x-%A_%a.out
#SBATCH --account=cnw@cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor@huggingface.co

source ~/.bashrc_cnw
eval "$(conda shell.bash hook)"
conda activate victor

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$cnw_ALL_CCFRSCRATCH/datasets

all_dataset=(
jz_wit
jz_conceptual_captions
)
dataset_name=${all_dataset[${SLURM_ARRAY_TASK_ID}]}

python jz_pmd.py --dataset_name $dataset_name --loading_script_path $WORK/code/m4/m4/sourcing/pmd/jz_loaders/$dataset_name
