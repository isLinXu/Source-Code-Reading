#!/bin/bash
#SBATCH --job-name=m4_get_dataset            # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 012:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/cnw/uue59kq/logs/get_dataset/%j-%x.out   # output file name
#SBATCH --account=cnw@cpu                                               # account
#SBATCH --array=0-2756
#SBATCH --partition=cpu_p1

set -x -e

source $cnw_ALL_CCFRWORK/start-m4-user
conda activate lucile-m4

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/gpfsscratch/rech/cnw/uue59kq/to_delete

WORKING_DIR=/gpfswork/rech/cnw/uue59kq/repos/m4/m4/sourcing/processing/extracting_documents/get_modelling_metadata_dataset
pushd $WORKING_DIR

readarray -t SHARD_NAMES < shard_names.txt
SHARD_NAME=${SHARD_NAMES[$SLURM_ARRAY_TASK_ID]}
echo "Downloading shard: "$SHARD_NAME

python get_modelling_metadata_dataset.py \
    --dataset-path /gpfsscratch/rech/cnw/urd43gx/c4-en-html-with-metadata/ \
    --save-dir /gpfsscratch/rech/cnw/commun/local_datasets/c4-en-html-with-metadata-arrow/ \
    --shard-name $SHARD_NAME
