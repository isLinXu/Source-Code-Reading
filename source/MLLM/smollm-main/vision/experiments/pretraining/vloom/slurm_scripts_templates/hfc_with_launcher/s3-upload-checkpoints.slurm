#!/bin/bash
#SBATCH --job-name=s3-upload-checkpoints
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/m4/experiments/local_experiment_dir/$RUN_NAME/logs/%x-%j.out


set -e

### EDIT ME START ###

# how often to try to run the checkpoint upload - hint: approximately as often as a checkpoint is saved
RUN_FREQUENCY_IN_HOURS=3

CONDA_ENV_NAME=shared-m4

M4_REPO_PATH=/fsx/m4/repos/m4
EXPERIMENT_NAME=tr_184_xxx

### EDIT ME END ###


echo "START TIME: $(date)"

source /fsx/m4/start-m4-user
conda activate base
conda activate $CONDA_ENV_NAME

# ensure to restart self first
echo scheduling to run again in $RUN_FREQUENCY_IN_HOURS hours
sbatch --begin=now+${RUN_FREQUENCY_IN_HOURS}hour s3-upload-checkpoints.slurm

echo "running checkpoint converter"

M4_CHECKPOINTS_PATH=/fsx/m4/experiments/local_experiment_dir/${EXPERIMENT_NAME}

python -u $M4_REPO_PATH/m4/scripts/s3-upload-checkpoints.py $M4_CHECKPOINTS_PATH

echo "END TIME: $(date)"
