#!/bin/bash
#SBATCH --job-name=tr_348-s3-upload-checkpoints
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --partition=hopper-cpu
#SBATCH --output=/fsx/m4/experiments/local_experiment_dir/tr_348_smolvlm_2B_token_fix/logs/crons/%x-%j.out
#SBATCH --qos high


set -e

# ----------------- Auto-Workdir -----------------
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
M4_REPO_PATH=$(builtin cd $SCRIPT_DIR/../../../../; pwd)
# --------------------------------------------------

### EDIT ME START ###

# how often to try to run the checkpoint upload - hint: approximately as often as a checkpoint is saved
RUN_FREQUENCY_IN_HOURS=8

CONDA_ENV_NAME=shared-m4-2024-05-28-copy3

EXPERIMENT_NAME=tr_348_smolvlm_2B_token_fix

### EDIT ME END ###


echo "START TIME: $(date)"

source /fsx/m4/start-m4-user
conda activate base
conda activate $CONDA_ENV_NAME

pushd $M4_REPO_PATH
export PYTHONPATH=$WORKING_DIR:$PYTHONPATH
cd $M4_REPO_PATH/experiments/pretraining/vloom/$EXPERIMENT_NAME

# ensure to restart self first
echo scheduling to run again in $RUN_FREQUENCY_IN_HOURS hours
sbatch --begin=now+${RUN_FREQUENCY_IN_HOURS}hour s3-upload-checkpoints.slurm

echo "running s3 checkpoint upload"

M4_CHECKPOINTS_PATH=/fsx/m4/experiments/local_experiment_dir/${EXPERIMENT_NAME}

python -u $M4_REPO_PATH/m4/scripts/s3-upload-checkpoints.py $M4_CHECKPOINTS_PATH

echo "END TIME: $(date)"
