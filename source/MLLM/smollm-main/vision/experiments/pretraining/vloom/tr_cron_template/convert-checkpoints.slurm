#!/bin/bash
#SBATCH --job-name=tr_XYZ-convert-checkpoints
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/m4/experiments/local_experiment_dir/tr_cron_template/logs/crons/%x-%j.out


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

# how often to try to run the checkpoint conversion - hint: approximately as often as a checkpoint is saved
RUN_FREQUENCY_IN_HOURS=1

CONDA_ENV_NAME=shared-m4

EXPERIMENT_NAME=tr_cron_template

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
sbatch --begin=now+${RUN_FREQUENCY_IN_HOURS}hour convert-checkpoints.slurm

echo "running checkpoint converter"

M4_CHECKPOINTS_PATH=/fsx/m4/experiments/local_experiment_dir/${EXPERIMENT_NAME}

$M4_REPO_PATH/m4/scripts/convert-checkpoints.py $M4_CHECKPOINTS_PATH

echo "END TIME: $(date)"
