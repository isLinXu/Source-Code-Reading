#!/bin/bash
#SBATCH --job-name=tr_312-slurm-status
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/m4/experiments/local_experiment_dir/tr_312/logs/crons/%x-%j.out

set -e


### EDIT ME START ###

# how often to try to run the checkpoint upload - hint: approximately as often as a checkpoint is saved
RUN_FREQUENCY_IN_HOURS=1

CONDA_ENV_NAME=shared-m4-2024-05-28-copy3

EXPERIMENT_NAME=tr_312

### EDIT ME END ###


# ----------------- Auto-Workdir -----------------
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
M4_REPO_PATH=$(builtin cd $SCRIPT_DIR/../../../../; pwd)
# --------------------------------------------------

PATH_TO_THIS_FILE=$SCRIPT_PATH
#echo "The absolute path of the current script file is: $PATH_TO_THIS_FILE"
RUN_NAME=$(basename $(dirname ${PATH_TO_THIS_FILE}))
SAVE_DIR="/fsx/m4/experiments/local_experiment_dir/$RUN_NAME"

echo "START TIME: $(date)"

source /fsx/m4/start-m4-user
conda activate base
conda activate $CONDA_ENV_NAME

pushd $M4_REPO_PATH
#export PYTHONPATH=$WORKING_DIR:$PYTHONPATH
cd $M4_REPO_PATH/experiments/pretraining/vloom/$EXPERIMENT_NAME

# ensure to restart self first
echo scheduling to run again in $RUN_FREQUENCY_IN_HOURS hours
sbatch --begin=now+${RUN_FREQUENCY_IN_HOURS}hour slurm-status.slurm

echo "running slurm status"

$M4_REPO_PATH/watchdogs/slurm-status.py --job-name $EXPERIMENT_NAME | tee -a $SAVE_DIR/logs/main_log.txt

echo "END TIME: $(date)"
