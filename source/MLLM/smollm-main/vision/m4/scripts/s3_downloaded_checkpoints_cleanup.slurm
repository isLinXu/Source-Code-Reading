#!/bin/bash
#SBATCH --job-name=tr_test-s3-cleanup-checkpoints
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/logs/%x-%j.out


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
M4_REPO_PATH=$(builtin cd $SCRIPT_DIR/../../; pwd)

# --------------------------------------------------

### EDIT ME START ###

CONDA_ENV_NAME=shared-m4

EXPERIMENT_NAME=tr_194_laion_cm4_mix

opt_step_num_list=(
   "1000"
   "2000"
)

### EDIT ME END ###


echo "START TIME: $(date)"

source /fsx/m4/start-m4-user
conda activate base
conda activate $CONDA_ENV_NAME
pushd $M4_REPO_PATH
export PYTHONPATH=$WORKING_DIR:$PYTHONPATH

for opt_step_num in ${opt_step_num_list[@]}
do
    OPT_STEP_DIR="/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/${EXPERIMENT_NAME}/opt_step-${opt_step_num}"
    rm -r $OPT_STEP_DIR
    echo "Deleted $OPT_STEP_DIR of experiment: $EXPERIMENT_NAME"
done

echo "END TIME: $(date)"
