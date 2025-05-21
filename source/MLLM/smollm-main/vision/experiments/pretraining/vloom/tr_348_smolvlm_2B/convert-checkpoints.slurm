#!/bin/bash
#SBATCH --job-name=tr_348-convert-checkpoints
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
#SBATCH --partition=hopper-prod
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

# how often to try to run the checkpoint conversion - hint: approximately as often as a checkpoint is saved
RUN_FREQUENCY_IN_HOURS=4

CONDA_ENV_NAME=shared-m4-2024-05-28-copy3

EXPERIMENT_NAME=tr_348_smolvlm_2B_token_fix

### EDIT ME END ###


echo "START TIME: $(date)"

source /fsx/m4/start-m4-user
conda activate base
conda activate $CONDA_ENV_NAME

CUDA_VERSION=11.8
export CUDA_HOME=/usr/local/cuda-11.8
export FI_EFA_ENABLE_SHM_TRANSFER=1
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/efa/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/cuda-${CUDA_VERSION}/lib/libnccl.so

NCCL_TEST_PATH=/usr/local/cuda-${CUDA_VERSION}/efa/test-cuda-${CUDA_VERSION}
MPI_PATH=/opt/amazon/openmpi

#export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-${CUDA_VERSION}/efa/lib:/usr/local/cuda-${CUDA
#_VERSION}/lib:/usr/local/cuda-${CUDA_VERSION}/lib64:/usr/local/cuda-${CUDA_VERSION}:/usr/local/cuda-${CUDA_VERSION}/extras/CUPTI/lib64:/usr/local/cuda-
#${CUDA_VERSION}/targets/x86_64-linux/lib

export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4de
# export NCCL_ALGO=Ring
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export NCCL_SOCKET_IFNAME=enp
#export FI_EFA_ENABLE_SHM_TRANSFER=1



pushd $M4_REPO_PATH
export PYTHONPATH=$WORKING_DIR:$PYTHONPATH
cd $M4_REPO_PATH/experiments/pretraining/vloom/$EXPERIMENT_NAME

# ensure to restart self first
echo scheduling to run again in $RUN_FREQUENCY_IN_HOURS hours
sbatch --begin=now+${RUN_FREQUENCY_IN_HOURS}hour convert-checkpoints.slurm

echo "running checkpoint converter"

M4_CHECKPOINTS_PATH=/fsx/m4/experiments/local_experiment_dir/${EXPERIMENT_NAME}

python -u $M4_REPO_PATH/m4/scripts/convert-checkpoints.py $M4_CHECKPOINTS_PATH --force

echo "END TIME: $(date)"
