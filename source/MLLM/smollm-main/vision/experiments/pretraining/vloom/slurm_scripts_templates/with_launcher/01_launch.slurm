#!/bin/bash
#SBATCH --job-name=launch_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1           # number of cores per tasks
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --time 0:30:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cnw/commun/experiments/general_logs/%x-%j.out           # output file name
#SBATCH --account=cnw@cpu
set -x -e

NUM_GPUS_PER_NODE=1
NUM_CPUS_PER_TASK=10
GRAD_ACC=4
NUM_NODES=1

PATH_TO_THIS_FILE=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')

# The name of the run is the name of the directory containing this file
RUN_NAME_SHORT=$(basename $(dirname ${PATH_TO_THIS_FILE}))

# This is a toy example to show how we can do several training runs with different parameters by using the same directory
# This is in particular useful for sweeps with hyperparameter search
RUN_NAME="${RUN_NAME_SHORT}_v1"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORKING_DIR=$(builtin cd $SCRIPT_DIR/../../../../; pwd)
echo "Working dir is: $WORKING_DIR"

CONDA_ENV_NAME=m4-user

# The output directory contains the logs of the training will be created by this script
OUTPUT_DIR="$cnw_ALL_CCFRSCRATCH/experiments/local_experiment_dir/$RUN_NAME/logs"
OUTPUT_FILE="$OUTPUT_DIR/%x_%j.out"
mkdir -p $OUTPUT_DIR

pushd $WORKING_DIR

# You'll probably need to change the following line to point to "experiments/pretraining/vloom/$RUN_NAME_SHORT"
TRAINING_CONFIGS_DIR="experiments/pretraining/vloom/slurm_scripts_templates/$RUN_NAME_SHORT"
TRAINING_SLURM_FILE="$TRAINING_CONFIGS_DIR/train.slurm"

CMD=" \
--array=1-10%1 \
--job-name=$RUN_NAME \
--nodes=$NUM_NODES \
--gres=gpu:$NUM_GPUS_PER_NODE \
--output=$OUTPUT_FILE \
--cpus-per-task=$NUM_CPUS_PER_TASK \
--constraint=v100-32g \
--account=cnw@v100 \
--export=ALL,CONDA_ENV_NAME=$CONDA_ENV_NAME,WORKING_DIR=$WORKING_DIR,RUN_NAME=$RUN_NAME,TRAINING_CONFIGS_DIR=$TRAINING_CONFIGS_DIR,NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE,GRAD_ACC=$GRAD_ACC \
$TRAINING_SLURM_FILE
"

sbatch $CMD
