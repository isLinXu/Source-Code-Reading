#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --nodes=1
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 00:05:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cnw/commun/experiments/generation_dir/logs/%x_%j.out
#SBATCH --account=cnw@cpu

source $cnw_ALL_CCFRWORK/start-m4-user

CONDA_ENV_NAME="leo"

conda activate $CONDA_ENV_NAME

# We are on an offline partition
WORKING_DIR=$WORK/code/m4_runs_5
pushd $WORKING_DIR

GIT_PYTHON_GIT_EXECUTABLE=`which git`
export GIT_PYTHON_GIT_EXECUTABLE

CONFIG="m4/evaluation/generation/generate_config.yaml"
JID_JOB=$(sbatch --job-name=make_generation_$SLURM_JOB_ID --export=ALL,CONDA_ENV_NAME=$CONDA_ENV_NAME,WORKING_DIR=$WORKING_DIR,CONFIG=$CONFIG m4/evaluation/generation/make_generation.slurm)

JID_JOB=${JID_JOB##* }

echo $JID_JOB

GEN_FILE="/gpfsscratch/rech/cnw/commun/experiments/generation_dir/generation_tmp_files_dir/gen_${JID_JOB}.json"

echo $GEN_FILE

sbatch --dependency=afterok:$JID_JOB --job-name=log_generation_$SLURM_JOB_ID --export=ALL,CONDA_ENV_NAME=$CONDA_ENV_NAME,WORKING_DIR=$WORKING_DIR,GEN_FILE=$GEN_FILE m4/evaluation/generation/log_generation.slurm
