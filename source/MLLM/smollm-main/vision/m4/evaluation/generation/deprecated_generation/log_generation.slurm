#!/bin/bash
#SBATCH --job-name=make_generation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition prepost
#SBATCH --cpus-per-task=1         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cnw/commun/experiments/generation_dir/logs/%x_%j.out
#SBATCH --account=cnw@cpu

source $cnw_ALL_CCFRWORK/start-m4-user

conda activate $CONDA_ENV_NAME

export WANDB_DIR=$cnw_ALL_CCFRSCRATCH/experiments

pushd $WORKING_DIR

GIT_PYTHON_GIT_EXECUTABLE=`which git`
export GIT_PYTHON_GIT_EXECUTABLE

python m4/evaluation/generation/log_generation.py  --gen_file $GEN_FILE
