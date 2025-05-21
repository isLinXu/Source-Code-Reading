#!/bin/bash
#SBATCH --job-name=run_eval_automatic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_cpu-dev
#SBATCH --partition=compil
#SBATCH --cpus-per-task=4            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cnw/commun/experiments/local_experiment_dir/evals/run_eval_master/logs/%x_%j.out
#SBATCH --account=cnw@cpu
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT,TIME_LIMIT
#SBATCH --mail-user=hf-m4-jz@googlegroups.com
#SBATCH --no-requeue

set -x -e

source $cnw_ALL_CCFRWORK/start-m4-user

conda activate $CONDA_ENV_NAME

pushd $WORKING_DIR

python m4/evaluation/scripts/sync_evaluations_on_wandb.py \
    --evaluation_jsonl_files $EVALUATION_JSONL_FILE \
    --run_name_to_log $RUN_NAME
