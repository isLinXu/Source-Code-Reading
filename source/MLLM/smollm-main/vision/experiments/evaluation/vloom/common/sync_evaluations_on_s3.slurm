#!/bin/bash
#SBATCH --job-name=s3_sync_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1            # number of cores per tasks
#SBATCH --time 00:05:00              # maximum execution time (HH:MM:SS)
#SBATCH --mem-per-cpu=11G
#SBATCH --output=/fsx/m4/evals/run_eval_master/logs/%x_%j.out
set -x -e

source $cnw_ALL_CCFRWORK/start-m4-user
BASENAME_EVALUATION_JSONL_FILE="$(basename $EVALUATION_JSONL_FILE)"
aws s3 cp $EVALUATION_JSONL_FILE s3://m4-exps/eval_results/$BASENAME_EVALUATION_JSONL_FILE
