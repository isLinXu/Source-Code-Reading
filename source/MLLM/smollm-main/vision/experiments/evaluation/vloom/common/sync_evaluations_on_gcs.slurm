#!/bin/bash
#SBATCH --job-name=gcs_sync_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_cpu-dev
#SBATCH --partition=compil
#SBATCH --cpus-per-task=1            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 00:05:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cnw/commun/experiments/local_experiment_dir/evals/run_eval_master/logs/%x_%j.out
#SBATCH --account=cnw@cpu
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT,TIME_LIMIT
#SBATCH --mail-user=hf-m4-jz@googlegroups.com
#SBATCH --no-requeue

set -x -e

source $cnw_ALL_CCFRWORK/start-m4-user

gsutil cp $EVALUATION_JSONL_FILE gs://hf-science-m4-cold/local_experiment_dir/evals/results/
