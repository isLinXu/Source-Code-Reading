#!/bin/bash
#SBATCH --job-name=make_generation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition gpu_p2
#SBATCH --gres=gpu:1                # number of gpu
#SBATCH --cpus-per-task=2           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 00:30:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cnw/commun/experiments/generation_dir/logs/%x_%j.out
#SBATCH --account=cnw@gpu

source $cnw_ALL_CCFRWORK/start-m4-user

conda activate leo-flash

# We are on an offline partition
export TRANSFORMERS_OFFLINE=1

pushd $WORKING_DIR
GIT_PYTHON_GIT_EXECUTABLE=`which git`
export GIT_PYTHON_GIT_EXECUTABLE

echo "model paths:"
echo $MODEL_PATHS
python m4/evaluation/generation/launch_generation.py  --load_config $CONFIG --job_id $SLURM_JOB_ID
