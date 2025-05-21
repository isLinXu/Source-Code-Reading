#!/bin/bash
#SBATCH --job-name=template-merge-lora-and-resize-eou
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=3:00:00
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/m4/experiments/general_logs/merge_loras_and_resize_eou/%x-%j.out
#SBATCH --qos=high

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
CONDA_ENV_NAME="shared-m4-2024-05-28-copy3"
OPT_STEP_DIR="/fsx/m4/experiments/local_experiment_dir/tr_343_vsmollm2_05b/opt_step-24750"
OUTPUT_DIR="/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_343_vsmollm2_05b_opt_step_24750_merge_and_resize_eou"

source /fsx/m4/start-m4-user
conda activate base
conda activate $CONDA_ENV_NAME

python m4/scripts/merge_lora_and_save.py $OPT_STEP_DIR $OUTPUT_DIR
python experiments/pretraining/vloom/tr_343_vsmollm2_05b/resize_embed_for_eou.py $OUTPUT_DIR
echo "Done"

