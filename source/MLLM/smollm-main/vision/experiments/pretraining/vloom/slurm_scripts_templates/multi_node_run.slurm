#!/bin/bash
#SBATCH --job-name=slurm_scripts_templates_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                 # number of gpu
#SBATCH --constraint=v100-32g
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --no-requeue
#SBATCH --time=20:00:00
#SBATCH --output=/gpfsscratch/rech/cnw/commun/experiments/local_experiment_dir/slurm_scripts_templates/logs/%x_%j.out
#SBATCH --account=cnw@v100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hf-m4-jz@googlegroups.com

set -x -e

source $cnw_ALL_CCFRWORK/start-m4-user

RUN_NAME="slurm_scripts_templates"
JZ_JOB_TILE_SEC=71100 # This is 19h45 in seconds
BASE_DATA_PATH="$cnw_ALL_CCFRSCRATCH/general_pmd/image/"
SAVE_DIR="$cnw_ALL_CCFRSCRATCH/experiments/local_experiment_dir/$RUN_NAME"
CONFIG_FILE="experiments/pretraining/vloom/$RUN_NAME/config.yaml"
ACCELERATE_CONFIG_FILE="experiments/pretraining/vloom/$RUN_NAME/accelerate_config_multi_node.yaml"

# important: once the job started it'll continually re-use the same environment it detected `shared-m4` symlink resolved to
LOCAL_CONDA_SYMLINK=shared-m4-env
if [[ ! -e "$LOCAL_CONDA_SYMLINK" ]]; then
    CONDA_TARGET=$(readlink -f /gpfswork/rech/cnw/commun/conda/shared-m4)
    ln -ns $CONDA_TARGET $LOCAL_CONDA_SYMLINK
fi
conda activate $LOCAL_CONDA_SYMLINK


# We are on an offline partition
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export WANDB_DIR=$cnw_ALL_CCFRSCRATCH/experiments

WORKING_DIR=$cnw_ALL_CCFRWORK/repos/m4
pushd $WORKING_DIR

GIT_PYTHON_GIT_EXECUTABLE=`which git`
export GIT_PYTHON_GIT_EXECUTABLE

export PYTHONPATH=$WORKING_DIR:$PYTHONPATH


if [[ ! -f $CONFIG_FILE ]] ; then
    echo "File ${CONFIG_FILE} is not there, aborting."
    exit
fi

if [[ ! -f $ACCELERATE_CONFIG_FILE ]] ; then
    echo "File $ACCELERATE_CONFIG_FILE is not there, aborting."
    exit
fi

train_subsets=(
"coco"
"conceptual_12m"
"conceptual_captions"
"localized_narratives__ADE20k"
"localized_narratives__coco"
"localized_narratives__flickr30k"
"localized_narratives__openimages"
"red_caps"
# "sbu_captions"
"visual_genome"
"wit"
"yfcc100m"
)
validation_subsets=(
"coco"
)

all_train_shards=()
for subset in ${train_subsets[@]}
do
    all_train_shards+=($(dirname $(ls -d $BASE_DATA_PATH/$subset/train/*/dataset.arrow)))
done
all_validation_shards=()
for subset in ${validation_subsets[@]}
do
    all_validation_shards+=($(dirname $(ls -d $BASE_DATA_PATH/$subset/validation/*/dataset.arrow)))
done

pip freeze > $SAVE_DIR/${SLURM_JOB_ID}_requirements.txt

MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
MASTER_PORT=6001 # Most of the ports are available (and if not, there will be an error thrown out)

export LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM=" \
    m4/training/main.py \
        --config $CONFIG_FILE \
        --training_datasets_paths ${all_train_shards[@]} \
        --validation_datasets_paths ${all_validation_shards[@]} \
        --job_id $SLURM_JOB_ID \
        --jz_job_time_sec $JZ_JOB_TILE_SEC \
        --save_dir ${SAVE_DIR}
    "

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $SAVE_DIR/logs/main_log.txt
