#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --no-requeue
#SBATCH --time 20:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lucile@huggingface.co

set -x -e

source $cnw_ALL_CCFRWORK/start-m4-user

conda activate $CONDA_ENV_NAME

pushd $WORKING_DIR

module purge
module load cuda/11.4.3
module load git

python -c 'import torch; cuda=torch.version.cuda; assert cuda.startswith("11"), f"cuda-11.x is needed for bf16, got {cuda}"'

# We are on an offline partition
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export WANDB_DIR=$cnw_ALL_CCFRSCRATCH/experiments


GIT_PYTHON_GIT_EXECUTABLE=`which git`
export GIT_PYTHON_GIT_EXECUTABLE

export PYTHONPATH=$WORKING_DIR:$PYTHONPATH

JZ_JOB_TILE_SEC=71100 # This is 19h45 in seconds
BASE_DATA_PATH="$cnw_ALL_CCFRSCRATCH/general_pmd/image/"
SAVE_DIR="$cnw_ALL_CCFRSCRATCH/experiments/local_experiment_dir/$RUN_NAME"
CONFIG_FILE="$TRAINING_CONFIGS_DIR/config.yaml"
ACCELERATE_CONFIG_FILE="$TRAINING_CONFIGS_DIR/accelerate_config.yaml"
DEEPSPEED_CONFIG_FILE="$TRAINING_CONFIGS_DIR/ds_config.json"

mkdir -p $SAVE_DIR

if [[ ! -f $CONFIG_FILE ]] ; then
    echo "File ${CONFIG_FILE} is not there, aborting."
    exit
fi

if [[ ! -f $ACCELERATE_CONFIG_FILE ]] ; then
    echo "File $ACCELERATE_CONFIG_FILE is not there, aborting."
    exit
fi

MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
# From https://i.hsfzxjy.site/2021-03-10-obtain-a-random-unused-tcp-port-with-bash/
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
MASTER_PORT=$(unused_port)

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

NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))

export LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_GPUS \
    --deepspeed_config_file $DEEPSPEED_CONFIG_FILE \
    "

# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="m4/training/main.py \
        --config $CONFIG_FILE \
        --training_datasets_paths ${all_train_shards[@]} \
        --validation_datasets_paths ${all_validation_shards[@]} \
        --job_id $SLURM_JOB_ID \
        --jz_job_time_sec $JZ_JOB_TILE_SEC \
        --save_dir ${SAVE_DIR} \
    "

if [ ! -z "$GRAD_ACC" ]
then
    PROGRAM="${PROGRAM} --grad_acc_size ${GRAD_ACC}"
    LAUNCHER="${LAUNCHER} --gradient_accumulation_steps ${GRAD_ACC}"
fi


export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $SAVE_DIR/logs/main_log.txt
