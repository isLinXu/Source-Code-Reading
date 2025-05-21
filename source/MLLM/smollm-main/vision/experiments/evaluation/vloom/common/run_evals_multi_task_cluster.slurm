#!/bin/bash
#SBATCH --job-name=run_eval_automatic
#SBATCH --ntasks-per-node=1
#SBATCH --output=/fsx/m4/experiments/local_experiment_dir/evals/auto/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT,TIME_LIMIT
#SBATCH --mail-user=hf-m4-jz@googlegroups.com
#SBATCH --no-requeue

set -x -e

source $cnw_ALL_CCFRWORK/start-m4-user

conda activate $CONDA_ENV_NAME

export GIT_PYTHON_REFRESH=quiet

export CUDA_LAUNCH_BLOCKING=1

python -c 'import torch; cuda=torch.version.cuda; assert cuda.startswith("11"), f"cuda-11.x is needed for bf16, got {cuda}"'

# We are on an offline partition
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

pushd $WORKING_DIR

commit_hash=`git rev-parse HEAD`

export PYTHONPATH=$WORKING_DIR:$PYTHONPATH

IFS='[|]' read -r -a DIR_CHECKPOINTS <<< "$DIR_CHECKPOINTS_STRING"
IFS='[|]' read -r -a TASKS_TO_DO <<< "$TASKS_TO_DO_STRING"

echo "DIR_CHECKPOINTS_str: $DIR_CHECKPOINTS_STRING"
echo "TASKS_TO_DO_str: $TASKS_TO_DO_STRING"
echo "DIR_CHECKPOINTS: ${DIR_CHECKPOINTS[@]}"
echo "TASKS_TO_DO: ${TASKS_TO_DO[@]}"


TASKS_STRING_PER_CHECKPOINT=${TASKS_TO_DO[${SLURM_ARRAY_TASK_ID}]}
IFS='[/]' read -r -a TASKS_TO_DO_PER_CHECKPOINT <<< "$TASKS_STRING_PER_CHECKPOINT"
echo "TASKS_TO_DO_PER_CHECKPOINT: ${TASKS_TO_DO_PER_CHECKPOINT[@]}"

DIR_CHECKPOINT=${DIR_CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}
MODEL_DIR=$DIR_CHECKPOINT"/unwrapped_model"
TOKENIZER_DIR=$DIR_CHECKPOINT"/tokenizer"


echo $TOKENIZER_DIR
echo $MODEL_DIR
echo ${TASKS_TO_DO_PER_CHECKPOINT[@]}

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

LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_PROCESSES \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
"


PROGRAM="   m4/evaluation/launch.py \
        --commit_hash $commit_hash \
        --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
        --tokenizer_name $TOKENIZER_DIR \
        --model_name $MODEL_DIR \
        --save_to_jsonl $EVALUATION_FILE \
        --dir_path_load_from_disk $EVALUATION_LOCAL_DATASETS \
        --do_tasks ${TASKS_TO_DO_PER_CHECKPOINT[@]} \
"

if [ ! -z "$MODEL_PRECISION" ]
then
    PROGRAM="${PROGRAM} --model_precision ${MODEL_PRECISION}"
fi

if [ ! -z "$TOKENIZER_USE_FAST" ]
then
    PROGRAM="${PROGRAM} --tokenizer_use_fast ${TOKENIZER_USE_FAST}"
fi

if [ ! -z "$EVALUATION_VERSION" ]
then
    PROGRAM="${PROGRAM} --evaluation_version ${EVALUATION_VERSION}"
fi

if [ ! -z "$NUM_SHOTS" ]
then
    PROGRAM="${PROGRAM} --num_shots ${NUM_SHOTS}"
fi

if [ ! -z "$SHOT_SELECTION_MODE" ]
then
    PROGRAM="${PROGRAM} --shot_selection_mode ${SHOT_SELECTION_MODE}"
fi

if [ ! -z "$NUM_BEAMS" ]
then
    PROGRAM="${PROGRAM} --num_beams ${NUM_BEAMS}"
fi

if [ ! -z "$NO_REPEAT_NGRAM_SIZE" ]
then
    PROGRAM="${PROGRAM} --no_repeat_ngram_size ${NO_REPEAT_NGRAM_SIZE}"
fi

if [ ! -z "$MAX_NEW_TOKENS" ]
then
    PROGRAM="${PROGRAM} --max_new_tokens ${MAX_NEW_TOKENS}"
fi

if [ ! -z "$SHOW_GPU_MEM_UTIL" ]
then
    PROGRAM="${PROGRAM} --show_gpu_mem_util ${SHOW_GPU_MEM_UTIL}"
fi

if [ ! -z "$DATASET_SPLIT" ]
then
    PROGRAM="${PROGRAM} --dataset_split ${DATASET_SPLIT}"
fi

if [ ! -z "$SAVE_GENERATIONS" ]
then
    PROGRAM="${PROGRAM} --save_generations ${SAVE_GENERATIONS}"
fi

if [ ! -z $VISION_ENCODER ]
then
    PROGRAM="${PROGRAM} --vision_encoder_name ${VISION_ENCODER}"
fi

if [ ! -z "$PROMPT_TEMPLATE_ID" ]
then
    PROGRAM="${PROGRAM} --prompt_template_id ${PROMPT_TEMPLATE_ID}"
fi

if [ ! -z "$SCALE_UP_IMAGES" ]
then
    PROGRAM="${PROGRAM} --scale_up_images ${SCALE_UP_IMAGES}"
fi

if [ ! -z "$IMAGE_SIZE_AFTER_SCALING" ]
then
    PROGRAM="${PROGRAM} --image_size_after_scaling ${IMAGE_SIZE_AFTER_SCALING}"
fi

export CMD="$LAUNCHER $PROGRAM"

# makes everything very slow
#export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
#export NCCL_ASYNC_ERROR_HANDLING=1

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "
LOG_DIR="/fsx/m4/experiments/local_experiment_dir/evals/auto/logs/"

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$CMD" 2>&1 | tee -a $LOG_DIR/${SLURM_JOB_ID}_main_log.txt

# We synchronize with s3 once the evaluations are finished
BASENAME_EVALUATION_FILE="$(basename $EVALUATION_FILE)"
aws s3 cp $EVALUATION_FILE s3://m4-exps/eval_results/$BASENAME_EVALUATION_FILE
