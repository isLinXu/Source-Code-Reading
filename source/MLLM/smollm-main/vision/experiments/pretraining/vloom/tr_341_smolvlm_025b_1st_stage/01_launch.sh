# ----------------- Functions -----------------
joinByChar() {
  local IFS="$1"
  shift
  echo "$*"
}
# ---------------------------------------------

# ----------------- Auto-Variables -----------------
unset ID_JOB

SCRIPT_RELATIVE_PATH="${BASH_SOURCE[0]}"
PATH_TO_THIS_FILE=$(realpath "$SCRIPT_RELATIVE_PATH")
echo "The absolute path of the current script file is: $PATH_TO_THIS_FILE"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORKING_DIR=$(builtin cd $SCRIPT_DIR/../../../../; pwd)
echo "Working dir is: $WORKING_DIR"

cd $WORKING_DIR
# --------------------------------------------------

# ---------------- HFC hyperparameters ----------------
MAX_NUM_GPUS_PER_NODE=8
MAX_NUM_CPUS_PER_TASK=88
NUM_CPUS_PER_GPU=11
# -------------------------------------------------------

# ---------------- SLURM hyperparameters to often change ----------------
# You might want to change this
NUM_NODES=16
NUM_GPUS_PER_NODE=8
NUM_HOURS=240
# -----------------------------------------------------------------------


# ----------------- Auto variables and checks -----------------
NUM_CPUS_PER_TASK=$((NUM_CPUS_PER_GPU * NUM_GPUS_PER_NODE))

JOB_TIME_SEC=$((NUM_HOURS * 60 * 60 - 15*60))
# Check if the number of GPUs per node is valid
if [ $NUM_GPUS_PER_NODE -gt $MAX_NUM_GPUS_PER_NODE ]; then
    echo "The number of GPUs per node is greater than the maximum number of GPUs per node"
    continue
fi

# Check if the number of CPUs per task is valid
if [ $NUM_CPUS_PER_TASK -gt $MAX_NUM_CPUS_PER_TASK ]; then
    echo "The number of CPUs per task is greater than the maximum number of CPUs per task"
    continue
fi
# -------------------------------------------------------------

# ----------------- Environment variables to rarely change -----------------

# The name of the run is the name of the directory containing this file
RUN_NAME=$(basename $(dirname ${PATH_TO_THIS_FILE}))

OUTPUT_DIR="/fsx/m4/experiments/local_experiment_dir/$RUN_NAME/logs"
OUTPUT_FILE="$OUTPUT_DIR/%x_%j.out"
mkdir -p $OUTPUT_DIR

OUTPUT_DIR_CRONS="$OUTPUT_DIR/crons"
mkdir -p $OUTPUT_DIR_CRONS

SAVE_DIR="/fsx/m4/experiments/local_experiment_dir/$RUN_NAME"

# important: once the job started it'll continually re-use the same environment it detected `shared-m4` symlink resolved to
CONDA_ENV_NAME=shared-m4-2024-05-28-copy3
#if [[ ! -e "$CONDA_ENV_NAME" ]]; then
#    CONDA_TARGET=$(readlink -f /fsx/m4/conda/shared-m4)
#    ln -ns $CONDA_TARGET $CONDA_ENV_NAME
#fi


TRAINING_CONFIGS_DIR="experiments/pretraining/vloom/$RUN_NAME"
TRAINING_SLURM_FILE="$TRAINING_CONFIGS_DIR/train.slurm"

ENV_VARIABLES=(
"ALL"
"CONDA_ENV_NAME=$CONDA_ENV_NAME"
"WORKING_DIR=$WORKING_DIR"
"RUN_NAME=$RUN_NAME"
"SWEEP_NAME=$SWEEP_NAME"
"TRAINING_CONFIGS_DIR=$TRAINING_CONFIGS_DIR"
"NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE"
"JOB_TIME_SEC=$JOB_TIME_SEC"
)
EXPORTED_ENV_VARIABLES=$(joinByChar ',' ${ENV_VARIABLES[@]})

CMD=" \
--job-name=$RUN_NAME \
--nodes=$NUM_NODES \
--gres=gpu:$NUM_GPUS_PER_NODE \
--output=$OUTPUT_FILE \
--cpus-per-task=$NUM_CPUS_PER_TASK \
--export=$EXPORTED_ENV_VARIABLES \
--time=$NUM_HOURS:00:00 \
--requeue \
--qos high \
$TRAINING_SLURM_FILE
"
# ----------------------------------------------------------

# ----------------- Logic to add a dependency to the previous job -----------------
# We check if there is already a job with the same name in the queue and we get its id
# If such a job exists, we add a dependency to the current job

# read input from stdin and store it in a variable
input=$(squeue -u `whoami` -o "%.16i %.9P %.90j %.8T %.10M %.8l %.6D %.20S %R")

# define the target value for column "C"
target=$RUN_NAME

# initialize the values array for each iteration
values=()

# loop through each line of the input
while read -r line; do
# split the line into an array using space as the delimiter
arr=($line)

# check if the value in the 3rd column (index 2) is equal to the target value
if [ "${arr[2]}" == "$target" ]; then
    # store the value in the 1st column (index 0)
    values+=(${arr[0]})
fi
done <<< "$input"

# sort the values in descending order and store the result in a variable
ID_JOB_DEPENDENCY=$(printf "%s\n" "${values[@]}" | sort -nr -n | head -n 1)

#if [ ! -z "$ID_JOB_DEPENDENCY" ]
#then
#    echo "The job $RUN_NAME will depend on the job $ID_JOB_DEPENDENCY"
#    CMD="-d ${ID_JOB_DEPENDENCY} ${CMD}"
#fi
# --------------------------------------------------------------------------------

#CMD="--nodelist ip-26-0-144-[95,156,189,236],ip-26-0-145-[137,152,154-155,193],ip-26-0-146-[1,4,15,27,33,100,122-123,149,183,212,249],ip-26-0-147-[6,94,120,135,141,187,189,193,204,233,245,247],ip-26-0-148-[55,93,115,151,170,193,245],ip-26-0-149-[1,47,73,84,105,133,194,199],ip-26-0-152-[71,101,119,157,178,186,207,211],ip-26-0-153-[6,62,112,132,166,251],ip-26-0-154-[16,38] ${CMD}"

# ----------------- Launch the job -----------------
ID_JOB=$(sbatch $CMD)

# We get the id of the previous job array
ID_JOB=${ID_JOB##* }
echo $ID_JOB # e.g. "773935"
# --------------------------------------------------
