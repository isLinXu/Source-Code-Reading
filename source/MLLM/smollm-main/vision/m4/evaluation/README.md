TODO: tune the values of `batch_size` and `mini_batch_size`

# Clip

## Zero-shot

```bash
conda activate m4-eval
commit_hash=`git rev-parse HEAD`

python m4/evaluation/launch.py \
    --commit_hash $commit_hash \
    --batch_size 16 \
    --mini_batch_size 1024 \
    --do_tasks Cifar10ClipZeroShoterEnsembleAcc
```

# Clip

## Linear probe

```bash
conda activate m4-eval
commit_hash=`git rev-parse HEAD`

python m4/evaluation/launch.py \
    --commit_hash $commit_hash \
    --batch_size 16 \
    --mini_batch_size 1024 \
    --do_tasks Cifar10ClipLinearProberAcc
```

# VGPT2

## Zero-shot

```bash
conda activate m4-eval
commit_hash=`git rev-parse HEAD`

python m4/evaluation/launch.py \
    --commit_hash $commit_hash \
    --batch_size 64 \
    --mini_batch_size 1024 \
    --tokenizer_name $ALL_CCFRSCRATCH/experiments/local_experiment_dir/tr_04/opt_step-3766/tokenizer/ \
    --model_name $ALL_CCFRSCRATCH/experiments/local_experiment_dir/tr_04/opt_step-3766/unwrapped_model/ \
    --do_tasks Cifar10Vgpt2ZeroShoterAcc
```

## Few-shot

```bash
conda activate m4-eval
commit_hash=`git rev-parse HEAD`

python m4/evaluation/launch.py \
    --commit_hash $commit_hash \
    --batch_size 64 \
    --mini_batch_size 1024 \
    --tokenizer_name $ALL_CCFRSCRATCH/experiments/local_experiment_dir/tr_04/opt_step-3766/tokenizer/ \
    --model_name $ALL_CCFRSCRATCH/experiments/local_experiment_dir/tr_04/opt_step-3766/unwrapped_model/ \
    --do_tasks Cifar10Vgpt2FewShoterAccWithKLAndEntropy \
    --num_shots 5 \
    --shot_selection_mode rices
```

# Multi-GPU Evaluation

To run multi-gpu evaluation, simply launch the above command using `accelerate` cli. Example below:

```bash
 accelerate launch --num_processes 2 --multi_gpu ./m4/evaluation/launch.py --batch_size 128 --mini_batch_size 4 --model_name /some/unwrapped_model --tokenizer_name /some/tokenizer --do_tasks Cifar10SampleVgpt2ZeroShoterAccWithKLAndEntropy --save_to_jsonl some.jsonl
 ```
