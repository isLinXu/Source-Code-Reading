# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `Mantis-Eval`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### Mantis-Eval

The evaluation script will automatically download the Mantis Eval dataset from HuggingFace, and the cached path is `data/mantis_eval`.

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/mantis_eval/evaluate_mantis.py --checkpoint ${CHECKPOINT} --dynamic
```

Alternatively, you can run the following simplified command:

```shell
GPUS=8 sh evaluate.sh ${CHECKPOINT} mantis --dynamic
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default         | Description                                                                                                       |
| ---------------- | ------ | --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`            | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'Mantis-Eval'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`         | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`             | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`         | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`         | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
