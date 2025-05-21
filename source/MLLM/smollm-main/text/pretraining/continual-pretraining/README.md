# Continual Pretraining
We use [nanotron](https://github.com/huggingface/nanotron/) library to do continual pretraining.

## Setup

Please refer to [nanotron](https://github.com/huggingface/nanotron/) for detailed instructions on setting up your training environment and launching jobs and [smollm/pre-training](https://github.com/huggingface/smollm/tree/main/pre-training) for and example with the pre-training scripts.

## Usage

The nanotron checkpoints for SmolLM2 models are available at: https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt. 

## Example: Finemath
For finemath, we did continual pretraining of llama3-3B with different data mixtures. Here we will detail the steps to do the same.

### Nanotron
For this example, you need to switch to this [PR](https://github.com/huggingface/nanotron/pull/255)
```
gh pr checkout 255
```

### Data
First step is to tokenize the datasets. To do this, we use the [datatrove](https://github.com/huggingface/datatrove) library. We tokenized the following datasets with the llama3 tokenizer:
- [HuggingFaceTB/smollm-corpus/fineweb-edu-dedup](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/fineweb-edu-dedup)
- [HuggingFaceTB/finemath/finemath-3plus](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/finemath-3plus)
- [HuggingFaceTB/finemath/finemath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/finemath-4plus)
- [HuggingFaceTB/finemath/infiwebmath-3plus](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/infiwebmath-3plus)
- [HuggingFaceTB/finemath/infiwebmath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/infiwebmath-4plus)
- [Infi-MM/InfiMM-WebMath-40B](https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B)
- [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)

You can find an example of how to tokenize the datasets in the `finemath/finemath-tokenize.py` script. You might encounter some issues with the tokenization, you can apply the following patches:
- For `Infi-MM/InfiMM-WebMath-40B`: `finemath/tokenization_InfiMM-WebMath-4OB.patch`
- For others: `finemath/tokenization_finemath.patch`
To apply the patch, install datatrove from source and run `git apply <path_to_patch>.patch` in the datatrove directory.

### Training
Once the dataset are tokenized, you can launch the training with a similar script as the one in [smollm/pre-training](https://github.com/huggingface/smollm/tree/main/pre-training). When resuming a training from a checkpoint, you have the choice to keep the learning rate scheduler and optimizer state by changing the following parameters in the yaml file:
- `load_lr_scheduler: false`
- `load_optimizer: false`

### Evaluation

For evaluation, you can follow the instructions in [smollm/evaluation](https://github.com/huggingface/smollm/tree/main/evaluation#finemath-dataset-ablations).
