# Fine-tuning

## SmolLM2 Instruct

We build the SmolLM2 Instruct family by finetuning the base 1.7B on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) and the base 360M and 135M models on [Smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) using `TRL` and the alignement handbook and then doing DPO on [UltraFeedBack](https://huggingface.co/datasets/openbmb/UltraFeedback). You can find the scipts and instructions for dohere: https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2#instructions-to-train-smollm2-17b-instruct 

## Custom script
Here, we provide a simple script for finetuning SmolLM2. In this case, we fine-tune the base 1.7B on python data.

### Setup

Install `pytorch` [see documentation](https://pytorch.org/), and then install the requirements 
```bash
pip install -r requirements.txt
```

Before you run any of the scripts make sure you are logged in `wandb` and HuggingFace Hub to push the checkpoints, and you have `accelerate` configured:
```bash
wandb login
huggingface-cli login
accelerate config
``` 
Now that everything is done, you can clone the repository and get into the corresponding directory.

```bash
git clone https://github.com/huggingface/smollm
cd smollm/finetune
```

### Training
To fine-tune efficiently with a low cost, we use [PEFT](https://github.com/huggingface/peft) library for Low-Rank Adaptation (LoRA) training. We also use the `SFTTrainer` from [TRL](https://github.com/huggingface/trl).

For this example, we will fine-tune SmolLM1-1.7B on the `Python` subset of [the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol). This is just for illustration purposes.

To launch the training:
```bash
accelerate launch train.py \
        --model_id "HuggingFaceTB/SmolLM2-1.7B" \
        --dataset_name "bigcode/the-stack-smol" \
        --subset "data/python" \
        --dataset_text_field "content" \
        --split "train" \
        --max_seq_length 2048 \
        --max_steps 5000 \
        --micro_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 3e-4 \
        --warmup_steps 100 \
        --num_proc "$(nproc)"
```

If you want to fine-tune on other text datasets, you need to change `dataset_text_field` argument to the name of the column containing the code/text you want to train on.


