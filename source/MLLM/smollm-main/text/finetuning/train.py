# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import argparse
import multiprocessing
import os

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--tokenizer_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-smol")
    parser.add_argument("--subset", type=str, default="data/python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", type=bool, default=False)
    parser.add_argument("--dataset_text_field", type=str, default="content")

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--use_bnb", type=bool, default=False)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_smollm2_python")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--repo_id", type=str, default="SmolLM2-1.7B-finetune")
    return parser.parse_args()


def main(args):
    # config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    bnb_config = None
    if args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # load model and dataset
    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)

    data = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        token=token,
        num_proc=args.num_proc if args.num_proc or args.streaming else multiprocessing.cpu_count(),
        streaming=args.streaming,
    )

    # setup the trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            dataset_text_field=args.dataset_text_field,
            dataset_num_proc=args.num_proc,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to="wandb",
            push_to_hub=args.push_to_hub,
            hub_model_id=args.repo_id,
        ),
        peft_config=lora_config,
    )

    # launch
    print("Training...")
    trainer.train()
    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
