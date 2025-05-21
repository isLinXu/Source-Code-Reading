# SmolLM evaluation scripts

We're using the [LightEval](https://github.com/huggingface/lighteval/) library to benchmark our models. 

Check out the [quick tour](https://github.com/huggingface/lighteval/wiki/Quicktour) to configure it to your own hardware and tasks.

## Setup

Use conda/venv with `python>=3.10`.

Adjust the pytorch installation according to your environment:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
For reproducibility, we recommend fixed versions of the libraries:
```bash
pip install -r requirements.txt
```

## Running the evaluations

### SmolLM2 base models

```bash
lighteval accelerate \
  --model_args "pretrained=HuggingFaceTB/SmolLM2-1.7B,revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048" \
  --custom_tasks "tasks.py" --tasks "smollm2_base.txt" --output_dir "./evals" --save_details
```

### SmolLM2 instruction-tuned models

(note the `--use_chat_template` flag)
```bash
lighteval accelerate \
  --model_args "pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct,revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048" \
  --custom_tasks "tasks.py" --tasks "smollm2_instruct.txt" --use_chat_template --output_dir "./evals" --save_details
```

### FineMath dataset ablations

See the collection for model names: https://huggingface.co/collections/HuggingFaceTB/finemath-6763fb8f71b6439b653482c2

```bash
lighteval accelerate \
  --model_args "pretrained=HuggingFaceTB/finemath-ablation-4plus-160B,revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.7,max_model_length=4096" \
  --custom_tasks "tasks.py" --tasks "custom|math|4|1,custom|gsm8k|5|1,custom|arc:challenge|0|1,custom|mmlu_pro|0|1,custom|hellaswag|0|1" --output_dir "./evals" --save_details
```
