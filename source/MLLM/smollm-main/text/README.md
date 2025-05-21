# SmolLM2
![image/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/XlT5TM3HWpfoZk_HSubrH.png)

SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device. You can find our most capable model **🤏 SmolLM2-1.7B-Instruct** [here](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

In this section you can find everything related to the training of SmolLM2. This includes pretraining and finetuning code, data curation as well as evaluation. We also recommend [SmolCourse](https://github.com/huggingface/smol-course) for more resources on smol models and how to leverage SmolLM2.

**News 📰**
- **Introducing [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), the best public math pre-training dataset 🚀**
- We added the code to do continual pre-training of Llama 3.2 3B on FineMath & FineWeb-Edu with `nanotron` at [pretraining/continual-pretraining](./pretraining/continual-pretraining)


## Table of Contents
1. [Usage](#usage)
    - [Transformers](#transformers)
    - [Chat in TRL](#chat-in-trl)
    - [Local inference](#local-inference)
    - [Smol-tools](#smol-tools)
2. [Pretraining](#pretraining)
3. [Finetuning](#finetuning)
4. [Evaluation](#evaluation)
5. [Data](#data)

## Usage
Our most powerful model is `SmolLM2-1.7B-Instruct`, which you can use as an assistant with `transformers`, `trl`, or using quantized versions with tools like `llama.cpp`, `MLX`, and `transformers.js`. For lighter applications, you can also use the smaller models `SmolLM2-360M` and`SmolLM2-135M`, which are suitable for on-device usage and can be integrated similarly.
All available in this [collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

### Transformers
```bash
pip install transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "Write a 100-word article on 'Benefits of Open-Source in AI research"}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
```

### Chat in TRL
You can also use the TRL CLI to chat with the model from the terminal:
```bash
pip install trl
trl chat --model_name_or_path HuggingFaceTB/SmolLM2-1.7B-Instruct --device cpu
```

You can find more details on how to leverage the model for use cases such as text summarization, text rewriting and function calling in the model card: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct 

### Local inference
You can use the models locally with frameworks like `llama.cpp`, `MLX`, `MLC` and `transformers.js`. You can find the instructions to run SmolLM2 with these frameworks at [local-inference](../tools/smollm_local_inference/README.md).

### Smol-tools
A collection of lightweight AI-powered tools built with LLaMA.cpp and small language models. These tools are designed to run locally on your machine without requiring expensive GPU resources.
Further instructions on how to use the tools can be found in the [smol-tools README](../tools/smol_tools/README.md).

## Pretraining
You can find scripts for launching pretraining with [nanotron](https://github.com/huggingface/nanotron/) under [pretraining](./pretraining/README.md), we share the exact configs for training SmolLM1 and will upload SmolLM2's configs soon. Additionally we provide code for continual-pretraining on SmolLM2 and Llama3.2 3B using nanotron. The SmolLM2 nanotron checkpoints are available [on the hub](https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt) with their optimizer states.

## Finetuning
You can find an example script to finetune SmolLM2 using `TRL` and `PEFT` in the `finetuning` folder. We also link to our post-training scripts for SmolLM2 using the alignment handbook. We also recommend [SmolCourse](https://github.com/huggingface/smol-course) for more resources on finetuning smol models and SmolLM2.

## Data
We also provide the code for curating the SmolLM datasets in [data](./data/README.md), this includes FineWeb-Edu, FineMath and the [distilabel](https://github.com/argilla-io/distilabel) pipelines for SmolTalk.