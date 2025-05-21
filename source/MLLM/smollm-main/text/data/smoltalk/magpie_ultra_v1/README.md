# MagPie Ultra v1.0

This [`distilabel`](https://github.com/argilla-io/distilabel) was used to generate the [magpie-ultra-v1.0](https://huggingface.co/datasets/argilla/magpie-ultra-v1.0) dataset. The dataset follows the [MagPie](https://magpie-align.github.io) pipeline recipe to generate a multi-turn conversation dataset using [meta-llama/Llama-3.1-405B-Instruct-FP8](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct-FP8).

## Setup

You will need to install `distilabel` with a few extra dependencies to be able to execute the pipeline:

```bash
pip install distilabel[ray,vllm,sentence-transformers,faiss-cpu,hf-transformers]
```