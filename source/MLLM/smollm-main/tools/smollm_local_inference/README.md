# SmolLM local inference

You can use SmolLM2 models locally with frameworks like Transformers.js, llama.cpp, MLX and MLC.

Here you can find the code for running SmolLM locally using each of these libraries. You can also find the conversions of SmolLM & SmolLM2 in these collections: [SmolLM1](https://huggingface.co/collections/HuggingFaceTB/local-smollms-66c0f3b2a15b4eed7fb198d0) and [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

Please first install each library by following its documentation:
- [Transformers.js](https://github.com/huggingface/transformers.js)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [MLX](https://github.com/ml-explore/mlx)
- [MLC](https://github.com/mlc-ai/web-llm)

## Demos
Below are some demos we built for running SmolLM models on-device.

### In-browser chat assistants
- [WebGPU chat demo](https://huggingface.co/spaces/HuggingFaceTB/SmolLM2-1.7B-Instruct-WebGPU) of SmolLM2 1.7B Instruct powered by Transformers.js and ONNX Runtime Web.
- [Instant SmolLM](https://huggingface.co/spaces/HuggingFaceTB/instant-smollm) powered by MLC for real-time generations of SmolLM-360M-Instruct.

The models are also available on [Ollama](https://ollama.com/library/smollm2) and [PocketPal-AI](https://github.com/a-ghorbani/pocketpal-ai).

### Other use cases
#### Text extraction 
- [Github Issue Generator running locally w/ SmolLM2 & WebGPU](https://huggingface.co/spaces/reach-vb/github-issue-generator-webgpu) showcases how to use SmolLM2 1.7B for structured text extraction to convert complaints to structured GitHub issues. The demo leverages MLC WebLLM and [XGrammar](https://github.com/mlc-ai/xgrammar) for structured generation. You can define a JSON schema, input free text and get structured data in your browser.

#### Function calling 
- [Bunny B1](https://github.com/dottxt-ai/demos/tree/main/its-a-smol-world) mapping natural language requests to local aplication calls using function calling and structured generation by [outlines](https://github.com/dottxt-ai/outlines).
- You can also leverage function calling (without structured generation) by following the instructions in the [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct#function-calling) or using SmolAgent from [smol-tools](../smol_tools/)

#### Rewriting and Summarization
- Check the rewriting and summarization tools in [smol-tools](../smol_tools/) using llama.cpp