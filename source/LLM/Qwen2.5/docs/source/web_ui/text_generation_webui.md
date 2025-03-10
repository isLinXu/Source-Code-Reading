# Text Generation Web UI

[Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) (TGW, or usually referred to "oobabooga") is a popular web UI for text generation, similar to[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
It has multiple interfaces, and supports multiple model backends, including 
- [Transformers](https://github.com/huggingface/transformers),
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)),
- [ExLlamaV2](https://github.com/turboderp/exllamav2),
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ),
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ),
- [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa),
- [CTransformers](https://github.com/marella/ctransformers),
- [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp). 

In this section, we introduce how to run Qwen locally with TGW.

## Quickstart

The simplest way to run TGW is to use the provided shell scripts in the [repo](https://github.com/oobabooga/text-generation-webui). 
For the first step, clone the repo and enter the directory:

```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
```

You can directly run the `start_linux.sh`, `start_windows.bat`, `start_macos.sh`, or `start_wsl.bat` script depending on your OS.
Alternatively you can manually install the requirements in your conda environment.
Here I take the practice on MacOS as an example:

```bash
conda create -n textgen python=3.11
conda activate textgen
pip install torch torchvision torchaudio
```

Then you can install the requirements by running `pip install -r` based on your OS, e.g.,

```bash
pip install -r requirements_apple_silicon.txt
```

For `bitsandbytes` and `llama-cpp-python` inside the requirements, I advise you to install them through `pip` directly. 
After finishing the installation of required packages, you need to prepare your models by putting the model files or directories in the folder `./models`.
For example, you should put the transformers model directory of `Qwen2.5-7B-Instruct` in the way shown below:

```
text-generation-webui
├── models
│   ├── Qwen2.5-7B-Instruct
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model-00001-of-00004.safetensor
│   │   ├── model-00002-of-00004.safetensor
│   │   ├── model-00003-of-00004.safetensor
│   │   ├── model-00004-of-00004.safetensor
│   │   ├── model.safetensor.index.json
│   │   ├── merges.txt
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
```

Then you just need to run

```bash
python server.py
```

to launch your web UI service. Please browse to

```
http://localhost:7860/?__theme=dark
```

and enjoy playing with Qwen in a web UI!

## Next Step

There are a lot more usages in TGW, where you can even enjoy role play, use different types of quantized models, train LoRA, incorporate extensions like stable diffusion and whisper, etc. 
Go to figure out more advanced usages and apply them to Qwen models!
