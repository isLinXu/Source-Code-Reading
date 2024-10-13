# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from omni_speech.model import *
from omni_speech.model.speech_encoder.builder import build_speech_encoder


def load_pretrained_model(model_path, model_base, is_lora=False, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, **kwargs):
    """
        加载预训练模型。

        该函数根据提供的参数加载并准备一个预训练的AI模型。它支持不同的加载选项，如加载8位或4位模型，
        使用Flash Attention等。它还支持LoRA模型的加载。

        参数:
        - model_path: 模型文件的路径。
        - model_base: 基础模型的名称或路径。
        - is_lora: 是否为LoRA模型。
        - s2s: 是否为speech-to-speech模型。
        - load_8bit: 是否加载8位模型。
        - load_4bit: 是否加载4位模型。
        - device: 所使用的设备（如'cuda'）。
        - use_flash_attn: 是否使用Flash Attention。
        - **kwargs: 其他关键字参数。

        返回:
        - tokenizer: 用于文本编码的tokenizer对象。
        - model: 加载的预训练模型。
        - context_len: 上下文长度。
    """
    # 设置模型加载选项
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # 设置Attention实现
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # 选择模型类
    model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM

    # Load OmniSpeech model
    # 加载OmniSpeech模型
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig
        lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading OmniSpeech from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional OmniSpeech weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        print('Loading OmniSpeech from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs)
        
        speech_projector_weights = torch.load(os.path.join(model_path, 'speech_projector.bin'), map_location='cpu')
        speech_projector_weights = {k: v.to(torch.float16) for k, v in speech_projector_weights.items()}
        model.load_state_dict(speech_projector_weights, strict=False)
        model = model.to(device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = model_cls.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            **kwargs
        )
        model = model.to(device=device)

    # 构建并加载speech encoder
    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(device=device, dtype=torch.float16)

    # 确定上下文长度
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
