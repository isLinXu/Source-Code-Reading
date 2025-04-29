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
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------

import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from mgm.model import *
from mgm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    """加载预训练模型主函数"""
    # 初始化设备映射参数
    kwargs = {"device_map": device_map, **kwargs}

    # 非CUDA设备处理
    if device != "cuda":
        kwargs['device_map'] = {"": device}  # 强制指定设备

    # 量化配置处理
    if load_8bit:
        kwargs['load_in_8bit'] = True  # 8位量化加载
    elif load_4bit:
        kwargs['load_in_4bit'] = True  # 4位量化配置
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # 计算时使用FP16
            bnb_4bit_use_double_quant=True,  # 双重量化节省空间
            bnb_4bit_quant_type='nf4'  # 4位量化类型
        )
    else:
        kwargs['torch_dtype'] = torch.float16  # 默认使用FP16

    # Flash Attention优化配置
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'  # 使用Flash Attention v2
    
    # MGM系列模型处理
    if 'mgm' in model_name.lower():        
        # 加载MGM多模态模型
        if model_base is not None:
            # 从基础模型加载（可能仅加载MM投影器）
            print('Loading MGM from base model...')
            
            # 根据模型类型选择不同架构
            if "8x7b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                model = MGMMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)  # Mixtral架构
            elif "2b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                model = MGMGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)  # Gemma架构
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = MGMLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)  # Llama架构
            
            # 加载多模态投影器权重
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}  # 转换为FP16
            model.load_state_dict(mm_projector_weights, strict=False)  # 非严格加载
        else:
            # 直接加载完整模型
            if "8x7b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = MGMMixtralForCausalLM.from_pretrained(model_path, **kwargs)
            elif "2b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = MGMGemmaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = MGMLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    else:
        # 加载普通语言模型
        if model_base is not None:
            # 使用PEFT（参数高效微调）模型
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)  # 加载LoRA权重
            print(f"Merging weights")
            model = model.merge_and_unload()  # 合并权重
            print('Convert to FP16...')
            model.to(torch.float16)  # 转换为FP16
        else:
            # 直接加载基础模型
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)  # MPT模型专用tokenizer
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None  # 初始化图像处理器

    # 处理特殊token
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)  # 是否使用图像起止符
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)  # 是否使用图像patch token
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)  # 添加图像patch token
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)  # 添加起止符
    
    model.resize_token_embeddings(len(tokenizer))  # 调整词嵌入层大小

    # 加载视觉模块
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()  # 延迟加载视觉模型
    vision_tower.to(device=device, dtype=torch.float16)  # 转换到指定设备和精度
    image_processor = vision_tower.image_processor  # 获取图像处理器
    
    # MGM特有处理
    if 'mgm' in model_name.lower():
        vision_tower_aux = model.get_vision_tower_aux()  # 获取辅助视觉模块
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()
        vision_tower_aux.to(device=device, dtype=torch.float16)
        
        # 初始化统一注意力模块
        model.config.model_path = model_path
        model.get_model().initialize_uni_modules(model.config, for_eval=True)  # 评估模式初始化
    
    # 获取上下文长度
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048  # 默认长度
    
    return tokenizer, model, image_processor, context_len