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
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

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

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            # 加载额外的LLaVA权重
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            # 检查是否存在非LoRA可训练参数的本地文件
            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # 如果不存在，可能是从HF Hub来的，需要从Hub下载文件
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            # 移除权重名称前缀
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            # 如果权重名称以'model.model.'开头，再次移除前缀
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

            # 加载非LoRA可训练参数到模型
            model.load_state_dict(non_lora_trainables, strict=False)
            # 加载LoRA权重
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        # 如果提供了基础模型路径
        elif model_base is not None:
            # this may be mm projector only # 这可能是mm投影器
            print('Loading LLaVA from base model...')
            # 处理mpt模型
            # 判断模型名称中是否包含 'mpt'，以确定使用哪种配置和模型加载方式
            if 'mpt' in model_name.lower():
                # 检查模型路径中是否缺少 'configuration_mpt.py' 文件，如果缺少，则从基础模型路径中复制过来
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))

                # 使用快速版本的 tokenizer，并从预训练模型基础路径中加载
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)

                # 从模型路径中加载配置信息，允许远程代码执行
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

                # 从预训练模型基础路径中加载 LlavaMpt 模型，使用低内存占用设置和预训练配置
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                # 对于非 'mpt' 模型，使用非快速版本的 tokenizer 并从预训练模型基础路径中加载
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

                # 从模型路径中加载配置信息，不包含远程代码执行
                cfg_pretrained = AutoConfig.from_pretrained(model_path)

                # 从预训练模型基础路径中加载 LlavaLlama 模型，使用低内存占用设置和预训练配置
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            # 加载多模态投影层的权重
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            # 将加载的权重转换为半精度浮点格式以减少内存消耗
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            # 加载转换后的权重到模型，strict=False表示允许部分权重不匹配
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # 根据模型名称加载相应的模型和tokenizer
            if 'mpt' in model_name.lower():
                # 如果模型名称包含mpt，则使用AutoTokenizer加载tokenizer，并从预训练模型路径加载LlavaMptForCausalLM模型
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                # 如果模型名称包含mistral，则使用AutoTokenizer加载tokenizer，并从预训练模型路径加载LlavaMistralForCausalLM模型
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                # 对于其他模型名称，使用AutoTokenizer加载tokenizer（不使用fast版本），并从预训练模型路径加载LlavaLlamaForCausalLM模型
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    # 加载语言模型
    else:
        # Load language model
        # 判断是否提供了模型基类，如果是，则使用PeftModel加载模型
        if model_base is not None:
            # PEFT模型处理
            # PEFT model
            from peft import PeftModel
            # 初始化tokenizer，使用指定的模型基类作为初始化tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            # 加载基础的因果语言模型
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            # 打印加载LoRA权重的信息
            print(f"Loading LoRA weights from {model_path}")
            # 加载LoRA模型
            model = PeftModel.from_pretrained(model, model_path)
            # 打印权重融合的信息
            print(f"Merging weights")
            # 融合LoRA权重并卸载
            model = model.merge_and_unload()
            # 转换模型为FP16精度
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            # 非PEFT模型处理，设置use_fast为False
            use_fast = False
            # 判断模型名称是否包含'mpt'
            if 'mpt' in model_name.lower():
                # 对于mpt模型，初始化tokenizer时允许使用fast版本
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                # 加载支持远程代码的模型
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                # 非mpt模型，初始化tokenizer时不使用fast版本
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                # 加载普通模型
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    # 初始化图像处理器
    image_processor = None
    # 如果是llava模型，处理图像相关配置
    if 'llava' in model_name.lower():
        # 获取模型配置中的mm_use_im_start_end属性，如果不存在，使用False作为默认值
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        # 获取模型配置中的mm_use_im_patch_token属性，如果不存在，使用True作为默认值
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        # 如果mm_use_im_patch_token为真，向tokenizer添加图像补丁标记
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # 如果mm_use_im_start_end为真，向tokenizer添加图像开始和结束标记
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        # 调整模型的令牌嵌入层大小以适应新的tokenizer
        model.resize_token_embeddings(len(tokenizer))
        # 获取视觉塔模型
        vision_tower = model.get_vision_tower()
        # 如果视觉塔模型未加载，则加载模型
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        # 如果设备映射不是自动设置，则将视觉塔模型移动到指定设备上
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        # 获取视觉塔模型的图像处理器
        image_processor = vision_tower.image_processor
    # 确定上下文长度
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
