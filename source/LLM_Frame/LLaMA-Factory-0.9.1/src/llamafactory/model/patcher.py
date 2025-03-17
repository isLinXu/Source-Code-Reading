# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase, is_torch_npu_available
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import (
    autocast_projector_dtype,
    configure_visual_model,
    get_image_seqlen,
    get_patch_size,
    get_vision_feature_select_strategy,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

logger = logging.get_logger(__name__)  # 获取日志记录器

def patch_tokenizer(tokenizer: "PreTrainedTokenizer") -> None:
    # 如果 tokenizer 的 _pad 方法不是 PreTrainedTokenizerBase 的方法，则进行修补
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)  # 将 _pad 方法绑定到 tokenizer

def patch_processor(
    processor: "ProcessorMixin",
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)  # 设置处理器的 tokenizer 属性
    setattr(processor, "image_seqlen", get_image_seqlen(config))  # 设置图像序列长度
    setattr(processor, "image_resolution", model_args.image_resolution)  # 设置图像分辨率
    setattr(processor, "patch_size", get_patch_size(config, processor))  # 设置补丁大小
    setattr(processor, "video_resolution", model_args.video_resolution)  # 设置视频分辨率
    setattr(processor, "video_fps", model_args.video_fps)  # 设置视频帧率
    setattr(processor, "video_maxlen", model_args.video_maxlen)  # 设置视频最大长度
    setattr(processor, "vision_feature_select_strategy", get_vision_feature_select_strategy(config, processor))  # 设置视觉特征选择策略

def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: Dict[str, Any],
    is_trainable: bool,
) -> None:
    # 如果 model_args.compute_dtype 为 None，则优先选择 bf16 > fp16 > fp32
    if model_args.compute_dtype is None:
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)  # 获取推断数据类型
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))  # 推断优化数据类型

    if is_torch_npu_available():  # 检查是否可用 NPU
        use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]  # 获取 JIT 编译环境变量
        torch.npu.set_compile_mode(jit_compile=use_jit_compile)  # 设置 NPU 编译模式

    configure_attn_implementation(config, model_args, is_trainable)  # 配置注意力实现
    configure_rope(config, model_args, is_trainable)  # 配置 ROPE
    configure_longlora(config, model_args, is_trainable)  # 配置 LongLoRA
    configure_quantization(config, tokenizer, model_args, init_kwargs)  # 配置量化
    configure_moe(config, model_args, is_trainable)  # 配置混合专家
    configure_visual_model(config)  # 配置视觉模型
    configure_packing(config, model_args, is_trainable)  # 配置打包

    if model_args.use_cache and not is_trainable:
        setattr(config, "use_cache", True)  # 如果使用缓存且不可训练，则启用缓存
        logger.info_rank0("Using KV cache for faster generation.")  # 记录使用 KV 缓存的信息

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")  # 设置是否使用闪存注意力
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)  # 设置数据类型

    if getattr(config, "model_type", None) == "qwen2" and is_trainable and model_args.flash_attn == "fa2":
        setattr(config, "use_cache", False)  # qwen2 在使用闪存注意力时不支持缓存

    if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")  # 抛出错误，提示下载兼容格式的模型

    # deepspeed zero3 与低 CPU 内存使用不兼容
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())  # 设置低 CPU 内存使用的初始化参数

    # 如果：
    # 1. 不是 deepspeed zero3 也不是 fsdp（保持 zero3 或 fsdp 为 float32）
    # 2. quantization_bit 不为 None（qlora）
    if (not is_deepspeed_zero3_enabled() and not is_fsdp_enabled()) or model_args.quantization_bit is not None:
        init_kwargs["torch_dtype"] = model_args.compute_dtype  # 设置 torch 数据类型

        if init_kwargs["low_cpu_mem_usage"]:  # 设备映射需要低 CPU 内存使用为 True
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map  # 设置设备映射

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder  # 设置卸载文件夹

def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # 检查并修复生成配置
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True  # 如果不采样，则启用采样

    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)  # 将生成方法绑定到模型

    if add_valuehead:
        prepare_valuehead_model(model)  # 准备价值头模型

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)  # 调整模型的词汇嵌入层

    if is_trainable:
        prepare_model_for_training(model, model_args)  # 准备模型进行训练
        autocast_projector_dtype(model, model_args)  # 自动投影数据类型
        add_z3_leaf_module(model)  # 添加 Z3 叶子模块

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)  # 打印注意力实现

    try:
        model.add_model_tags(["llama-factory"])  # 添加模型标签
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")  # 记录无法正确标记模型的警告

def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()  # 绑定权重

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()  # 获取输入嵌入层

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()  # 获取输出嵌入层

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)  # 创建或更新模型卡片

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]  # 忽略的模块
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)  # 设置保存时忽略的键
    setattr(model, "tie_weights", MethodType(tie_weights, model))  # 绑定权重方法
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))  # 绑定获取输入嵌入方法
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))  # 绑定获取输出嵌入方法
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))  # 绑定创建或更新模型卡片方法