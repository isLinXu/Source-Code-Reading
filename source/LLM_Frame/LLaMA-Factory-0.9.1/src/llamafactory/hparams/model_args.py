# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Union

import torch
from transformers.training_args import _convert_str_dict
from typing_extensions import Self

@dataclass
class QuantizationArguments:
    r"""
    Arguments pertaining to the quantization method.
    与量化方法相关的参数。
    """

    quantization_method: Literal["bitsandbytes", "hqq", "eetq"] = field(
        default="bitsandbytes",
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    # 用于动态量化的量化方法，默认为"bitsandbytes"。
    
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using on-the-fly quantization."},
    )
    # 动态量化时用于量化模型的位数，默认为None。
    
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    # 在bitsandbytes int4训练中使用的量化数据类型，默认为"nf4"。
    
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in bitsandbytes int4 training."},
    )
    # 是否在bitsandbytes int4训练中使用双重量化，默认为True。
    
    quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )
    # 用于推断4位量化模型的设备映射，默认为None，要求bitsandbytes版本>=0.43.0。

@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    与图像处理器相关的参数。
    """

    image_resolution: int = field(
        default=512 * 512,
        metadata={"help": "Keeps the number of pixels of image below this resolution."},
    )
    # 图像的像素数保持在此分辨率以下，默认为512*512。
    
    video_resolution: int = field(
        default=128 * 128,
        metadata={"help": "Keeps the number of pixels of video below this resolution."},
    )
    # 视频的像素数保持在此分辨率以下，默认为128*128。
    
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    # 视频输入每秒采样的帧数，默认为2.0。
    
    video_maxlen: int = field(
        default=64,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )
    # 视频输入的最大采样帧数，默认为64。

@dataclass
class ExportArguments:
    r"""
    Arguments pertaining to the model export.
    与模型导出相关的参数。
    """

    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    # 导出模型保存的目录路径，默认为None。
    
    export_size: int = field(
        default=1,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    # 导出模型的文件分片大小（以GB为单位），默认为1。
    
    export_device: Literal["cpu", "auto"] = field(
        default="cpu",
        metadata={"help": "The device used in model export, use `auto` to accelerate exporting."},
    )
    # 模型导出时使用的设备，默认为"cpu"，可以选择"auto"以加速导出。
    
    export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    # 导出模型的量化位数，默认为None。
    
    export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    # 用于量化导出模型的数据集路径或名称，默认为None。
    
    export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    # 用于量化的样本数量，默认为128。
    
    export_quantization_maxlen: int = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    # 用于量化的模型输入的最大长度，默认为1024。
    
    export_legacy_format: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    # 是否保存为`.bin`文件而不是`.safetensors`，默认为False。
    
    export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )
    # 如果将模型推送到Hugging Face hub，则为存储库的名称，默认为None。

@dataclass
class VllmArguments:
    r"""
    Arguments pertaining to the vLLM worker.
    与vLLM工作者相关的参数。
    """

    vllm_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum sequence (prompt + response) length of the vLLM engine."},
    )
    # vLLM引擎的最大序列长度（提示+响应），默认为4096。
    
    vllm_gpu_util: float = field(
        default=0.9,
        metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    # vLLM引擎使用的GPU内存比例（0到1之间），默认为0.9。
    
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    # 是否在vLLM引擎中禁用CUDA图，默认为False。
    
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )
    # vLLM引擎中所有LoRA的最大秩，默认为32。
    
    vllm_config: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "Config to initialize the vllm engine. Please use JSON strings."},
    )
    # 用于初始化vLLM引擎的配置，默认为None，建议使用JSON字符串。

@dataclass
class ModelArguments(QuantizationArguments, ProcessorArguments, ExportArguments, VllmArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    与我们要微调或推断的模型/配置/分词器相关的参数。
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    # 模型权重的路径或来自huggingface.co/models或modelscope.cn/models的标识符，默认为None。
    
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    # 适配器权重的路径或来自huggingface.co/models的标识符，默认为None，多个适配器用逗号分隔。
    
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    # 包含要加载的适配器权重的文件夹路径，默认为None。
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    # 存储从huggingface.co或modelscope.cn下载的预训练模型的路径，默认为None。
    
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    # 是否使用快速分词器（由tokenizers库支持），默认为True。
    
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    # 是否调整分词器词汇表和嵌入层的大小，默认为False。
    
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    # 是否在分词过程中拆分特殊token，默认为False。
    
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    # 要添加到分词器中的特殊token，默认为None，多个token用逗号分隔。
    
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # 要使用的特定模型版本（可以是分支名称、标签名称或提交ID），默认为"main"。
    
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    # 是否使用内存高效的模型加载，默认为True。
    
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    # RoPE嵌入应采用的缩放策略，默认为None。
    
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = field(
        default="auto",
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    # 启用FlashAttention以加快训练和推断，默认为"auto"。
    
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    # 启用LongLoRA提出的短注意力（S^2-Attn），默认为False。
    
    mixture_of_depths: Optional[Literal["convert", "load"]] = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    # 将模型转换为深度混合（MoD）或加载MoD模型，默认为None。
    
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    # 是否使用unsloth的优化进行LoRA训练，默认为False。
    
    use_unsloth_gc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's gradient checkpointing."},
    )
    # 是否使用unsloth的梯度检查点，默认为False。
    
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    # 是否启用liger内核以加快训练，默认为False。
    
    moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    # 在混合专家模型中辅助路由损失的系数，默认为None。
    
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    # 是否禁用梯度检查点，默认为False。
    
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    # 是否将layernorm权重上升到fp32，默认为False。
    
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    # 是否将lm_head的输出上升到fp32，默认为False。
    
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    # 是否随机初始化模型权重，默认为False。
    
    infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    # 推断时使用的后端引擎，默认为"huggingface"。
    
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    # 模型权重的卸载路径，默认为"offload"。
    
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    # 是否在生成中使用KV缓存，默认为True。
    
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    # 推断时模型权重和激活的数值类型，默认为"auto"。
    
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    # 用于登录Hugging Face Hub的身份验证令牌，默认为None。
    
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    # 用于登录ModelScope Hub的身份验证令牌，默认为None。
    
    om_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Modelers Hub."},
    )
    # 用于登录Modelers Hub的身份验证令牌，默认为None。
    
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    # 用于调试目的，打印模型中参数的状态，默认为False。
    
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    # 用于计算模型输出的Torch数据类型，从`fp/bf16`推导而来，默认为None。
    
    device_map: Optional[Union[str, Dict[str, Any]]] = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    # 模型放置的设备映射，从训练阶段推导而来，默认为None。
    
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    # 模型的最大输入长度，从`cutoff_len`推导而来，默认为None。
    
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={"help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."},
    )
    # 是否使用块对角注意力，从`neat_packing`推导而来，默认为False。

    def __post_init__(self):
        # 后处理初始化
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")
            # 如果未提供模型名称或路径，则引发错误。

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")
            # 如果拆分特殊token且使用快速分词器，则引发错误。

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]
            # 如果适配器名称或路径不为None，则支持合并多个lora权重。

        if self.new_special_tokens is not None:  # support multiple special tokens
            self.new_special_tokens = [token.strip() for token in self.new_special_tokens.split(",")]
            # 如果新特殊token不为None，则支持多个特殊token。

        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")
            # 如果导出量化位数不为None且导出量化数据集为None，则引发错误。

        if isinstance(self.vllm_config, str) and self.vllm_config.startswith("{"):
            self.vllm_config = _convert_str_dict(json.loads(self.vllm_config))
            # 如果vllm_config是字符串并且以"{"开头，则将其转换为字典。

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        # 从源对象复制参数
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
                # 如果属性可以初始化，则将其值添加到init_args。
            else:
                lazy_args[attr.name] = getattr(source, attr.name)
                # 否则将其值添加到lazy_args。

        init_args.update(kwargs)
        # 更新初始化参数。
        result = cls(**init_args)
        # 创建新实例。
        for name, value in lazy_args.items():
            setattr(result, name, value)
            # 将延迟参数的值设置到新实例中。

        return result
        # 返回新实例。