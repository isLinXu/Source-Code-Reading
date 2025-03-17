# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and Optimum library.
# https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/utils/quantization_config.py
# https://github.com/huggingface/optimum/blob/v1.20.0/optimum/gptq/data.py
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

import os  # 导入操作系统模块
import random  # 导入随机数模块
from enum import Enum, unique  # 从枚举模块导入 Enum 和 unique
from typing import TYPE_CHECKING, Any, Dict, List  # 导入类型检查、任意类型、字典和列表

import torch  # 导入 PyTorch 库
from datasets import load_dataset  # 从 datasets 导入加载数据集的函数
from transformers import BitsAndBytesConfig, EetqConfig, GPTQConfig, HqqConfig  # 从 transformers 导入各种量化配置类
from transformers.integrations import is_deepspeed_zero3_enabled  # 从 transformers 导入检查是否启用 DeepSpeed Zero3 的函数
from transformers.modeling_utils import is_fsdp_enabled  # 从 transformers 导入检查是否启用 FSDP 的函数
from transformers.utils.versions import require_version  # 从 transformers 导入版本检查函数

from ...extras import logging  # 从 extras 导入日志模块
from ...extras.constants import FILEEXT2TYPE  # 从 extras.constants 导入文件扩展名与类型的映射
from ...extras.misc import get_current_device  # 从 extras.misc 导入获取当前设备的函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig, PreTrainedTokenizer  # 导入预训练配置和分词器的类型

    from ...hparams import ModelArguments  # 导入模型参数的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


@unique
class QuantizationMethod(str, Enum):
    r"""
    Borrowed from `transformers.utils.quantization_config.QuantizationMethod`.  # 从 transformers.utils.quantization_config 导入的量化方法
    """

    BITS_AND_BYTES = "bitsandbytes"  # 定义量化方法：bitsandbytes
    GPTQ = "gptq"  # 定义量化方法：gptq
    AWQ = "awq"  # 定义量化方法：awq
    AQLM = "aqlm"  # 定义量化方法：aqlm
    QUANTO = "quanto"  # 定义量化方法：quanto
    EETQ = "eetq"  # 定义量化方法：eetq
    HQQ = "hqq"  # 定义量化方法：hqq


def _get_quantization_dataset(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> List[Dict[str, Any]]:
    r"""
    Prepares the tokenized dataset to perform AutoGPTQ. Do not use tensor output for JSON serialization.  # 准备标记化的数据集以执行 AutoGPTQ。不要使用张量输出进行 JSON 序列化。
    """
    if os.path.isfile(model_args.export_quantization_dataset):  # 如果指定的量化数据集是文件
        data_path = FILEEXT2TYPE.get(model_args.export_quantization_dataset.split(".")[-1], None)  # 获取数据路径
        data_files = model_args.export_quantization_dataset  # 设置数据文件路径
    else:  # 如果不是文件
        data_path = model_args.export_quantization_dataset  # 设置数据路径
        data_files = None  # 数据文件为 None

    dataset = load_dataset(  # 加载数据集
        path=data_path,  # 数据集路径
        data_files=data_files,  # 数据文件
        split="train",  # 使用训练集
        cache_dir=model_args.cache_dir,  # 缓存目录
        token=model_args.hf_hub_token,  # Hugging Face Hub 令牌
    )

    samples = []  # 初始化样本列表
    maxlen = model_args.export_quantization_maxlen  # 获取最大长度
    for _ in range(model_args.export_quantization_nsamples):  # 遍历样本数量
        n_try = 0  # 尝试次数
        while True:  # 无限循环
            if n_try > 100:  # 如果尝试次数超过 100
                raise ValueError("Cannot find satisfying example, considering decrease `export_quantization_maxlen`.")  # 抛出错误

            sample_idx = random.randint(0, len(dataset) - 1)  # 随机选择样本索引
            sample: Dict[str, "torch.Tensor"] = tokenizer(dataset[sample_idx]["text"], return_tensors="pt")  # 标记化样本
            n_try += 1  # 尝试次数加一
            if sample["input_ids"].size(1) > maxlen:  # 如果输入 ID 的长度大于最大长度
                break  # 跳出循环

        word_idx = random.randint(0, sample["input_ids"].size(1) - maxlen - 1)  # 随机选择单词索引
        input_ids = sample["input_ids"][:, word_idx : word_idx + maxlen]  # 获取输入 ID
        attention_mask = sample["attention_mask"][:, word_idx : word_idx + maxlen]  # 获取注意力掩码
        samples.append({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()})  # 将样本添加到列表

    return samples  # 返回样本列表


def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: Dict[str, Any],
) -> None:
    r"""
    Priority: PTQ-quantized (train/infer) > AutoGPTQ (export) > On-the-fly quantization (train/infer)  # 优先级：PTQ 量化（训练/推理）> AutoGPTQ（导出）> 动态量化（训练/推理）
    """
    if getattr(config, "quantization_config", None):  # 如果存在量化配置
        if model_args.quantization_bit is not None:  # 如果量化位数不为 None
            logger.warning_rank0("`quantization_bit` will not affect on the PTQ-quantized models.")  # 记录警告

        if is_deepspeed_zero3_enabled() or is_fsdp_enabled():  # 如果启用了 DeepSpeed ZeRO-3 或 FSDP
            raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")  # 抛出错误

        quantization_config: Dict[str, Any] = getattr(config, "quantization_config", None)  # 获取量化配置
        quant_method = quantization_config.get("quant_method", "")  # 获取量化方法

        if quant_method == QuantizationMethod.GPTQ:  # 如果量化方法为 GPTQ
            require_version("auto_gptq>=0.5.0", "To fix: pip install auto_gptq>=0.5.0")  # 检查版本
            quantization_config.pop("disable_exllama", None)  # 移除已弃用的参数
            quantization_config["use_exllama"] = False  # 禁用 exllama

        if quant_method == QuantizationMethod.AWQ:  # 如果量化方法为 AWQ
            require_version("autoawq", "To fix: pip install autoawq")  # 检查版本

        if quant_method == QuantizationMethod.AQLM:  # 如果量化方法为 AQLM
            require_version("aqlm>=1.1.0", "To fix: pip install aqlm[gpu]>=1.1.0")  # 检查版本
            quantization_config["bits"] = 2  # 设置位数为 2

        quant_bits = quantization_config.get("bits", "?")  # 获取量化位数
        logger.info_rank0(f"Loading {quant_bits}-bit {quant_method.upper()}-quantized model.")  # 记录加载模型的信息

    elif model_args.export_quantization_bit is not None:  # 如果存在自动量化位数
        if model_args.export_quantization_bit not in [8, 4, 3, 2]:  # 如果量化位数不在支持的范围内
            raise ValueError("AutoGPTQ only accepts 2/3/4/8-bit quantization.")  # 抛出错误

        require_version("optimum>=1.17.0", "To fix: pip install optimum>=1.17.0")  # 检查版本
        require_version("auto_gptq>=0.5.0", "To fix: pip install auto_gptq>=0.5.0")  # 检查版本
        from accelerate.utils import get_max_memory  # 从 accelerate 导入获取最大内存的函数

        if getattr(config, "model_type", None) == "chatglm":  # 如果模型类型为 chatglm
            raise ValueError("ChatGLM model is not supported yet.")  # 抛出错误

        init_kwargs["quantization_config"] = GPTQConfig(  # 设置量化配置
            bits=model_args.export_quantization_bit,  # 设置位数
            dataset=_get_quantization_dataset(tokenizer, model_args),  # 获取量化数据集
        )
        init_kwargs["device_map"] = "auto"  # 设置设备映射为自动
        init_kwargs["max_memory"] = get_max_memory()  # 获取最大内存并设置
        logger.info_rank0(f"Quantizing model to {model_args.export_quantization_bit} bit with AutoGPTQ.")  # 记录量化模型的信息

    elif model_args.quantization_bit is not None:  # 如果存在量化位数
        if model_args.quantization_method == QuantizationMethod.BITS_AND_BYTES.value:  # 如果量化方法为 bitsandbytes
            if model_args.quantization_bit == 8:  # 如果量化位数为 8
                require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")  # 检查版本
                init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)  # 设置量化配置为 8 位
            elif model_args.quantization_bit == 4:  # 如果量化位数为 4
                require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")  # 检查版本
                init_kwargs["quantization_config"] = BitsAndBytesConfig(  # 设置量化配置为 4 位
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_args.compute_dtype,  # 设置计算数据类型
                    bnb_4bit_use_double_quant=model_args.double_quantization,  # 设置是否使用双重量化
                    bnb_4bit_quant_type=model_args.quantization_type,  # 设置量化类型
                    bnb_4bit_quant_storage=model_args.compute_dtype,  # 设置量化存储类型
                )
            else:  # 如果量化位数不在支持范围内
                raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")  # 抛出错误

            # Do not assign device map if:
            # 1. deepspeed zero3 or fsdp (train)  # 如果启用了 DeepSpeed ZeRO-3 或 FSDP（训练）
            # 2. auto quantization device map (inference)  # 如果为自动量化设备映射（推理）
            if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quantization_device_map == "auto":  # 如果启用了 DeepSpeed ZeRO-3 或 FSDP，或设备映射为自动
                if model_args.quantization_bit != 4:  # 如果量化位数不为 4
                    raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")  # 抛出错误

                require_version("bitsandbytes>=0.43.0", "To fix: pip install bitsandbytes>=0.43.0")  # 检查版本
            else:  # 如果不满足上述条件
                init_kwargs["device_map"] = {"": get_current_device()}  # 设置设备映射为当前设备

            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with bitsandbytes.")  # 记录量化模型的信息
        elif model_args.quantization_method == QuantizationMethod.HQQ.value:  # 如果量化方法为 HQQ
            if model_args.quantization_bit not in [8, 6, 5, 4, 3, 2, 1]:  # 如果量化位数不在支持范围内
                raise ValueError("HQQ only accepts 1/2/3/4/5/6/8-bit quantization.")  # 抛出错误

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():  # 如果启用了 DeepSpeed ZeRO-3 或 FSDP
                raise ValueError("HQQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")  # 抛出错误

            require_version("hqq", "To fix: pip install hqq")  # 检查版本
            init_kwargs["quantization_config"] = HqqConfig(  # 设置量化配置
                nbits=model_args.quantization_bit, quant_zero=False, quant_scale=False, axis=0  # 使用 ATEN 内核（轴=0）以提高性能
            )
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with HQQ.")  # 记录量化模型的信息
        elif model_args.quantization_method == QuantizationMethod.EETQ.value:  # 如果量化方法为 EETQ
            if model_args.quantization_bit != 8:  # 如果量化位数不为 8
                raise ValueError("EETQ only accepts 8-bit quantization.")  # 抛出错误

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():  # 如果启用了 DeepSpeed ZeRO-3 或 FSDP
                raise ValueError("EETQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")  # 抛出错误

            require_version("eetq", "To fix: pip install eetq")  # 检查版本
            init_kwargs["quantization_config"] = EetqConfig()  # 设置量化配置
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with EETQ.")  # 记录量化模型的信息