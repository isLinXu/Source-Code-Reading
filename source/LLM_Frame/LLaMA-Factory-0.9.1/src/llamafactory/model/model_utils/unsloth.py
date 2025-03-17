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

from typing import TYPE_CHECKING, Any, Dict, Optional  # 导入类型检查、任意类型、字典和可选类型

from ...extras import logging  # 从 extras 导入日志模块
from ...extras.misc import get_current_device  # 从 extras.misc 导入获取当前设备的函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig, PreTrainedModel  # 导入预训练配置和预训练模型的类型

    from ...hparams import ModelArguments  # 导入模型参数的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def _get_unsloth_kwargs(
    config: "PretrainedConfig", model_name_or_path: str, model_args: "ModelArguments"
) -> Dict[str, Any]:
    return {
        "model_name": model_name_or_path,  # 模型名称或路径
        "max_seq_length": model_args.model_max_length or 4096,  # 最大序列长度，默认为 4096
        "dtype": model_args.compute_dtype,  # 计算数据类型
        "load_in_4bit": model_args.quantization_bit == 4,  # 是否以 4 位加载
        "token": model_args.hf_hub_token,  # Hugging Face Hub 令牌
        "device_map": {"": get_current_device()},  # 设备映射为当前设备
        "rope_scaling": getattr(config, "rope_scaling", None),  # 获取 RoPE 缩放参数
        "fix_tokenizer": False,  # 是否修复分词器
        "trust_remote_code": True,  # 是否信任远程代码
        "use_gradient_checkpointing": "unsloth",  # 使用 unsloth 的梯度检查点
    }


def load_unsloth_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments"
) -> Optional["PreTrainedModel"]:
    r"""
    Optionally loads pretrained model with unsloth. Used in training.  # 可选地加载带有 unsloth 的预训练模型，用于训练。
    """
    from unsloth import FastLanguageModel  # 从 unsloth 导入快速语言模型

    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.model_name_or_path, model_args)  # 获取 unsloth 的参数
    try:
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)  # 加载预训练模型
    except NotImplementedError:  # 如果未实现
        logger.warning_rank0("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))  # 记录警告信息
        model = None  # 模型设置为 None
        model_args.use_unsloth = False  # 不使用 unsloth

    return model  # 返回模型


def get_unsloth_peft_model(
    model: "PreTrainedModel", model_args: "ModelArguments", peft_kwargs: Dict[str, Any]
) -> "PreTrainedModel":
    r"""
    Gets the peft model for the pretrained model with unsloth. Used in training.  # 获取带有 unsloth 的预训练模型的 peft 模型，用于训练。
    """
    from unsloth import FastLanguageModel  # 从 unsloth 导入快速语言模型

    unsloth_peft_kwargs = {
        "model": model,  # 模型
        "max_seq_length": model_args.model_max_length,  # 最大序列长度
        "use_gradient_checkpointing": "unsloth",  # 使用 unsloth 的梯度检查点
    }
    return FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)  # 获取 peft 模型


def load_unsloth_peft_model(
    config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Loads peft model with unsloth. Used in both training and inference.  # 加载带有 unsloth 的 peft 模型，用于训练和推理。
    """
    from unsloth import FastLanguageModel  # 从 unsloth 导入快速语言模型

    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.adapter_name_or_path[0], model_args)  # 获取 unsloth 的参数
    try:
        if not is_trainable:  # 如果不可训练
            unsloth_kwargs["use_gradient_checkpointing"] = False  # 不使用梯度检查点

        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)  # 加载预训练模型
    except NotImplementedError:  # 如果未实现
        raise ValueError("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))  # 抛出错误

    if not is_trainable:  # 如果不可训练
        FastLanguageModel.for_inference(model)  # 将模型设置为推理模式

    return model  # 返回模型