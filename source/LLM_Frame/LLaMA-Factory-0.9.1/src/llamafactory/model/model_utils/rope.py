# Copyright 2024 LMSYS and the LlamaFactory team.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
# This code is inspired by the LMSYS's FastChat library.
# https://github.com/lm-sys/FastChat/blob/v0.2.30/fastchat/train/train.py
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

import math  # 导入数学模块
from typing import TYPE_CHECKING  # 导入类型检查模块

from ...extras import logging  # 从 extras 导入日志模块


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig  # 导入预训练配置的类型

    from ...hparams import ModelArguments  # 导入模型参数的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if model_args.rope_scaling is None:  # 如果 RoPE 缩放参数为 None
        return  # 直接返回

    if not hasattr(config, "rope_scaling"):  # 如果配置中没有 rope_scaling 属性
        logger.warning_rank0("Current model does not support RoPE scaling.")  # 记录警告信息
        return  # 直接返回

    if model_args.model_max_length is not None:  # 如果模型最大长度不为 None
        if is_trainable and model_args.rope_scaling == "dynamic":  # 如果是可训练的并且 RoPE 缩放为动态
            logger.warning_rank0(  # 记录警告信息
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        current_max_length = getattr(config, "max_position_embeddings", None)  # 获取当前最大位置嵌入长度
        if current_max_length and model_args.model_max_length > current_max_length:  # 如果当前最大长度存在且模型最大长度大于当前最大长度
            logger.info_rank0(f"Enlarge max model length from {current_max_length} to {model_args.model_max_length}.")  # 记录信息，扩大最大模型长度
            setattr(config, "max_position_embeddings", model_args.model_max_length)  # 更新配置中的最大位置嵌入长度
            scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))  # 计算缩放因子
        else:  # 如果输入长度小于最大长度
            logger.warning_rank0("Input length is smaller than max length. Consider increase input length.")  # 记录警告信息
            scaling_factor = 1.0  # 设置缩放因子为 1.0
    else:  # 如果模型最大长度为 None
        scaling_factor = 2.0  # 设置缩放因子为 2.0

    setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})  # 更新配置中的 RoPE 缩放信息
    logger.info_rank0(  # 记录使用的 RoPE 缩放策略和缩放因子
        f"Using {model_args.rope_scaling} scaling strategy and setting scaling factor to {scaling_factor}"
    )
