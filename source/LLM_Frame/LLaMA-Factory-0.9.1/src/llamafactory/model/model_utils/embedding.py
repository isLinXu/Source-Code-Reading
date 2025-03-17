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

import math  # 导入数学库
from contextlib import nullcontext  # 从上下文管理库导入 nullcontext
from typing import TYPE_CHECKING  # 导入类型检查相关的模块

import torch  # 导入 PyTorch 库
from transformers.integrations import is_deepspeed_zero3_enabled  # 从 transformers 库导入是否启用 DeepSpeed Zero3 的函数

from ...extras import logging  # 从 extras 模块导入 logging

if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PreTrainedModel, PreTrainedTokenizer  # 导入预训练模型和预训练分词器的类型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)  # 获取嵌入权重的维度
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)  # 计算现有嵌入权重的平均值
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])  # 创建与新嵌入权重相同形状的空张量
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))  # 生成均值为0，标准差为1/√embedding_dim的噪声
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight  # 将新嵌入权重设置为平均值加噪声


def resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""
    Resize token embeddings.  # 调整令牌嵌入的大小
    """
    if is_deepspeed_zero3_enabled():  # 检查是否启用 DeepSpeed Zero3
        import deepspeed  # type: ignore  # 导入 DeepSpeed 库，忽略类型检查

        params = [model.get_input_embeddings().weight]  # 获取模型的输入嵌入权重
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)  # 如果存在输出嵌入且未绑定，添加输出嵌入权重

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)  # 创建 GatheredParameters 上下文
    else:
        context_maybe_zero3 = nullcontext()  # 否则使用 nullcontext

    with context_maybe_zero3:  # 在可能的 Zero3 上下文中执行
        current_embedding_size = model.get_input_embeddings().weight.size(0)  # 获取当前嵌入大小

    if len(tokenizer) > current_embedding_size:  # 如果分词器的大小大于当前嵌入大小
        if getattr(model, "quantization_method", None):  # 检查模型是否有量化方法
            raise ValueError("Cannot resize embedding layers of a quantized model.")  # 如果是量化模型，抛出错误

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):  # 检查输出嵌入是否为线性层
            raise ValueError("Current model does not support resizing embedding layers.")  # 如果不支持，抛出错误

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)  # 调整令牌嵌入的大小
        with context_maybe_zero3:  # 在可能的 Zero3 上下文中执行
            new_embedding_size = model.get_input_embeddings().weight.size(0)  # 获取新的嵌入大小
            num_new_tokens = new_embedding_size - current_embedding_size  # 计算新令牌的数量
            _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)  # 初始化输入嵌入
            _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)  # 初始化输出嵌入

        logger.info_rank0(f"Resized token embeddings from {current_embedding_size} to {new_embedding_size}.")  # 记录调整嵌入大小的信息