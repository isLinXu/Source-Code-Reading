# Copyright 2024 Musab Gultekin and the LlamaFactory team.
#
# This code is based on the Musab Gultekin's functionary library.
# https://github.com/MeetKai/functionary/blob/main/functionary/train/packing/monkey_patch_packing.py
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
#
# MIT License
#
# Copyright (c) 2023 Musab Gultekin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import TYPE_CHECKING, Tuple  # 导入类型检查和元组类型

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式神经网络模块
from transformers.utils.versions import require_version  # 从 transformers 导入版本检查函数

from ...extras import logging  # 从 extras 导入日志模块
from ...extras.constants import SUPPORTED_CLASS_FOR_BLOCK_DIAG_ATTN  # 从 extras.constants 导入支持的块对角注意力的类
from ...extras.packages import is_transformers_version_greater_than  # 从 extras.packages 导入检查 transformers 版本的函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig  # 导入预训练配置的类型

    from ...hparams import ModelArguments  # 导入模型参数的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    r"""
    Gets the sequence lengths in the current batch.  # 获取当前批次中的序列长度

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [2, 3, 1, 2, 3]
    ```
    """
    bsz = attention_mask.size(0)  # 获取批次大小
    dtype, device = attention_mask.dtype, attention_mask.device  # 获取张量的数据类型和设备
    max_num = torch.max(attention_mask).item()  # 获取注意力掩码中的最大值
    counts: "torch.Tensor" = torch.zeros((bsz, max_num), dtype=dtype, device=device)  # 初始化计数张量
    for i in range(max_num):  # 遍历最大值
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)  # 计算每个序列中 i+1 的出现次数

    counts = counts.flatten()  # 将计数张量展平
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]  # 获取非零计数的序列长度
    return seqlens  # 返回序列长度


def get_unpad_data(attention_mask: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", int]:
    r"""
    Prepares the indices and seqlens for flash attn varlen function.  # 为闪存注意力变长函数准备索引和序列长度

    Returns:
        indices: indices of non-masked tokens from the flattened sequence.  # 返回展平序列中未被掩码的令牌的索引
        cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.  # 当前批次中的累积序列长度，总是从 0 开始
        max_seqlen_in_batch: the largest seqlen in the current batch.  # 当前批次中最大的序列长度

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    [0, 2, 5, 6, 8, 11]
    3
    ```
    """
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)  # 获取当前批次的序列长度
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()  # 获取未被掩码的令牌的索引
    max_seqlen_in_batch = seqlens_in_batch.max().item()  # 获取当前批次中的最大序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))  # 计算累积序列长度并填充
    return indices, cu_seqlens, max_seqlen_in_batch  # 返回索引、累积序列长度和最大序列长度


def _patch_for_block_diag_attn(model_type: str) -> None:
    require_version("transformers>=4.41.2,<=4.46.1", "To fix: pip install transformers>=4.41.2,<=4.46.1")  # 检查 transformers 版本
    if is_transformers_version_greater_than("4.43.0"):  # 如果 transformers 版本大于 4.43.0
        import transformers.modeling_flash_attention_utils  # 导入闪存注意力模型的工具

        transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
        return  # 直接返回

    import transformers.models  # 导入 transformers 模型

    if model_type == "cohere":  # 如果模型类型为 cohere
        transformers.models.cohere.modeling_cohere._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "falcon":  # 如果模型类型为 falcon
        transformers.models.falcon.modeling_falcon._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "gemma":  # 如果模型类型为 gemma
        transformers.models.gemma.modeling_gemma._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "gemma2":  # 如果模型类型为 gemma2
        transformers.models.gemma2.modeling_gemma2._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "llama":  # 如果模型类型为 llama
        transformers.models.llama.modeling_llama._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "mistral":  # 如果模型类型为 mistral
        transformers.models.mistral.modeling_mistral._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "phi":  # 如果模型类型为 phi
        transformers.models.phi.modeling_phi._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "phi3":  # 如果模型类型为 phi3
        transformers.models.phi3.modeling_phi3._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "qwen2":  # 如果模型类型为 qwen2
        transformers.models.qwen2.modeling_qwen2._get_unpad_data = get_unpad_data  # 替换未填充数据的函数
    elif model_type == "starcoder2":  # 如果模型类型为 starcoder2
        transformers.models.starcoder2.modeling_starcoder2._get_unpad_data = get_unpad_data  # 替换未填充数据的函数


def configure_packing(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.block_diag_attn:  # 如果不可训练或未启用块对角注意力
        return  # 直接返回

    model_type = getattr(config, "model_type", None)  # 获取模型类型
    if model_type in SUPPORTED_CLASS_FOR_BLOCK_DIAG_ATTN:  # 如果模型类型在支持的块对角注意力类中
        _patch_for_block_diag_attn(model_type)  # 进行块对角注意力的补丁
        logger.info_rank0("Using block diagonal attention for sequence packing without cross-attention.")  # 记录使用块对角注意力的信息
    else:  # 如果模型类型不支持
        raise ValueError("Current model does not support block diagonal attention.")  # 抛出不支持块对角注意力的错误