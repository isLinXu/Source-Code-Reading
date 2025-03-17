# Copyright 2024 EleutherAI, HuggingFace Inc., Yukang Chen, and the LlamaFactory team.
#
# This code is based on the EleutherAI's GPT-NeoX and the HuggingFace's Transformers libraries.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
# This code is also inspired by the original LongLoRA implementation.
# https://github.com/dvlab-research/LongLoRA/blob/main/llama_attn_replace.py
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
from typing import TYPE_CHECKING, Optional, Tuple  # 导入类型检查、可选类型和元组类型

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import transformers  # 导入 transformers 库
from transformers.models.llama.modeling_llama import (  # 从 transformers 中导入 llama 模型相关的类和函数
    Cache,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils.versions import require_version  # 从 transformers 导入版本检查函数

from ...extras import logging  # 从 extras 模块导入 logging
from ...extras.constants import SUPPORTED_CLASS_FOR_S2ATTN  # 从 extras.constants 导入支持的 S2ATTN 类
from ...extras.packages import is_transformers_version_greater_than  # 从 extras.packages 导入版本比较函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig  # 导入预训练配置的类型
    from ...hparams import ModelArguments  # 导入模型参数的类型


transformers_logger = transformers.utils.logging.get_logger(__name__)  # 获取 transformers 的日志记录器


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
def llama_attention_forward(
    self: "LlamaAttention",  # 当前类的实例
    hidden_states: "torch.Tensor",  # 输入的隐藏状态
    attention_mask: Optional["torch.Tensor"] = None,  # 可选的注意力掩码
    position_ids: Optional["torch.LongTensor"] = None,  # 可选的位置 ID
    past_key_value: Optional["Cache"] = None,  # 可选的过去的键值缓存
    output_attentions: bool = False,  # 是否输出注意力权重
    cache_position: Optional["torch.LongTensor"] = None,  # 可选的缓存位置
    position_embeddings: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,  # 可选的位置嵌入
    **kwargs,  # 其他关键字参数
) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[Tuple["torch.Tensor"]]]:
    bsz, q_len, _ = hidden_states.size()  # 获取批次大小和序列长度

    query_states: "torch.Tensor" = self.q_proj(hidden_states)  # 计算查询状态
    key_states: "torch.Tensor" = self.k_proj(hidden_states)  # 计算键状态
    value_states: "torch.Tensor" = self.v_proj(hidden_states)  # 计算值状态

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # 调整查询状态的形状
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # 调整键状态的形状
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # 调整值状态的形状

    if position_embeddings is None:  # 如果没有位置嵌入
        cos, sin = self.rotary_emb(value_states, position_ids)  # 计算旋转位置嵌入
    else:
        cos, sin = position_embeddings  # 使用提供的位置嵌入

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 应用旋转位置嵌入

    if past_key_value is not None:  # 如果存在过去的键值
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # 缓存参数
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)  # 更新键值

    key_states = repeat_kv(key_states, self.num_key_value_groups)  # 重复键状态
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # 重复值状态

    if getattr(self.config, "group_size_ratio", None) and self.training:  # 如果启用了分组大小比例并且正在训练
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))  # 计算组大小
        assert q_len % groupsz == 0, f"q_len {q_len} should be divisible by group size {groupsz}."  # 确保 q_len 可以被组大小整除
        num_groups = q_len // groupsz  # 计算组的数量

        def shift(state: "torch.Tensor") -> "torch.Tensor":  # 定义状态移动函数
            state = state.transpose(1, 2)  # 转置状态
            state = torch.cat(  # 连接状态
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2 :].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)  # 调整形状并转置

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)  # 移动状态
        if attention_mask is not None:  # 如果存在注意力掩码
            attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)  # 重复注意力掩码

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力权重

    if attention_mask is not None:  # 如果存在注意力掩码
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]  # 获取因果掩码
        attn_weights = attn_weights + causal_mask  # 添加因果掩码

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # 计算注意力权重并转换为 fp32
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)  # 应用 dropout
    attn_output = torch.matmul(attn_weights, value_states)  # 计算注意力输出
    attn_output = attn_output.transpose(1, 2).contiguous()  # 转置输出并确保内存连续

    if getattr(self.config, "group_size_ratio", None) and self.training:  # 如果启用了分组大小比例并且正在训练
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)  # 调整输出形状
        attn_output = torch.cat(  # 连接输出
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2 :].roll(groupsz // 2, dims=1),
            ),
            dim=2,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)  # 调整输出形状
    attn_output = self.o_proj(attn_output)  # 通过输出投影层

    if not output_attentions:  # 如果不输出注意力权重
        attn_weights = None  # 设置注意力权重为 None

    return attn_output, attn_weights, past_key_value  # 返回注意力输出、注意力权重和过去的键值


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
def llama_flash_attention_2_forward(
    self: "LlamaFlashAttention2",  # 当前类的实例
    hidden_states: "torch.Tensor",  # 输入的隐藏状态
    attention_mask: Optional["torch.Tensor"] = None,  # 可选的注意力掩码
    position_ids: Optional["torch.LongTensor"] = None,  # 可选的位置 ID
    past_key_value: Optional["Cache"] = None,  # 可选的过去的键值缓存
    output_attentions: bool = False,  # 是否输出注意力权重
    cache_position: Optional["torch.LongTensor"] = None,  # 可选的缓存位置
    position_embeddings: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,  # 可选的位置嵌入
    **kwargs,  # 其他关键字参数
) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[Tuple["torch.Tensor"]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False  # LlamaFlashAttention2 不支持输出注意力权重

    bsz, q_len, _ = hidden_states.size()  # 获取批次大小和序列长度

    query_states: "torch.Tensor" = self.q_proj(hidden_states)  # 计算查询状态
    key_states: "torch.Tensor" = self.k_proj(hidden_states)  # 计算键状态
    value_states: "torch.Tensor" = self.v_proj(hidden_states)  # 计算值状态

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # 调整查询状态的形状
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # 调整键状态的形状
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # 调整值状态的形状

    if position_embeddings is None:  # 如果没有位置嵌入
        cos, sin = self.rotary_emb(value_states, position_ids)  # 计算旋转位置嵌入
    else:
        cos, sin = position_embeddings  # 使用提供的位置嵌入

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 应用旋转位置嵌入

    if past_key_value is not None:  # 如果存在过去的键值
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # 缓存参数
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)  # 更新键值

    key_states = repeat_kv(key_states, self.num_key_value_groups)  # 重复键状态
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # 重复值状态

    # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
    query_states = query_states.transpose(1, 2)  # 转置查询状态
    key_states = key_states.transpose(1, 2)  # 转置键状态
    value_states = value_states.transpose(1, 2)  # 转置值状态

    dropout_rate = self.attention_dropout if self.training else 0.0  # 设置 dropout 率

    input_dtype = query_states.dtype  # 获取输入数据类型
    if input_dtype == torch.float32:  # 如果输入数据类型为 float32
        if torch.is_autocast_enabled():  # 如果启用了自动类型转换
            target_dtype = torch.get_autocast_gpu_dtype()  # 获取目标数据类型
        elif hasattr(self.config, "_pre_quantization_dtype"):  # 如果配置中有预量化数据类型
            target_dtype = self.config._pre_quantization_dtype  # 使用预量化数据类型
        else:
            target_dtype = self.q_proj.weight.dtype  # 否则使用查询投影权重的数据类型

        transformers_logger.warning_once("The input hidden states seems to be silently casted in float32.")  # 记录警告：输入隐藏状态似乎被默默转换为 float32。
        query_states = query_states.to(target_dtype)  # 转换查询状态的数据类型
        key_states = key_states.to(target_dtype)  # 转换键状态的数据类型
        value_states = value_states.to(target_dtype)  # 转换值状态的数据类型

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))  # 计算组大小
        assert q_len % groupsz == 0, f"q_len {q_len} should be divisible by group size {groupsz}."  # 确保 q_len 可以被组大小整除
        num_groups = q_len // groupsz  # 计算组的数量

        def shift(state: "torch.Tensor") -> "torch.Tensor":  # 定义状态移动函数
            state = torch.cat(  # 连接状态
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2 :].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)  # 调整形状

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)  # 移动状态
        if attention_mask is not None:  # 如果存在注意力掩码
            attention_mask = attention_mask[:, :groupsz].repeat(num_groups, 1)  # 重复注意力掩码

    if is_transformers_version_greater_than("4.43.0"):  # 如果 transformers 版本大于 4.43.0
        from transformers.modeling_flash_attention_utils import _flash_attention_forward  # 导入 FlashAttention 前向计算函数

        attn_output: "torch.Tensor" = _flash_attention_forward(  # 计算注意力输出
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_states.size(1),
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
    else:
        attn_output: "torch.Tensor" = self._flash_attention_forward(  # 使用自定义的 FlashAttention 前向计算
            query_states, key_states, value_states, attention_mask, query_states.size(1), dropout=dropout_rate
        )

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)  # 调整输出形状
        attn_output = torch.cat(  # 连接输出
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2 :].roll(groupsz // 2, dims=1),
            ),
            dim=2,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()  # 调整输出形状并确保内存连续
    attn_output = self.o_proj(attn_output)  # 通过输出投影层

    if not output_attentions:  # 如果不输出注意力权重
        attn_weights = None  # 设置注意力权重为 None

    return attn_output, attn_weights, past_key_value  # 返回注意力输出、注意力权重和过去的键值


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
def llama_sdpa_attention_forward(
    self: "LlamaSdpaAttention",  # 当前类的实例
    hidden_states: "torch.Tensor",  # 输入的隐藏状态
    attention_mask: Optional["torch.Tensor"] = None,  # 可选的注意力掩码
    position_ids: Optional["torch.LongTensor"] = None,  # 可选的位置 ID
    past_key_value: Optional["Cache"] = None,  # 可选的过去的键值缓存
    output_attentions: bool = False,  # 是否输出注意力权重
    cache_position: Optional["torch.LongTensor"] = None,  # 可选的缓存位置
    position_embeddings: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,  # 可选的位置嵌入
    **kwargs,  # 其他关键字参数
) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[Tuple["torch.Tensor"]]]:
    if output_attentions:  # 如果请求输出注意力权重
        transformers_logger.warning_once(  # 记录警告：SDPA 不支持输出注意力权重
            "SDPA does not support `output_attentions=True`. Falling back to the vanilla attention"
        )
        return llama_attention_forward(  # 回退到标准注意力计算
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()  # 获取批次大小和序列长度

    query_states: "torch.Tensor" = self.q_proj(hidden_states)  # 计算查询状态
    key_states: "torch.Tensor" = self.k_proj(hidden_states)  # 计算键状态
    value_states: "torch.Tensor" = self.v_proj(hidden_states)  # 计算值状态

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # 调整查询状态的形状
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # 调整键状态的形状
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # 调整值状态的形状

    if position_embeddings is None:  # 如果没有位置嵌入
        cos, sin = self.rotary_emb(value_states, position_ids)  # 计算旋转位置嵌入
    else:
        cos, sin = position_embeddings  # 使用提供的位置嵌入

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 应用旋转位置嵌入

    if past_key_value is not None:  # 如果存在过去的键值
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # 缓存参数
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)  # 更新键值

    key_states = repeat_kv(key_states, self.num_key_value_groups)  # 重复键状态
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # 重复值状态

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))  # 计算组大小
        assert q_len % groupsz == 0, f"q_len {q_len} should be divisible by group size {groupsz}."  # 确保 q_len 可以被组大小整除
        num_groups = q_len // groupsz  # 计算组的数量

        def shift(state: "torch.Tensor") -> "torch.Tensor":  # 定义状态移动函数
            state = state.transpose(1, 2)  # 转置状态
            state = torch.cat(  # 连接状态
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2 :].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)  # 调整形状并返回

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)  # 移动状态
        if attention_mask is not None:  # 如果存在注意力掩码
            attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)  # 重复注意力掩码

    causal_mask = attention_mask  # 设置因果掩码
    if attention_mask is not None:  # 如果存在注意力掩码
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]  # 获取因果掩码

    if query_states.device.type == "cuda" and causal_mask is not None:  # 避免 PyTorch 的错误
        query_states = query_states.contiguous()  # 确保查询状态在内存中是连续的
        key_states = key_states.contiguous()  # 确保键状态在内存中是连续的
        value_states = value_states.contiguous()  # 确保值状态在内存中是连续的

    is_causal = True if causal_mask is None and q_len > 1 else False  # 判断是否为因果注意力
    attn_output = torch.nn.functional.scaled_dot_product_attention(  # 计算缩放点积注意力
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,  # 设置 dropout 率
        is_causal=is_causal,  # 是否为因果注意力
    )
    attn_output = attn_output.transpose(1, 2).contiguous()  # 转置输出并确保内存连续

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)  # 调整输出形状
        attn_output = torch.cat(  # 连接输出
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2 :].roll(groupsz // 2, dims=1),
            ),
            dim=2,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)  # 调整输出形状
    attn_output = self.o_proj(attn_output)  # 通过输出投影层

    return attn_output, None, past_key_value  # 返回注意力输出、None 和过去的键值


def _apply_llama_patch() -> None:  # 定义应用 llama 补丁的函数
    require_version("transformers>=4.41.2,<=4.46.1", "To fix: pip install transformers>=4.41.2,<=4.46.1")  # 检查 transformers 版本
    LlamaAttention.forward = llama_attention_forward  # 替换 LlamaAttention 的前向计算方法
    LlamaFlashAttention2.forward = llama_flash_attention_2_forward  # 替换 LlamaFlashAttention2 的前向计算方法
    LlamaSdpaAttention.forward = llama_sdpa_attention_forward  # 替换 LlamaSdpaAttention 的前向计算方法


def configure_longlora(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.shift_attn:  # 如果不可训练或未启用 shift_attn
        return  # 直接返回

    logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

    if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_S2ATTN:  # 如果模型类型在支持的类中
        setattr(config, "group_size_ratio", 0.25)  # 设置组大小比例
        _apply_llama_patch()  # 应用 llama 补丁
        logger.info_rank0("Using shift short attention with group_size_ratio=1/4.")  # 记录信息：使用组大小比例为 1/4 的短注意力
    else:
        logger.warning_rank0("Current model does not support shift short attention.")  # 记录警告：当前模型不支持短注意力