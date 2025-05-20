# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch OPT model."""
# 导入必要的库
import random
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的类

import torch
import torch.utils.checkpoint  # 导入检查点工具，用于节省内存
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入各种损失函数

# 从transformers库导入激活函数
from transformers.activations import ACT2FN
# 导入输出类型定义
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel  # 导入预训练模型基类
# 导入各种工具函数
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入OPT配置类
from transformers.models.opt.configuration_opt import OPTConfig


logger = logging.get_logger(__name__)  # 获取日志记录器

# 文档相关的常量定义
_CHECKPOINT_FOR_DOC = "facebook/opt-350m"  # 文档中使用的检查点
_CONFIG_FOR_DOC = "OPTConfig"  # 文档中使用的配置
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"  # 文档中使用的分词器

# 基础模型文档中预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# 序列分类文档相关的常量
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

# 问答任务文档相关的常量
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# OPT预训练模型存档列表，包含不同规模的OPT模型
OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
    # 可以在huggingface.co网站上查看所有OPT模型
]


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    创建用于双向自注意力的因果掩码（实际上是单向的，确保当前位置只能看到之前的位置）。
    """
    bsz, tgt_len = input_ids_shape  # 获取批次大小和目标序列长度
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))  # 创建填充了dtype最小值的掩码矩阵
    mask_cond = torch.arange(mask.size(-1))  # 创建序列索引
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)  # 填充下三角矩阵为0（包括对角线）
    mask = mask.to(dtype)  # 转换掩码类型

    if past_key_values_length > 0:
        # 如果有过去的键值对，则在左侧添加零填充
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1
        )
    # 扩展维度以匹配注意力层的需求 [bsz, 1, tgt_len, tgt_len + past_key_values_length]
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    bsz, src_len = mask.size()  # 获取批次大小和源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len  # 如果未指定目标长度，使用源长度

    # 扩展掩码维度 [bsz, 1, tgt_len, src_len]
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask  # 反转掩码（0变为1，1变为0）

    # 将True位置（原始掩码中的0）填充为dtype的最小值，用于屏蔽注意力
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    该模块学习固定最大大小的位置嵌入。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        # OPT的设置是：如果指定了padding_idx，则将嵌入ID偏移2，并相应地调整num_embeddings。其他模型没有这个技巧。
        self.offset = 2  # 嵌入ID的偏移量
        super().__init__(num_embeddings + self.offset, embedding_dim)  # 调整嵌入数量并初始化父类

    def forward(
        self, attention_mask: torch.LongTensor, past_key_values_length: int = 0
    ):
        """`input_ids_shape` is expected to be [bsz x seqlen].
        `input_ids_shape` 预期形状为 [bsz x seqlen]。
        """
        attention_mask = attention_mask.long()  # 将注意力掩码转换为长整型

        # 基于注意力掩码创建位置索引：累积和 * 掩码，实现只为非掩码位置计算位置编号
        positions = (
            torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
        ).long() - 1

        # 如果有过去的键值对，则截取位置索引
        positions = positions[:, past_key_values_length:]

        # 应用偏移并调用父类的forward方法获取嵌入
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    来自《Attention Is All You Need》论文的多头注意力机制
    """

    def __init__(
        self,
        embed_dim: int,  # 嵌入维度
        num_heads: int,  # 注意力头数
        dropout: float = 0.0,  # 丢弃率
        is_decoder: bool = False,  # 是否为解码器
        bias: bool = True,  # 是否使用偏置
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # 确保嵌入维度能被头数整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于缩放点积注意力
        self.is_decoder = is_decoder  # 是否为解码器模式

        # 创建查询、键、值和输出的线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 键投影
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 值投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 查询投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出投影

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑张量形状以便多头注意力计算 [bsz, seq_len, num_heads, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        key_value_states: Optional[torch.Tensor] = None,  # 用于交叉注意力的键值状态
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值对，用于缓存
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        layer_head_mask: Optional[torch.Tensor] = None,  # 层级头掩码
        output_attentions: bool = False,  # 是否输出注意力权重
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel
        输入形状：批次 x 时间 x 通道
        """

        # 判断是否为交叉注意力层（如果提供了key_value_states则是交叉注意力）
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()  # 获取批次大小和目标序列长度

        # 获取查询投影并应用缩放
        query_states = self.q_proj(hidden_states) * self.scaling
        
        # 获取键和值的投影，根据不同情况处理
        if is_cross_attention and past_key_value is not None:
            # 复用交叉注意力的键值缓存
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # 交叉注意力：从key_value_states计算键值
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # 自注意力但有缓存：复用并扩展键值
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # 普通自注意力：从隐藏状态计算键值
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # 如果是解码器模式，保存键值状态以便后续复用
        if self.is_decoder:
            # 如果是交叉注意力，保存所有交叉注意力的键/值状态
            # 如果是单向自注意力（解码器），保存所有先前解码器的键/值状态
            # 如果是编码器双向自注意力，past_key_value 始终为 None
            past_key_value = (key_states, value_states)

        # 重塑张量以便进行批量矩阵乘法
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)  # 获取源序列长度
        
        # 计算注意力权重：查询和键的点积
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # 检查注意力权重的形状是否符合预期
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 检查注意力掩码的形状
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # 将掩码添加到注意力权重
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            # 确保权重不小于数据类型的最小值
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            # 重塑回原始形状
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # 如果权重是fp16类型，则将softmax计算提升到fp32以提高数值稳定性
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 应用层级头掩码（如果提供）
        if layer_head_mask is not None:
            # 检查层级头掩码的形状
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            # 将掩码应用到注意力权重
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            # 重塑回原始形状
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # 如果需要输出注意力权重
        if output_attentions:
            # 这个操作有点复杂，但它是为了确保attn_weights保持其梯度
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        # 对注意力权重应用dropout
        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # 计算注意力输出：注意力概率和值的加权和
        attn_output = torch.bmm(attn_probs, value_states)

        # 检查注意力输出的形状是否符合预期
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 重塑注意力输出并连接所有头
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # 使用配置中的embed_dim而不是hidden_state，因为在使用张量并行时，attn_output可能在GPU之间分割
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # 应用输出投影
        attn_output = self.out_proj(attn_output)

        # 返回注意力输出、重塑的注意力权重（如果需要）和过去的键值对（如果是解码器）
        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):
    """OPT解码器层，是OPT模型的基本构建模块"""
    def __init__(self, config: OPTConfig):
        """
        初始化OPT解码器层
        
        参数:
            config: OPT模型配置
        """
        super().__init__()
        self.embed_dim = config.hidden_size  # 嵌入维度，决定了层内部的特征维度
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,  # 传递嵌入维度到注意力层
            num_heads=config.num_attention_heads,  # 注意力头的数量
            dropout=config.attention_dropout,  # 注意力dropout率
            is_decoder=True,  # 设为解码器模式
        )
        self.do_layer_norm_before = config.do_layer_norm_before  # 是否在注意力之前应用层归一化
        self.dropout = config.dropout  # 模型的dropout率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数，由配置指定
        
        # 定义层归一化和前馈网络组件
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力后的层归一化
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)  # 前馈网络第一层，扩展维度
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)  # 前馈网络第二层，恢复维度
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最后的层归一化
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        layer_head_mask: Optional[torch.Tensor] = None,  # 层头掩码
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重
        use_cache: Optional[bool] = False,  # 是否使用缓存加速解码
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 缓存的键值对
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # 参数说明:
        # hidden_states: 形状为(batch, seq_len, embed_dim)的输入张量
        # attention_mask: 可选的注意力掩码，形状为(batch, 1, tgt_len, src_len)，填充元素用很大的负值表示
        # layer_head_mask: 可选的特定层的注意力头掩码，形状为(encoder_attention_heads,)
        # output_attentions: 是否返回所有注意力层的注意力张量
        # use_cache: 如果为True，返回past_key_values状态，可用于加速解码
        # past_key_value: 缓存的过去键值投影状态

        # 保存残差连接的输入
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        # 125m, 1.7B, ..., 175B模型在注意力之前应用层归一化
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention - 自注意力机制
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,  # 传递隐藏状态
            past_key_value=past_key_value,  # 传递过去的键值对（如果有）
            attention_mask=attention_mask,  # 传递注意力掩码
            layer_head_mask=layer_head_mask,  # 传递层头掩码
            output_attentions=output_attentions,  # 是否输出注意力权重
        )
        # 对自注意力输出应用dropout
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        # 残差连接：将原始输入与注意力输出相加
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        # 350m模型在注意力之后应用层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected - 全连接前馈网络部分
        # 保存当前隐藏状态的形状
        hidden_states_shape = hidden_states.shape
        # 将隐藏状态重塑为二维张量，以便于线性层处理
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        
        # 保存残差连接输入
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        # 125m, 1.7B, ..., 175B模型在前馈网络之前应用最终层归一化
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        # 前馈网络第一层变换
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数（通常是GELU）
        hidden_states = self.activation_fn(hidden_states)

        # 前馈网络第二层变换
        hidden_states = self.fc2(hidden_states)
        # 应用dropout
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # 残差连接并恢复原始形状
        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        # 350m模型在前馈网络之后应用最终层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        # 准备输出元组，首先包含隐藏状态
        outputs = (hidden_states,)

        # 如果需要，添加注意力权重到输出
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果使用缓存，添加当前键值对到输出
        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):

    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OPTDecoder)):
            module.gradient_checkpointing = value


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`GPT2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """OPT解码器模型的实现"""
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    # 从BART解码器复制的准备解码器注意力掩码的方法
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # 创建因果掩码，将形状从[bsz, seq_len]变换为[bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # 如果序列长度大于1，创建因果掩码（确保当前位置只能看到过去的位置）
            combined_attention_mask = _make_causal_mask(
                input_shape,  # 输入形状
                inputs_embeds.dtype,  # 使用与输入嵌入相同的数据类型
                past_key_values_length=past_key_values_length,  # 过去键值对的长度
            ).to(inputs_embeds.device)  # 移动到与输入嵌入相同的设备

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # 如果提供了注意力掩码，将其扩展为适合注意力机制的形状
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            # 合并扩展的注意力掩码与因果掩码（如果有）
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask  # 返回合并后的注意力掩码

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        query_embeds: Optional[torch.FloatTensor] = None,  # 查询嵌入（BLIP2特有）
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出所有隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """OPT解码器的前向传播函数"""
        
        # ... (省略了部分代码)
        
        # 准备解码器注意力掩码
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # 如果有投影层，应用输入投影
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        # 将输入嵌入与位置嵌入相加
        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        # 解码器层处理
        all_hidden_states = () if output_hidden_states else None  # 存储所有隐藏状态（如果需要）
        all_self_attns = () if output_attentions else None  # 存储所有自注意力权重（如果需要）
        next_decoder_cache = () if use_cache else None  # 存储下一个解码器缓存（如果使用缓存）

        # check if head_mask has a correct number of layers specified if desired
        # 检查头部掩码是否指定了正确数量的层（如果提供）
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        # 遍历解码器层
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # 添加LayerDrop（见https://arxiv.org/abs/1909.11556描述）
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # 保存当前隐藏状态

            # 随机丢弃层的概率（LayerDrop技术）
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue  # 跳过当前层

            # 获取当前层的过去键值对（如果有）
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # 梯度检查点处理（节省内存但稍慢）
            if self.gradient_checkpointing and self.training:
                # 如果使用梯度检查点，不能同时使用缓存
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                # 创建自定义前向函数用于梯度检查点
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        # 对past_key_value使用None
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                # 使用梯度检查点调用解码器层
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                # 正常调用解码器层
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果使用缓存，保存当前层的键值对
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # 如果输出注意力权重，保存当前层的注意力权重
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 如果有最终层归一化，应用它
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        # 如果有输出投影层，应用它
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        # 添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 确定是否有下一个缓存
        next_cache = next_decoder_cache if use_cache else None

        # 根据return_dict决定返回格式
        if not return_dict:
            # 返回元组形式的输出
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        # 返回BaseModelOutputWithPast对象
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,  # 最后的隐藏状态
            past_key_values=next_cache,  # 下一个键值对缓存
            hidden_states=all_hidden_states,  # 所有隐藏状态
            attentions=all_self_attns,  # 所有注意力权重
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    """OPT基础模型，只输出原始隐藏状态，不包含特定任务的预测头"""
    def __init__(self, config: OPTConfig):
        """
        初始化OPT模型
        
        参数:
            config: OPT模型配置对象
        """
        super().__init__(config)
        self.decoder = OPTDecoder(config)  # 创建OPT解码器
        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.decoder.embed_tokens = value

    def get_decoder(self):
        """获取解码器对象"""
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,  # 文档中使用的分词器类
        checkpoint=_CHECKPOINT_FOR_DOC,  # 文档中使用的检查点
        output_type=BaseModelOutputWithPast,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 配置类
        expected_output=_EXPECTED_OUTPUT_SHAPE,  # 预期输出形状
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        query_embeds: Optional[torch.FloatTensor] = None,  # 查询嵌入（BLIP2特有）
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出所有隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典而非元组
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """模型前向传播方法"""

        # 如果未指定输出注意力参数，则使用配置中的默认值
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # 如果未指定输出隐藏状态参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # 如果未指定使用缓存参数，则使用配置中的默认值
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果未指定返回字典参数，则使用配置中的默认值
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # 解码器输出包含(解码器特征, 过去键值对, 解码器隐藏状态, 解码器注意力)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            query_embeds=query_embeds,  # BLIP2模型特有参数
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不需要返回字典，直接返回解码器输出
        if not return_dict:
            return decoder_outputs

        # 返回包装为BaseModelOutputWithPast对象的输出
        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,  # 最后一层隐藏状态
            past_key_values=decoder_outputs.past_key_values,  # 用于加速生成的键值对
            hidden_states=decoder_outputs.hidden_states,  # 所有层的隐藏状态
            attentions=decoder_outputs.attentions,  # 注意力权重
        )


class OPTForCausalLM(OPTPreTrainedModel):
    """OPT因果语言模型，用于生成文本"""
    # 在加载模型时可以忽略缺失的键
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        初始化OPT因果语言模型
        
        参数:
            config: 模型配置
        """
        super().__init__(config)
        self.model = OPTModel(config)  # 创建OPT基础模型

        # the lm_head weight is automatically tied to the embed tokens weight
        # 语言模型头的权重自动与嵌入层的权重绑定
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """获取输出嵌入层（即语言模型头）"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """设置解码器"""
        self.model.decoder = decoder

    def get_decoder(self):
        """获取解码器"""
        return self.model.decoder

    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        query_embeds: Optional[torch.FloatTensor] = None,  # 查询嵌入（BLIP2特有）
        labels: Optional[torch.LongTensor] = None,  # 用于计算损失的标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出所有隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        reduction: Optional[str] = "mean",  # 损失计算的归约方式
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import GPT2Tokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        # 参数说明中文翻译:
        # input_ids: 形状为(batch_size, sequence_length)的输入序列token索引
        # attention_mask: 用于避免在填充token上执行注意力的掩码，1表示非掩码token，0表示掩码token
        # head_mask: 用于将选定的注意力头置零的掩码，1表示非掩码头，0表示掩码头
        # past_key_values: 预计算的隐藏状态，可用于加速顺序解码
        # inputs_embeds: 可选地直接传递嵌入表示，而不是input_ids
        # labels: 用于计算掩码语言建模损失的标签
        # use_cache: 是否返回键值状态以加速解码
        # output_attentions: 是否返回所有注意力层的注意力张量
        # output_hidden_states: 是否返回所有层的隐藏状态
        # return_dict: 是否返回ModelOutput对象而非元组

        # 设置输出参数，如果未指定则使用配置中的默认值
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # 解码器输出包含(解码器特征, 层状态, 解码器隐藏状态, 解码器注意力)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            query_embeds=query_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 通过语言模型头将隐藏状态映射到词汇表大小的logits
        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        # 如果提供了标签，计算损失
        if labels is not None:
            # 截取logits以匹配标签大小
            logits = logits[:, -labels.size(1) :, :]

            # Shift so that tokens < n predict n
            # 移位使得tokens < n 预测 n（自回归预测下一个token）
            shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个位置
            shift_labels = labels[..., 1:].contiguous()  # 去掉第一个位置
            # Flatten the tokens
            # 展平tokens以计算交叉熵损失
            loss_fct = CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )
            # 如果reduction为"none"，则保持每个样本的损失
            if reduction == "none":
                loss = loss.view(shift_logits.size(0), -1).sum(1)

        # 如果不需要返回字典，直接返回元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 返回包装为CausalLMOutputWithPast对象的输出
        return CausalLMOutputWithPast(
            loss=loss,  # 计算的损失（如果有标签）
            logits=logits,  # 预测的logits
            past_key_values=outputs.past_key_values,  # 键值对缓存
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )

    def prepare_inputs_for_generation(
        self,
        input_ids=None,  # 输入token IDs
        query_embeds=None,  # 查询嵌入（BLIP2特有）
        past=None,  # 过去的键值对
        attention_mask=None,  # 注意力掩码
        use_cache=None,  # 是否使用缓存
        **kwargs,  # 其他参数
    ):
        """为生成准备输入
        
        在自回归生成过程中准备每一步的输入
        """
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        # 如果模型用作编码器-解码器模型中的解码器，解码器注意力掩码会即时创建
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = input_ids.new_ones(input_ids.shape)  # 创建全1掩码
        
        # 如果有过去状态（非第一步），只保留最后一个token作为输入
        if past:
            input_ids = input_ids[:, -1:]
            query_embeds = None  # 非第一步不需要查询嵌入
            
        # first step, decoder_cached_states are empty
        # 第一步，解码器缓存状态为空
        return {
            "input_ids": input_ids,
            "query_embeds": query_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        重新排序缓存，用于波束搜索
        
        参数:
            past: 过去的缓存状态
            beam_idx: 波束索引
            
        返回:
            重新排序的缓存
        """
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
