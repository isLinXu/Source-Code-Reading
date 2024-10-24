"""
Directly copied the code from https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/modules/llama_attn_hijack.py and made some adjustments
"""

import logging
import math
from typing import Optional, Tuple

import torch
import transformers.models.llama.modeling_llama
from torch import nn

try:
    import xformers.ops
except ImportError:
    logging.error("xformers not found! Please install it before trying to use it.")


def replace_llama_attn_with_xformers_attn():
    """
    用xformers库中的前向传播函数替换LLaMA模型中的注意力机制前向传播函数。
    这是为了利用xformers库的优化，提高模型的计算效率和性能。
    """
    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward


def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    执行xformers前向传播。

    参数:
    - hidden_states: 隐藏状态张量，形状为(bsz, q_len, hidden_size)。
    - attention_mask: 注意力掩码张量，可选。
    - position_ids: 位置ID张量，可选。
    - past_key_value: 过去的键值对张量，用于加速推理，可选。
    - output_attentions: 是否输出注意力权重。
    - use_cache: 是否使用缓存。

    返回:
    - attn_output: 注意力输出张量。
    - attn_weights: 注意力权重张量，如果output_attentions为True则返回。
    - past_key_value: 过去的键值对张量，如果use_cache为True则返回。
    """
    # pylint: disable=duplicate-code
    bsz, q_len, _ = hidden_states.size()

    # 投影隐藏状态到查询、键和值空间
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    (
        query_states,
        key_states,
    ) = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        # 重用k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # We only apply xformers optimizations if we don't need to output the whole attention matrix
    # 我们只在不需要输出整个注意力矩阵时应用xformers优化
    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        # 这是一个肮脏的黑客方法。我们知道transformers中的attention_mask要么是LowerTriangular要么是全零。
        # 因此，我们检查上三角部分的一个元素是否为零。如果是，则mask全为零。
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            # 输入和输出应为形式(bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=None
            )
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            # 输入和输出应为形式(bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=xformers.ops.LowerTriangularMask(),
            )
        attn_weights = None
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        # 将注意力权重提升到fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value
