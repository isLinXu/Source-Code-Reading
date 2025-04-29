"""
Directly copied the code from https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/modules/llama_attn_hijack.py and made some adjustments
"""
# 英文注释保留：Directly copied the code from https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/modules/llama_attn_hijack.py and made some adjustments
# 中文：直接复制自上述链接并做了部分调整

import logging  # 导入日志模块，用于记录错误和调试信息
import math  # 导入数学模块，用于执行数学操作（如开平方）
from typing import Optional, Tuple  # 导入 Optional 和 Tuple 类型，用于类型提示

import torch  # 导入 PyTorch 库，用于张量运算
import transformers.models.llama.modeling_llama  # 导入 HuggingFace 中 Llama 模型的实现
from torch import nn  # 导入神经网络模块

try:
    import xformers.ops  # 尝试导入 xformers.ops，以使用高效注意力实现
except ImportError:
    logging.error("xformers not found! Please install it before trying to use it.")
    # 如果导入失败，则打印错误，提示用户安装 xformers


def replace_llama_attn_with_xformers_attn():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward
    # 用 xformers_forward 替换 LlamaAttention.forward，以启用 xformers 加速


def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # pylint: disable=duplicate-code
    bsz, q_len, _ = hidden_states.size()
    # bsz: batch size，q_len: 查询序列长度，_: 隐藏维度（hidden_size）

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    # 线性投影 Q，然后 reshape 为 (bsz, q_len, num_heads, head_dim)，再转为 (bsz, num_heads, q_len, head_dim)

    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    # 同上，对 K 做投影和维度转换

    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    # 同上，对 V 做投影和维度转换

    kv_seq_len = key_states.shape[-2]
    # 当前 K/V 的序列长度

    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    # 如果存在历史缓存，则将其长度累加

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # 生成旋转嵌入的 cos 和 sin 分量，长度为 kv_seq_len

    (
        query_states,
        key_states,
    ) = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # 应用旋转位置编码到 Q 和 K

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    # 如果有过去的 key/value，则在时间维度拼接

    past_key_value = (key_states, value_states) if use_cache else None
    # 如果需要缓存，则返回新的 past_key_value，否则置 None

    # We only apply xformers optimizations if we don't need to output the whole attention matrix
    # 仅在不需要输出完整注意力矩阵时，才使用 xformers 优化
    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # 转回 (bsz, q_len, num_heads, head_dim)

        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # 我们知道 transformers 中的 attention_mask 要么是下三角矩阵，要么全 0，这是一个不优雅的 hack。
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # 如果没有 mask 或上三角 (0,1) 处为 0，则视为全 0 mask
            attn_output = xformers.ops.memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=None
            )
        else:
            # 否则传入下三角掩码
            attn_output = xformers.ops.memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=xformers.ops.LowerTriangularMask(),
            )
        attn_weights = None
        # 不返回注意力权重
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        # 计算未归一化的注意力分数 Q·K^T / sqrt(head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # 验证注意力分数的形状是否正确

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            # 验证 attention_mask 形状

            attn_weights = attn_weights + attention_mask
            # 应用 mask（将被掩蔽位置设为 -inf）
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            # 防止数值下溢

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # 先用 fp32 做 softmax 再转回原 dtype，以提高数值稳定性

        attn_output = torch.matmul(attn_weights, value_states)
        # 加权求和得到注意力输出

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # 验证输出形状

        attn_output = attn_output.transpose(1, 2)
        # 转回 (bsz, q_len, num_heads, head_dim)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    # 将多头输出拼回 (bsz, q_len, hidden_size)

    attn_output = self.o_proj(attn_output)
    # 最后做一次线性映射

    return attn_output, attn_weights, past_key_value
    # 返回：(输出张量, 注意力权重或 None, 更新后的 past_key_value)