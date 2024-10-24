from typing import Optional, Tuple
import warnings

import torch

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    LlamaAttention前向传播函数。

    参数:
    - hidden_states: 隐藏状态张量，形状为(batch_size, sequence_length, hidden_size)。
    - attention_mask: 注意力掩码张量，可选。
    - position_ids: 位置ID张量，可选。
    - past_key_value: 过去的键值对张量，用于加速推理，可选。
    - output_attentions: 是否输出注意力权重，可选，默认为False。
    - use_cache: 是否使用缓存，用于加速推理，可选，默认为False。

    返回:
    - output: 注意力层的输出张量。
    - None: 由于不支持输出注意力权重，此位置始终返回None。
    - past_key_value: 如果use_cache为True，则返回当前层的键值对，否则返回None。
    """
    # 检查是否请求输出注意力权重，如果是，则发出警告，因为当前实现不支持。
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    # 获取批量大小(bsz)、查询长度(q_len)和隐藏状态的维度
    bsz, q_len, _ = hidden_states.size()

    # 计算查询、键和值状态
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)

    # 计算键值序列长度
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # 应用旋转位置嵌入
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # 如果提供了过去的键值对，则将当前的键值状态与之合并起来
    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # 如果使用缓存，则保存当前的键值状态作为past_key_value
    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    # 如果键值头数小于总头数，则重复键值头
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Transform the data into the format required by flash attention
    # 将查询、键和值状态堆叠起来，准备进行注意力计算
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    key_padding_mask = attention_mask

    # 根据是否存在注意力掩码，选择合适的注意力计算方式
    # 根据key_padding_mask是否为空，决定是否使用flash_attn进行优化计算
    if key_padding_mask is None:
        # 如果key_padding_mask为空，直接对qkv进行重塑，准备进行flash_attn计算
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        # 计算累积的query长度，用于flash_attn计算
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        # 设置最大序列长度
        max_s = q_len
        # 使用flash_attn函数计算注意力输出
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        # 将输出重塑为期望的形状
        output = output.view(bsz, q_len, -1)
    else:
        # 如果key_padding_mask不为空，首先根据mask对输入进行重塑和压缩，以优化计算
        qkv = qkv.reshape(bsz, q_len, -1)
        # 对输入进行unpad，移除不需要计算的部分，提高计算效率
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        # 使用flash_attn函数计算unpad后的注意力输出
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        # 将unpad的输出重塑为期望的形状
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        # 将unpad的输出重新填充回原始形状
        output = pad_input(output_unpad, indices, bsz, q_len)

    # 返回注意力层的输出、None（因为不输出注意力权重）和可能的缓存键值对
    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    """
    准备解码器注意力掩码。

    该函数用于准备解码器的注意力掩码，以适应特定的输入形状和嵌入。它考虑了过去的键值长度，
    以便在注意力机制中正确地应用掩码。

    参数:
        attention_mask (torch.Tensor): 注意力掩码，用于指示哪些部分应该被关注。
        input_shape (tuple): 输入的形状。
        inputs_embeds (torch.Tensor): 输入的嵌入表示。
        past_key_values_length (int): 过去的键值的长度。

    返回:
        torch.Tensor: 准备好的解码器注意力掩码。
    """
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    """
    用Flash注意力替换LLaMA模型中的注意力。

    该函数检查CUDA设备的能力，并根据设备的主版本号决定是否发出警告。然后，它用Flash注意力
    替换LLaMA模型中的注意力机制。
    """
    # 检查CUDA设备能力，以确定是否支持Flash attention
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        # 如果CUDA主版本号小于8，则发出警告
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    # 为Llama模型准备decoder注意力掩码的函数
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    # Llama注意力机制的前向传播函数
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
