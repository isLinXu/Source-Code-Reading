from typing import Optional, Tuple  # 导入 Optional 和 Tuple 类型，用于类型提示
import warnings  # 导入 warnings 模块，用于发出警告

import torch  # 导入 PyTorch
import transformers  # 导入 HuggingFace Transformers 库
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half
# 从 transformers 的 llama 模型中导入 apply_rotary_pos_emb（旋转位置编码应用函数）、repeat_kv（重复 K/V 头函数）、rotate_half（半旋转函数）

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    # 尝试导入 flash-attn 接口中的无填充 QKV 打包函数
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
    # 如果没有该接口，则回退到可变长度 QKV 打包函数，并重命名为 flash_attn_unpadded_qkvpacked_func

from flash_attn.bert_padding import unpad_input, pad_input
# 从 flash_attn 的 bert_padding 模块中导入 unpad_input（去填充输入）和 pad_input（补填充输出）
from flash_attn import __version__ as flash_attn_version  # 导入 flash_attn 版本号
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)
# 从 flash_attn_interface 中再导入 flash_attn_func（标准 flash-attn 函数）和 flash_attn_varlen_kvpacked_func（可变长度 KV 打包函数）


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
        # 英文注释保留：Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.
        # 中文翻译：已打补丁的 LlamaAttention 不支持输出 attention 权重，将返回 None。

    bsz, q_len, _ = hidden_states.size()
    # 获取 batch size（bsz）、查询序列长度（q_len），忽略最后一个维度

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    # 计算查询投影，并 reshape 为 (bsz, num_heads, q_len, head_dim)

    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # 计算键投影，并 reshape 为 (bsz, num_key_value_heads, q_len, head_dim)

    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)
    # 计算值投影，并 reshape 为 (bsz, num_key_value_heads, q_len, head_dim)

    kv_seq_len = key_states.shape[-2]
    # 当前键/值序列长度

    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    # 如果存在 past_key_value，则将其长度也加到 kv_seq_len 上

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # 生成旋转嵌入的 cos 和 sin 分量，序列长度为 kv_seq_len

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # 应用旋转位置编码到 query_states 和 key_states

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    # 如果有 past_key_value，则在时间维度上拼接历史的 k 和 v

    past_key_value = (key_states, value_states) if use_cache else None
    # 如果需要缓存，则更新 past_key_value，否则置 None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    # 如果 num_key_value_heads < num_heads，则重复 k/v 头，使之与 num_heads 对齐

    # Transform the data into the format required by flash attention
    # 将数据转换为 flash-attn 所需格式
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    # 在第 2 维将 Q、K、V 堆叠，形状为 (bsz, num_heads, 3, seq_len, head_dim)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    # 转置以得到 (bsz, seq_len, 3, num_heads, head_dim)
    key_padding_mask = attention_mask  # 将 attention_mask 作为 key_padding_mask

    if key_padding_mask is None:
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        # 展平 batch 和 seq_len，用于无填充场景
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        # 构造 cu_q_lens：每个样本的起始偏移
        max_s = q_len  # 最大序列长度
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        # 调用无填充 QKV-packed flash-attn，开启 causal
        output = output.view(bsz, q_len, -1)
        # 恢复输出形状为 (bsz, seq_len, num_heads*head_dim)
    else:
        qkv = qkv.reshape(bsz, q_len, -1)
        # 在有 padding 情况下，先将 qkv reshape 为 (bsz, seq_len, 3*num_heads*head_dim)
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        # 去填充，返回去填充后的 qkv，indices，cu_q_lens，max_s
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        # reshape 为无填充格式
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        # 调用无填充 QKV-packed flash-attn
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        # 展平输出
        output = pad_input(output_unpad, indices, bsz, q_len)
        # 回复填充，恢复 (bsz, seq_len, hidden_dim)

    return self.o_proj(output), None, past_key_value
    # 返回输出投影、None（无 attentions）、以及 past_key_value


def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    # 扩展 position_ids 维度，为 gather 做准备
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    # 重复以匹配 cos_sin 的维度
    bsz = gather_indices.shape[0]  # batch size
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    # 从 cos_sin 中按位置索引出对应的 cos 和 sin
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    # 应用旋转：x * cos + rotate_half(x) * sin
    return q, k  # 返回旋转后 q 和 k


def forward_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
        # 英文注释保留：Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.
        # 中文翻译：已打补丁的 LlamaAttention 推理不支持输出 attention 权重，将返回 None。

    bsz, q_len, _ = hidden_states.size()  # batch size 和序列长度
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)
    # 获取 num_key_value_heads，如果没有则退到 num_heads

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # 分别计算 q, k, v 投影并 reshape

    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]  # 当前 k 的序列长度
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len
    # 如果有缓存，增加 past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    # 生成旋转嵌入
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)
    # 推理时使用 apply_rotary_pos_emb_inference

    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # 英文注释保留：past_key_value support requires flash-attn >= 2.1.0
        # 中文翻译：支持 past_key_value 需要 flash-attn 版本 >= 2.1.0
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)
        # 拼接历史 k 和 v

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None
    # 更新缓存

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
        # 无 mask 情况下直接调用 flash_attn_func
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # 去填充查询
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        # 去填充 k, v
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        # 调用可变长度 KV-packed flash-attn
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        # 展平输出
        output = pad_input(output_unpad, indices, bsz, q_len)
        # 恢复填充

    return self.o_proj(output), None, past_key_value
    # 返回输出投影、None、past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
# 禁用 LlamaModel 中对 attention mask 的额外变换，因为 flash-attn 要求 attention_mask 与 key_padding_mask 相同
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask  # 直接返回原始 attention_mask


def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )
    # 如果有历史 key/value，则在前面补上全 True 的 mask

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples
    # 如果全 True，则返回 None，以使用更快的分支


def replace_llama_attn_with_flash_attn(inference=False):
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    # 获取当前 CUDA 设备能力 (major, minor)
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
        # 英文注释保留：Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward...
        # 中文翻译：训练时仅在 A100 或 H100 GPU 上支持 flash-attn（因为 head_dim>64 的反向不支持）。

    if inference:
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_inference
        # 推理模式下替换 LlamaModel 和 LlamaAttention 的方法
    else:
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
            _prepare_decoder_attention_mask
        )
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
        # 训练模式下替换 LlamaModel 和 LlamaAttention 的方法