# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import Optional, Tuple  # 从typing模块导入Optional和Tuple类型

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
from torch import nn  # 从torch模块导入nn子模块


@dataclass
class ModelArgs:
    dim: int = 4096  # 模型的维度
    n_layers: int = 32  # 模型的层数
    n_heads: int = 32  # 注意力头的数量
    n_kv_heads: Optional[int] = None  # 可选的键值头数量
    vocab_size: int = -1  # 词汇表大小，稍后由分词器定义
    multiple_of: int = 256  # 确保SwiGLU隐藏层大小是大2的倍数
    ffn_dim_multiplier: Optional[float] = None  # 可选的FFN维度乘数
    norm_eps: float = 1e-5  # 归一化的epsilon值

    max_batch_size: int = 32  # 最大批处理大小
    max_seq_len: int = 32768  # 最大序列长度
    # 如果为`True`，则每个变换块初始化使用其层ID，如果
    # `False`，则每个块使用变换块的总数
    depth_init: bool = True  # 深度初始化标志


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """  # 预计算给定维度的复指数频率张量
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 计算频率
    t = torch.arange(end, device=freqs.device)  # 创建时间索引
    freqs = torch.outer(t, freqs).float()  # 计算外积并转换为浮点数
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 创建复数张量
    return freqs_cis  # 返回复数频率张量


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """  # 重新调整频率张量以便与另一个张量进行广播
    ndim = x.ndim  # 获取目标张量的维度
    assert 0 <= 1 < ndim  # 确保维度有效
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # 确保频率张量形状匹配
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # 计算新形状
    return freqs_cis.view(*shape)  # 返回调整后的频率张量


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """  # 将旋转嵌入应用于输入张量
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 将查询张量视为复数
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # 将键张量视为复数
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # 调整频率张量以便广播
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # 应用旋转嵌入并返回实数张量
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # 应用旋转嵌入并返回实数张量
    return xq_out.type_as(xq), xk_out.type_as(xk)  # 返回调整后的查询和键张量


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""  # 重复键值张量
    bs, slen, n_kv_heads, head_dim = x.shape  # 获取张量的形状
    if n_rep == 1:  # 如果不需要重复
        return x  # 直接返回原张量
    return (
        x[:, :, :, None, :]  # 在维度2上添加新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展张量
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新调整形状
    )


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """  # 初始化RMSNorm归一化层
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()  # 调用父类构造函数
        self.eps = eps  # 保存epsilon值
        self.weight = nn.Parameter(torch.ones(dim))  # 创建可学习的缩放参数

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算RMS归一化

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)  # 执行归一化并返回结果
        return output * self.weight  # 返回归一化后的输出乘以缩放参数

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # 重置权重为1


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_local_kv_heads (int): Number of local key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """  # 多头注意力模块
    def __init__(self, model_args: ModelArgs):
        super().__init__()  # 调用父类构造函数
        self.n_heads = model_args.n_heads  # 保存注意力头的数量
        self.n_kv_heads = (  # 保存键值头的数量
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads  # 计算本地头的重复次数
        self.head_dim = model_args.dim // model_args.n_heads  # 计算每个头的维度

        self.wq = nn.Linear(  # 创建查询的线性变换
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 创建键的线性变换
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 创建值的线性变换
        self.wo = nn.Linear(  # 创建输出的线性变换
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):  # 初始化线性层权重
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)  # 初始化输出层权重

    def forward(  # 前向传播函数
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """  # 注意力模块的前向传播
        bsz, seqlen, _ = x.shape  # 获取输入张量的形状
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # 计算查询、键和值

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)  # 调整查询张量的形状
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)  # 调整键张量的形状
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)  # 调整值张量的形状

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)  # 应用旋转嵌入

        keys = repeat_kv(xk, self.n_rep)  # 重复键张量
        values = repeat_kv(xv, self.n_rep)  # 重复值张量

        xq = xq.transpose(1, 2)  # 转置查询张量
        xk = keys.transpose(1, 2)  # 转置键张量
        xv = values.transpose(1, 2)  # 转置值张量

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)  # 计算缩放点积注意力
        output = output.transpose(1, 2).contiguous()  # 转置输出张量
        output = output.view(bsz, seqlen, -1)  # 重新调整输出张量的形状
        return self.wo(output)  # 返回最终输出


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """  # 前馈模块
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()  # 调用父类构造函数
        hidden_dim = int(2 * hidden_dim / 3)  # 计算隐藏层维度
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:  # 如果指定了FFN维度乘数
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)  # 应用乘数
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 确保隐藏层维度是指定值的倍数

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 创建第一层的线性变换
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 创建第二层的线性变换
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 创建第三层的线性变换

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # 前向传播函数

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)  # 初始化第一层权重
        for linear in (self.w2, self.w3):  # 初始化第二层和第三层权重
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """  # 变换块模块
    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()  # 调用父类构造函数
        self.n_heads = model_args.n_heads  # 保存注意力头的数量
        self.dim = model_args.dim  # 保存模型维度
        self.attention = Attention(model_args)  # 创建注意力模块
        self.feed_forward = FeedForward(  # 创建前馈模块
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id  # 保存层ID
        self.num_layers = model_args.n_layers  # 保存总层数

        self.attention_norm = RMSNorm(  # 创建注意力输出的归一化层
            dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = RMSNorm(  # 创建前馈输出的归一化层
            dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:  # 如果启用深度初始化
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5  # 计算权重初始化标准差
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5  # 计算权重初始化标准差

    def forward(  # 前向传播函数
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """  # 执行变换块的前向传播
        h = x + self.attention(self.attention_norm(x), freqs_cis)  # 应用注意力机制
        out = h + self.feed_forward(self.ffn_norm(h))  # 应用前馈层
        return out  # 返回输出

    def init_weights(self):  # 初始化权重
        for norm in (self.attention_norm, self.ffn_norm):  # 初始化归一化层
            norm.reset_parameters()  # 重置参数
        self.attention.init_weights(self.weight_init_std)  # 初始化注意力模块权重
        self.feed_forward.init_weights(self.weight_init_std)  # 初始化前馈模块权重


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """  # 变换模块
    def __init__(self, model_args: ModelArgs):
        super().__init__()  # 调用父类构造函数
        self.model_args = model_args  # 保存模型参数
        self.vocab_size = model_args.vocab_size  # 保存词汇表大小
        self.n_layers = model_args.n_layers  # 保存层数
        self.model_dim = model_args.dim  # 保存模型维度

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)  # 创建词嵌入层
        self.register_buffer(  # 注册缓冲区以保存频率张量
            "freqs_cis",
            precompute_freqs_cis(  # 预计算频率
                model_args.dim // model_args.n_heads,
                # Need to compute until at least the max token limit for generation
                # (use 2x max sequence length to be safe)
                model_args.max_seq_len * 2,
            ),
        )
        self.layers = torch.nn.ModuleList()  # 创建变换块列表
        for layer_id in range(model_args.n_layers):  # 遍历层数
            self.layers.append(TransformerBlock(layer_id, model_args))  # 添加变换块

        self.norm = RMSNorm(  # 创建模型输出的归一化层
            dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)  # 创建最终输出的线性层
        self.init_weights()  # 初始化权重

    def init_weights(self):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """  # 初始化权重的说明
        with torch.device(self.freqs_cis.device):  # 在频率张量的设备上执行
            self.freqs_cis = precompute_freqs_cis(  # 重新计算频率
                self.model_args.dim // self.model_args.n_heads,
                # Need to compute until at least the max token limit for generation
                # (use 2x max sequence length to be safe)
                self.model_args.max_seq_len * 2,
            )
        nn.init.normal_(self.tok_embeddings.weight)  # 初始化词嵌入权重
        for layer in self.layers:  # 初始化每个变换块的权重
            layer.init_weights()
        self.norm.reset_parameters()  # 重置归一化层参数
        final_out_std = self.model_args.dim**-0.5  # 计算最终输出标准差
        cutoff_factor = 3  # 截断因子
        nn.init.trunc_normal_(  # 初始化输出层权重
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """  # 执行变换模型的前向传播
        _bsz, seqlen = tokens.shape  # 获取输入张量的形状
        h = self.tok_embeddings(tokens)  # 获取词嵌入
        self.freqs_cis = self.freqs_cis.to(h.device)  # 将频率张量移动到输入张量的设备
        freqs_cis = self.freqs_cis[0:seqlen]  # 获取适当的频率张量

        for layer in self.layers:  # 遍历每个变换块
            h = layer(h, freqs_cis)  # 通过变换块进行前向传播
        h = self.norm(h)  # 应用归一化
        output = self.output(h).float()  # 获取最终输出
        return output  # 返回输出

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """  # 从ModelArgs对象初始化变换模型
        return cls(model_args)  # 返回变换模型实例