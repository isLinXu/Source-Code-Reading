import torch
import torch.nn as nn
from . import register_connector
from .base import Connector
import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum


# 根据是否返回字典格式来组织输出
class PerceiverResampler(nn.Module):
    """
    PerceiverResampler类是一个基于Transformer结构的重采样模块。

    Args:
        config (object): 包含模型配置参数的对象。
    """
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size                                                # 获取隐藏层大小
        depth=config.num_resampler_layers                                       # 获取重采样层数
        num_latents=config.num_queries                                          # 获取潜在向量的数量
        self.latents = nn.Parameter(torch.randn(num_latents, dim))              # 初始化潜在向量参数
        self.layers = nn.ModuleList([])                                         # 初始化层列表
        self.linear = nn.Linear(config.vision_hidden_size, config.hidden_size)  # 定义线性变换层

        # 构建重采样层列表
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=64, heads=8),      # 添加Perceiver注意力层
                        FeedForward(dim=dim, mult=4),                           # 添加前馈神经网络层
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)                                           # 定义层归一化层

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 重采样后的张量。
        """
        b, v = x.shape[:2]                                          # 获取批量大小和视觉特征维度
        x = self.linear(x)                                          # 应用线性变换
        # blocks                                                    # 重复潜在向量以匹配输入张量的批次大小
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=1)  # 在第1维上增加一个维度以匹配潜在向量的形状
        x = x.unsqueeze(1)
        # 应用重采样层
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents                    # 应用注意力层并加上残差连接
            latents = ff(latents) + latents                         # 应用前馈神经网络层并加上残差连接
        return self.norm(latents).squeeze(1)                        # 应用层归一化并移除多余的维度


# 定义ResamplerConnector类，继承自Connector
@register_connector('resampler')    
class ResamplerConnector(Connector):
    """
    ResamplerConnector类是一个连接器，用于将PerceiverResampler模块集成到更大的模型中。

    Args:
        config (object): 包含模型配置参数的对象。
    """
    def __init__(self, config):
        super().__init__()

        self._connector = PerceiverResampler(config)                # 初始化PerceiverResampler模块

   
# =================================resampler related =================================
def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        """
        初始化PerceiverAttention模块。

        Args:
            dim (int): 输入和输出的维度。
            dim_head (int): 每个头的维度，默认为64。
            heads (int): 头的数量，默认为8。
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)

        前向传播函数。
        Args:
            x (torch.Tensor): 图像特征，形状为(b, T, n1, D)。
            latents (torch.Tensor): 潜在特征，形状为(b, T, n2, D)。

        Returns:
            torch.Tensor: 经过注意力机制处理后的输出特征。
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        # 注意力机制
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)
    
