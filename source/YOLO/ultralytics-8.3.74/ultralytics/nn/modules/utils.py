# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Module utils."""  # 工具模块

import copy  # 导入copy模块
import math  # 导入math模块

import numpy as np  # 导入numpy库并命名为np
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
from torch.nn.init import uniform_  # 从PyTorch的nn.init模块导入uniform_函数

__all__ = "multi_scale_deformable_attn_pytorch", "inverse_sigmoid"  # 定义模块的公共接口

def _get_clones(module, n):
    """Create a list of cloned modules from the given module."""  # 根据给定模块创建一个克隆模块列表
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])  # 返回克隆的模块列表

def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""  # 根据给定的概率值初始化卷积/全连接层的偏置值
    return float(-np.log((1 - prior_prob) / prior_prob))  # 返回偏置初始化值

def linear_init(module):
    """Initialize the weights and biases of a linear module."""  # 初始化线性模块的权重和偏置
    bound = 1 / math.sqrt(module.weight.shape[0])  # 计算权重的边界
    uniform_(module.weight, -bound, bound)  # 使用均匀分布初始化权重
    if hasattr(module, "bias") and module.bias is not None:  # 检查模块是否有偏置
        uniform_(module.bias, -bound, bound)  # 使用均匀分布初始化偏置

def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor."""  # 计算张量的反sigmoid函数
    x = x.clamp(min=0, max=1)  # 将x限制在[0, 1]范围内
    x1 = x.clamp(min=eps)  # 将x1限制在[eps, 1]范围内
    x2 = (1 - x).clamp(min=eps)  # 将x2限制在[eps, 1]范围内
    return torch.log(x1 / x2)  # 返回反sigmoid值

def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,  # 输入值张量
    value_spatial_shapes: torch.Tensor,  # 输入值的空间形状
    sampling_locations: torch.Tensor,  # 采样位置
    attention_weights: torch.Tensor,  # 注意力权重
) -> torch.Tensor:
    """
    Multiscale deformable attention.  # 多尺度可变形注意力

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py  # 相关链接
    """
    bs, _, num_heads, embed_dims = value.shape  # 获取批次大小、通道数、头数和嵌入维度
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape  # 获取查询数量、头数、层数和点数
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)  # 将值张量按空间形状分割
    sampling_grids = 2 * sampling_locations - 1  # 计算采样网格
    sampling_value_list = []  # 初始化采样值列表
    for level, (H_, W_) in enumerate(value_spatial_shapes):  # 遍历每个空间层
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)  # 处理值张量
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)  # 处理采样网格
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # 使用双线性插值进行网格采样
        sampling_value_list.append(sampling_value_l_)  # 将采样值添加到列表中
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )  # 处理注意力权重
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)  # 计算输出
        .sum(-1)  # 对最后一个维度求和
        .view(bs, num_heads * embed_dims, num_queries)  # 调整输出形状
    )
    return output.transpose(1, 2).contiguous()  # 返回输出并调整维度