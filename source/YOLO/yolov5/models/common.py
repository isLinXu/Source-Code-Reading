# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Common modules."""

import ast  # 导入抽象语法树模块
import contextlib  # 导入上下文管理模块
import json  # 导入JSON模块
import math  # 导入数学模块
import platform  # 导入平台模块
import warnings  # 导入警告模块
import zipfile  # 导入压缩文件模块
from collections import OrderedDict, namedtuple  # 从collections导入有序字典和命名元组
from copy import copy  # 从copy模块导入复制函数
from pathlib import Path  # 从pathlib导入Path类
from urllib.parse import urlparse  # 从urllib.parse导入urlparse函数

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import pandas as pd  # 导入Pandas库
import requests  # 导入Requests库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from PIL import Image  # 导入PIL库中的Image模块
from torch.cuda import amp  # 导入PyTorch的自动混合精度模块

# Import 'ultralytics' package or install if missing
# 导入'ultralytics'包，如果缺失则安装
try:
    import ultralytics  # 尝试导入ultralytics包

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
    # 验证包不是目录
except (ImportError, AssertionError):
    import os  # 导入os模块

    os.system("pip install -U ultralytics")  # 执行命令安装ultralytics包
    import ultralytics  # 再次尝试导入ultralytics包

from ultralytics.utils.plotting import Annotator, colors, save_one_box  # 从ultralytics导入绘图工具

from utils import TryExcept  # 从utils导入TryExcept类
from utils.dataloaders import exif_transpose, letterbox  # 从utils.dataloaders导入exif_transpose和letterbox函数
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)  # 从utils.general导入多个实用函数
from utils.torch_utils import copy_attr, smart_inference_mode  # 从utils.torch_utils导入copy_attr和smart_inference_mode


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    # 将卷积核填充到“相同”的输出形状，调整可选的膨胀；返回填充大小

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        # 如果膨胀大于1，则计算实际的卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        # 如果没有提供填充，则自动计算填充
    return p  # 返回填充大小


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # 标准卷积，参数包括输入通道、输出通道、卷积核、步幅、填充、组、膨胀和激活函数
    default_act = nn.SiLU()  # default activation
    # 默认激活函数为SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        # 初始化一个标准卷积层，支持可选的批归一化和激活
        super().__init__()  # 调用父类初始化方法
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 创建卷积层
        self.bn = nn.BatchNorm2d(c2)  # 添加批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 设置激活函数，如果未提供则使用默认激活函数

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        # 对输入张量`x`应用卷积、批归一化和激活函数
        return self.act(self.bn(self.conv(x)))  # 返回处理后的结果

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        # 对输入张量`x`应用融合的卷积和激活函数
        return self.act(self.conv(x))  # 返回处理后的结果


class DWConv(Conv):
    # Depth-wise convolution
    # 深度卷积
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        # 初始化一个深度卷积层，支持可选激活；参数包括输入通道（c1）、输出通道（c2）、卷积核大小（k）、步幅（s）、膨胀（d）和激活标志（act）
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        # 调用父类构造函数，设置组数为输入通道和输出通道的最大公约数


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    # 深度转置卷积
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        # 初始化YOLOv5的深度转置卷积层；参数包括输入通道（c1）、输出通道（c2）、卷积核大小（k）、步幅（s）、输入填充（p1）、输出填充（p2）
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
        # 调用父类构造函数，设置组数为输入通道和输出通道的最大公约数


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    # Transformer层，参考文献：https://arxiv.org/abs/2010.11929（去除LayerNorm层以提高性能）
    def __init__(self, c, num_heads):
        """
        Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        See  as described in https://arxiv.org/abs/2010.11929.
        """
        # 初始化一个Transformer层，不使用LayerNorm以提高性能，包含多头注意力和线性层
        super().__init__()  # 调用父类初始化方法
        self.q = nn.Linear(c, c, bias=False)  # 查询线性变换
        self.k = nn.Linear(c, c, bias=False)  # 键线性变换
        self.v = nn.Linear(c, c, bias=False)  # 值线性变换
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)  # 多头注意力机制
        self.fc1 = nn.Linear(c, c, bias=False)  # 第一个线性层
        self.fc2 = nn.Linear(c, c, bias=False)  # 第二个线性层

    def forward(self, x):
        """Performs forward pass using MultiheadAttention and two linear transformations with residual connections."""
        # 使用多头注意力和两个线性变换执行前向传播，并带有残差连接
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x  # 计算注意力输出并加上输入x
        x = self.fc2(self.fc1(x)) + x  # 经过两个线性层后加上输入x
        return x  # 返回处理后的结果


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    # 视觉Transformer，参考文献：https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified
        layers.
        """
        # 初始化一个用于视觉任务的Transformer块，如果需要则调整维度并堆叠指定层
        super().__init__()  # 调用父类初始化方法
        self.conv = None  # 初始化卷积层为None
        if c1 != c2:
            self.conv = Conv(c1, c2)  # 如果输入通道与输出通道不同，则创建卷积层
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        # 创建可学习的位置嵌入线性层
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        # 创建多个Transformer层的序列
        self.c2 = c2  # 保存输出通道数

    def forward(self, x):
        """Processes input through an optional convolution, followed by Transformer layers and position embeddings for
        object detection.
        """
        # 处理输入，通过可选的卷积、Transformer层和位置嵌入进行对象检测
        if self.conv is not None:
            x = self.conv(x)  # 如果有卷积层，则先通过卷积层处理输入
        b, _, w, h = x.shape  # 获取输入的形状
        p = x.flatten(2).permute(2, 0, 1)  # 将输入展平并调整维度
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
        # 通过Transformer层处理，并返回调整后的结果


class Bottleneck(nn.Module):
    # Standard bottleneck
    # 标准瓶颈
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        # 初始化一个标准瓶颈层，支持可选的shortcut和组卷积，支持通道扩展
        super().__init__()  # 调用父类初始化方法
        c_ = int(c2 * e)  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # 第二个卷积层
        self.add = shortcut and c1 == c2  # 如果使用shortcut且输入通道与输出通道相同，则设置add为True

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # 通过两个卷积处理输入，如果通道维度匹配则可选地添加shortcut；输入为张量
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # 如果add为True，则返回输入加上卷积输出，否则只返回卷积输出


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # CSP瓶颈，参考文献：https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        # 初始化CSP瓶颈，支持可选的shortcut；参数包括输入通道、输出通道、重复次数、shortcut布尔值、组数和扩展比例
        super().__init__()  # 调用父类初始化方法
        c_ = int(c2 * e)  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # 第二个卷积层
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  # 第三个卷积层
        self.cv4 = Conv(2 * c_, c2, 1, 1)  # 第四个卷积层
        self.bn = nn.BatchNorm2d(2 * c_)  # 应用于拼接后的批归一化层
        self.act = nn.SiLU()  # 激活函数
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # 创建多个Bottleneck层的序列

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        # 执行前向传播，通过应用层、激活和拼接处理输入x，返回特征增强的输出
        y1 = self.cv3(self.m(self.cv1(x)))  # 通过第一个卷积和Bottleneck层处理输入
        y2 = self.cv2(x)  # 通过第二个卷积处理输入
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
        # 将y1和y2在通道维度上拼接，经过批归一化和激活后通过第四个卷积层返回结果


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    # 交叉卷积下采样
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        # 初始化CrossConv，支持下采样、扩展和可选的shortcut；输入通道为c1，输出通道为c2
        super().__init__()  # 调用父类初始化方法
        c_ = int(c2 * e)  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, (1, k), (1, s))  # 第一个卷积层，卷积核为(1, k)
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)  # 第二个卷积层，卷积核为(k, 1)
        self.add = shortcut and c1 == c2  # 如果使用shortcut且输入通道与输出通道相同，则设置add为True

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        # 执行特征采样、扩展，并在通道匹配时应用shortcut；输入为张量x
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # 如果add为True，则返回输入加上卷积输出，否则只返回卷积输出


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # CSP瓶颈，包含3个卷积
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        # 初始化C3模块，支持通道数、瓶颈重复次数、shortcut使用、组卷积和扩展选项
        super().__init__()  # 调用父类初始化方法
        c_ = int(c2 * e)  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = Conv(c1, c_, 1, 1)  # 第二个卷积层
        self.cv3 = Conv(2 * c_, c2, 1)  # 第三个卷积层，输出通道为c2
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # 创建多个Bottleneck层的序列

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        # 执行前向传播，使用来自两个卷积和一个Bottleneck序列的拼接输出
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        # 将两个卷积的输出在通道维度上拼接并通过第三个卷积层返回结果


class C3x(C3):
    # C3 module with cross-convolutions
    # C3模块，包含交叉卷积
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        # 初始化C3x模块，包含交叉卷积，扩展C3，支持自定义通道维度、组数和扩展
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类构造函数
        c_ = int(c2 * e)  # hidden channels
        # 计算隐藏通道数
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))
        # 创建多个CrossConv层的序列

class C3TR(C3):
    # C3 module with TransformerBlock()
    # C3模块，包含TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with TransformerBlock for enhanced feature extraction, accepts channel sizes, shortcut
        config, group, and expansion.
        """
        # 初始化C3模块，包含TransformerBlock以增强特征提取，接受通道大小、shortcut配置、组数和扩展比例
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类初始化方法
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.m = TransformerBlock(c_, c_, 4, n)  # 创建TransformerBlock，设置头数为4，层数为n


class C3SPP(C3):
    # C3 module with SPP()
    # C3模块，包含SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        # 初始化C3模块，包含SPP层以进行高级空间特征提取，接受通道大小、卷积核大小、shortcut、组数和扩展比例
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类初始化方法
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.m = SPP(c_, c_, k)  # 创建SPP层，使用指定的卷积核大小k


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    # C3模块，包含GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        # 初始化YOLOv5的C3模块，使用Ghost Bottlenecks进行高效特征提取
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类初始化方法
        c_ = int(c2 * e)  # hidden channels
        # 计算隐藏通道数
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))  # 创建多个GhostBottleneck层的序列


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    # 空间金字塔池化（SPP）层，参考文献：https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input channels), c2 (output channels), k (kernel sizes)."""
        # 初始化SPP层，使用空间金字塔池化，参考文献：https://arxiv.org/abs/1406.4729，参数包括输入通道c1、输出通道c2和卷积核大小k
        super().__init__()  # 调用父类初始化方法
        c_ = c1 // 2  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层，卷积核为1x1
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)  # 第二个卷积层，输入通道为隐藏通道数乘以(k的长度+1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        # 创建最大池化层列表，使用指定的卷积核大小k

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        # 将卷积和最大池化层应用于输入张量x，拼接结果并返回输出张量
        x = self.cv1(x)  # 先通过第一个卷积层处理输入
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            # 忽略torch 1.9.0的max_pool2d()警告
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
            # 将x和所有最大池化层的输出在通道维度拼接，然后通过第二个卷积层返回结果

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    # 空间金字塔池化 - 快速（SPPF）层，适用于YOLOv5，由Glenn Jocher提供
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        # 初始化YOLOv5的SPPF层，给定通道数和卷积核大小，结合卷积和最大池化
        super().__init__()  # 调用父类初始化方法
        c_ = c1 // 2  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层，卷积核为1x1
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 第二个卷积层，输入通道为隐藏通道数的4倍
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 创建最大池化层，卷积核大小为k

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        # 通过一系列卷积和最大池化操作处理输入以进行特征提取
        x = self.cv1(x)  # 先通过第一个卷积层处理输入
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            # 忽略torch 1.9.0的max_pool2d()警告
            y1 = self.m(x)  # 通过最大池化层处理x
            y2 = self.m(y1)  # 再次通过最大池化层处理y1
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
            # 将x、y1、y2和y2经过最大池化后的结果在通道维度上拼接，然后通过第二个卷积层返回结果


class Focus(nn.Module):
    # Focus wh information into c-space
    # 将宽高信息集中到通道空间
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate width-height info into channel space with configurable convolution
        parameters.
        """
        # 初始化Focus模块，将宽高信息集中到通道空间，支持可配置的卷积参数
        super().__init__()  # 调用父类初始化方法
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)  # 创建卷积层，输入通道为c1的4倍

    def forward(self, x):
        """Processes input through Focus mechanism, reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution."""
        # 通过Focus机制处理输入，将形状从(b,c,w,h)重塑为(b,4c,w/2,h/2)，然后应用卷积
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # 将输入x在宽高维度上进行下采样并拼接，然后通过卷积层返回结果


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    # Ghost卷积，参考文献：https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups, and activation; halves out channels
        for efficiency.
        """
        # 初始化GhostConv，设置输入输出通道、卷积核大小、步幅、组数和激活函数；为了提高效率，输出通道数减半
        super().__init__()  # 调用父类初始化方法
        c_ = c2 // 2  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)  # 第一个卷积层
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)  # 第二个卷积层，卷积核为5x5

    def forward(self, x):
        """Performs forward pass, concatenating outputs of two convolutions on input `x`: shape (B,C,H,W)."""
        # 执行前向传播，将两个卷积的输出拼接在输入x上，形状为(B,C,H,W)
        y = self.cv1(x)  # 通过第一个卷积层处理输入
        return torch.cat((y, self.cv2(y)), 1)  # 将y和第二个卷积的输出在通道维度上拼接


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    # Ghost瓶颈，参考文献：https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
        # 初始化GhostBottleneck，输入通道为c1，输出通道为c2，卷积核大小为k，步幅为s；参考文献：https://github.com/huawei-noah/ghostnet
        super().__init__()  # 调用父类初始化方法
        c_ = c2 // 2  # 计算隐藏通道数
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            # 第一个卷积层，逐点卷积
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            # 如果步幅为2，则使用深度卷积；否则使用恒等映射
            GhostConv(c_, c2, 1, 1, act=False),
            # 第二个卷积层，逐点卷积
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
            # 如果步幅为2，则使用深度卷积和逐点卷积的shortcut；否则使用恒等映射
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        # 通过卷积和shortcut层处理输入，返回它们的和
        return self.conv(x) + self.shortcut(x)  # 返回卷积输出和shortcut输出的和



class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    # 将宽高压缩到通道中，例如将x(1,64,80,80)转换为x(1,256,40,40)
    def __init__(self, gain=2):
        """Initializes a layer to contract spatial dimensions (width-height) into channels, e.g., input shape
        (1,64,80,80) to (1,256,40,40).
        """
        # 初始化一个层，将空间维度（宽高）压缩到通道中，例如输入形状从(1,64,80,80)转换为(1,256,40,40)
        super().__init__()  # 调用父类初始化方法
        self.gain = gain  # 设置压缩因子gain

    def forward(self, x):
        """Processes input tensor to expand channel dimensions by contracting spatial dimensions, yielding output shape
        [(b, c*s*s, h//s, w//s)](cci:2://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov5/models/common.py:255:0-270:73).
        """
        # 处理输入张量，通过压缩空间维度来扩展通道维度，输出形状为[(b, c*s*s, h//s, w//s)](cci:2://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov5/models/common.py:255:0-270:73)
        b, c, h, w = x.size()  # 获取输入张量的形状
        s = self.gain  # 获取压缩因子
        x = x.view(b, c, h // s, s, w // s, s)  # 将x重塑为x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # 调整维度为x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # 重塑为输出形状x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    # 将通道扩展到宽高，例如将x(1,64,80,80)转换为x(1,16,160,160)
    def __init__(self, gain=2):
        """
        Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional gain
        factor.

        Example: x(1,64,80,80) to x(1,16,160,160).
        """
        # 初始化Expand模块，通过重新分配通道来增加空间维度，支持可选的增益因子
        super().__init__()  # 调用父类初始化方法
        self.gain = gain  # 设置增益因子

    def forward(self, x):
        """Processes input tensor x to expand spatial dimensions by redistributing channels, requiring C / gain^2 ==
        0.
        """
        # 处理输入张量x，通过重新分配通道来扩展空间维度，要求C / gain^2 == 0
        b, c, h, w = x.size()  # 获取输入张量的形状
        s = self.gain  # 获取增益因子
        x = x.view(b, s, s, c // s**2, h, w)  # 将x重塑为x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # 调整维度为x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # 重塑为输出形状x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    # 在指定维度上连接一组张量
    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        # 初始化Concat模块，在指定维度上连接张量
        super().__init__()  # 调用父类初始化方法
        self.d = dimension  # 设置连接的维度

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        # 在指定维度上连接一组张量；`x`是一个张量列表，`dimension`是一个整数
        return torch.cat(x, self.d)  # 使用torch.cat函数在指定维度上连接张量


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    # YOLOv5 多后端类，用于在各种后端上进行 Python 推理
    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes DetectMultiBackend with support for various inference backends, including PyTorch and ONNX."""
        # 初始化 DetectMultiBackend，支持多种推理后端，包括 PyTorch 和 ONNX。
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
        # 从 models.experimental 导入 attempt_download 和 attempt_load，作用域限制以避免循环导入

        super().__init__()  # 调用父类的构造函数
        w = str(weights[0] if isinstance(weights, list) else weights)  # 如果 weights 是列表，则取第一个元素，否则直接使用 weights
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        # 根据模型权重的类型，解包出不同的后端支持标志
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        # 如果使用 fp16，且模型类型为 PyTorch、TorchScript、ONNX、TensorRT 或 Triton，则保持 fp16 为 True
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        # 检查是否使用 CoreML、SavedModel、GraphDef、TFLite 或 Edge TPU 格式，设置 nhwc 标志
        stride = 32  # default stride
        # 默认步幅为 32
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        # 检查 CUDA 是否可用且设备不是 CPU，设置 cuda 标志
        if not (pt or triton):
            w = attempt_download(w)  # download if not local
            # 如果不是 PyTorch 或 Triton 模型，则尝试下载模型权重

        if pt:  # PyTorch
            # 如果是 PyTorch 模型
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            # 加载模型权重，指定设备和是否融合
            stride = max(int(model.stride.max()), 32)  # model stride
            # 获取模型的步幅，确保不小于 32
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            # 获取类名，如果模型有 module 属性则从中获取
            model.half() if fp16 else model.float()  # 根据 fp16 标志设置模型为半精度或单精度
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            # 显式赋值模型，以便后续调用 to()、cpu()、cuda() 和 half() 方法
        elif jit:  # TorchScript
            # 如果是 TorchScript 模型
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            # 记录加载信息
            extra_files = {"config.txt": ""}  # model metadata
            # 定义额外文件，用于存储模型元数据
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            # 加载 TorchScript 模型
            model.half() if fp16 else model.float()  # 根据 fp16 标志设置模型为半精度或单精度
            if extra_files["config.txt"]:  # load metadata dict
                # 如果存在配置文件，则加载元数据字典
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
                # 从元数据中提取步幅和类名
        elif dnn:  # ONNX OpenCV DNN
            # 如果使用 ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            # 记录加载信息
            check_requirements("opencv-python>=4.5.4")
            # 检查 OpenCV 版本要求
            net = cv2.dnn.readNetFromONNX(w)
            # 从 ONNX 文件加载网络


        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")  # 记录日志，表示正在为 ONNX Runtime 推理加载模型
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))  # 检查所需的库是否已安装
            import onnxruntime  # 导入 ONNX Runtime 库

            # 根据是否使用 CUDA 选择执行提供者
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]  # 如果使用 CUDA，则使用 CUDA 执行提供者，否则使用 CPU 执行提供者
            session = onnxruntime.InferenceSession(w, providers=providers)  # 创建 ONNX 推理会话
            output_names = [x.name for x in session.get_outputs()]  # 获取输出名称
            meta = session.get_modelmeta().custom_metadata_map  # metadata  # 获取模型的自定义元数据
            if "stride" in meta:  # 如果元数据中包含 "stride"
                stride, names = int(meta["stride"]), eval(meta["names"])  # 从元数据中提取步幅和名称

        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")  # 记录日志，表示正在为 OpenVINO 推理加载模型
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/  # 检查 OpenVINO 的版本要求
            from openvino.runtime import Core, Layout, get_batch  # 从 OpenVINO 库导入必要的类和函数

            core = Core()  # 创建 OpenVINO 核心对象
            if not Path(w).is_file():  # if not *.xml  # 如果指定路径不是文件
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir  # 从 *_openvino_model 目录中获取 *.xml 文件
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))  # 读取 OpenVINO 模型及其权重
            if ov_model.get_parameters()[0].get_layout().empty:  # 如果模型参数的布局为空
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))  # 设置模型参数的布局为 NCHW
            batch_dim = get_batch(ov_model)  # 获取模型的批次维度
            if batch_dim.is_static:  # 如果批次维度是静态的
                batch_size = batch_dim.get_length()  # 获取批次大小
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device  # 编译模型，自动选择最佳可用设备
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata  # 从 YAML 文件中加载元数据
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")  # 记录日志，表示正在为 TensorRT 推理加载模型
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download  # 导入 TensorRT 库

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0  # 检查 TensorRT 版本，要求版本大于等于 7.0.0
            if device.type == "cpu":  # 如果设备类型为 CPU
                device = torch.device("cuda:0")  # 则将设备设置为 CUDA 设备

            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))  # 定义一个命名元组，用于存储绑定信息
            logger = trt.Logger(trt.Logger.INFO)  # 创建 TensorRT 日志记录器
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:  # 打开模型文件并创建 TensorRT 运行时
                model = runtime.deserialize_cuda_engine(f.read())  # 反序列化 CUDA 引擎
            context = model.create_execution_context()  # 创建执行上下文
            bindings = OrderedDict()  # 创建有序字典以存储绑定信息
            output_names = []  # 初始化输出名称列表
            fp16 = False  # default updated below  # 默认 FP16 设置为 False，后续可能会更新
            dynamic = False  # 默认动态设置为 False

            for i in range(model.num_bindings):  # 遍历模型的所有绑定
                name = model.get_binding_name(i)  # 获取绑定名称
                dtype = trt.nptype(model.get_binding_dtype(i))  # 获取绑定数据类型
                if model.binding_is_input(i):  # 如果当前绑定是输入
                    if -1 in tuple(model.get_binding_shape(i)):  # 如果绑定形状中包含 -1，表示动态形状
                        dynamic = True  # 设置动态为 True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))  # 设置绑定形状
                    if dtype == np.float16:  # 如果数据类型为 float16
                        fp16 = True  # 设置 FP16 为 True
                else:  # 如果当前绑定是输出
                    output_names.append(name)  # 将输出名称添加到输出名称列表中
                shape = tuple(context.get_binding_shape(i))  # 获取绑定的形状
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # 创建一个空的张量并将其移动到指定设备
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 将绑定信息存储到字典中

            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 创建绑定地址的有序字典
            batch_size = bindings["images"].shape[0]  # 获取批次大小，如果是动态的，则此处为最大批次大小

        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")  # 记录日志，表示正在为 CoreML 推理加载模型
            import coremltools as ct  # 导入 CoreML 工具库

            model = ct.models.MLModel(w)  # 加载 CoreML 模型

        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")  # 记录日志，表示正在为 TensorFlow SavedModel 推理加载模型
            import tensorflow as tf  # 导入 TensorFlow 库

            keras = False  # assume TF1 saved_model  # 假设是 TF1 的 SavedModel
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)  # 根据是否为 Keras 模型加载模型

        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")  # 记录日志，表示正在为 TensorFlow GraphDef 推理加载模型
            import tensorflow as tf  # 导入 TensorFlow 库

            def wrap_frozen_graph(gd, inputs, outputs):  # 定义函数，用于包装 TensorFlow GraphDef 以进行推理，返回修剪后的函数
                """Wraps a TensorFlow GraphDef for inference, returning a pruned function."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped  # 包装图
                ge = x.graph.as_graph_element  # 获取图的元素
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))  # 修剪图

            def gd_outputs(gd):  # 定义函数，生成排序后的图输出列表，排除 NoOp 节点和输入，格式为 '<name>:0'
                """Generates a sorted list of graph outputs excluding NoOp nodes and inputs, formatted as '<name>:0'."""
                name_list, input_list = [], []  # 初始化名称列表和输入列表
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)  # 添加节点名称
                    input_list.extend(node.input)  # 扩展输入列表
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))  # 返回排序后的输出列表

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:  # 打开模型文件
                gd.ParseFromString(f.read())  # 解析 GraphDef
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))  # 包装冻结的图

        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # 尝试导入 TFLite 运行时
                from tflite_runtime.interpreter import Interpreter, load_delegate  # 导入 TFLite 解释器和加载委托
            except ImportError:  # 如果导入失败
                import tensorflow as tf  # 导入 TensorFlow 库

                Interpreter, load_delegate = (  # 从 TensorFlow 中获取解释器和加载委托
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")  # 记录日志，表示正在为 TensorFlow Lite Edge TPU 推理加载模型
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]  # 根据操作系统选择 Edge TPU 委托库
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])  # 创建 TFLite 解释器并加载委托
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")  # 记录日志，表示正在为 TensorFlow Lite 推理加载模型
                interpreter = Interpreter(model_path=w)  # 加载 TFLite 模型
            interpreter.allocate_tensors()  # 分配张量
            input_details = interpreter.get_input_details()  # 获取输入细节
            output_details = interpreter.get_output_details()  # 获取输出细节
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):  # 使用上下文管理器抑制 BadZipFile 异常
                with zipfile.ZipFile(w, "r") as model:  # 打开模型文件
                    meta_file = model.namelist()[0]  # 获取第一个文件名
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))  # 读取元数据并解析
                    stride, names = int(meta["stride"]), meta["names"]  # 从元数据中提取步幅和名称

        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")  # 抛出未实现错误，表示不支持 TF.js 推理

        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")  # 记录日志，表示正在为 PaddlePaddle 推理加载模型
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")  # 检查所需的 PaddlePaddle 库
            import paddle.inference as pdi  # 导入 PaddlePaddle 推理库

            if not Path(w).is_file():  # if not *.pdmodel  # 如果指定路径不是文件
                w = next(Path(w).rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir  # 从 *_paddle_model 目录中获取 *.pdmodel 文件
            weights = Path(w).with_suffix(".pdiparams")  # 获取权重文件路径
            config = pdi.Config(str(w), str(weights))  # 创建 PaddlePaddle 配置
            if cuda:  # 如果使用 CUDA
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)  # 启用 GPU，初始化内存池
            predictor = pdi.create_predictor(config)  # 创建预测器
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])  # 获取输入句柄
            output_names = predictor.get_output_names()  # 获取输出名称

        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")  # 记录日志，表示正在使用 Triton 推理服务器
            check_requirements("tritonclient[all]")  # 检查 Triton 客户端的要求
            from utils.triton import TritonRemoteModel  # 从 utils.triton 导入 TritonRemoteModel

            model = TritonRemoteModel(url=w)  # 创建 Triton 远程模型
            nhwc = model.runtime.startswith("tensorflow")  # 检查模型运行时是否以 TensorFlow 开头

        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")  # 抛出未实现错误，表示不支持该格式

        # class names
        if "names" not in locals():  # 如果本地没有定义 names
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}  # 从数据中加载类名，或生成默认类名

        if names[0] == "n01440764" and len(names) == 1000:  # 如果类名为 ImageNet 的特定类名并且数量为 1000
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # 从 ImageNet YAML 文件中加载人类可读的名称

        self.__dict__.update(locals())  # 将所有局部变量赋值给 self


    def forward(self, im, augment=False, visualize=False):
        """Performs YOLOv5 inference on input images with options for augmentation and visualization."""
        # 对输入图像执行 YOLOv5 推理，支持增强和可视化选项
        b, ch, h, w = im.shape  # batch, channel, height, width  # 获取输入图像的形状，分别为批次、通道、高度和宽度
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16  # 如果使用 FP16 并且输入图像不是 FP16 类型，则将其转换为 FP16

        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)  # 如果使用 NHWC 格式，则调整图像维度顺序

        if self.pt:  # PyTorch
            # 如果使用 PyTorch 模型，根据是否需要增强或可视化选择调用模型
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)  # 如果使用 TorchScript 模型，直接调用模型
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy  # 将图像从 PyTorch 转换为 NumPy 数组
            self.net.setInput(im)  # 设置输入到 DNN 网络
            y = self.net.forward()  # 执行前向推理
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy  # 将图像从 PyTorch 转换为 NumPy 数组
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})  # 使用 ONNX Runtime 进行推理
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32  # 将图像转换为 NumPy 数组，保持 FP32 格式
            y = list(self.ov_compiled_model(im).values())  # 执行 OpenVINO 推理并获取输出
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                # 如果模型是动态的并且输入形状与绑定形状不匹配
                i = self.model.get_binding_index("images")  # 获取绑定索引
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic  # 如果是动态，则设置绑定形状
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)  # 更新绑定形状
                for name in self.output_names:
                    i = self.model.get_binding_index(name)  # 获取输出绑定索引
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))  # 调整输出绑定的大小
            s = self.bindings["images"].shape  # 获取图像绑定的形状
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"  # 确保输入大小与模型最大大小匹配
            self.binding_addrs["images"] = int(im.data_ptr())  # 更新输入绑定的地址
            self.context.execute_v2(list(self.binding_addrs.values()))  # 执行推理
            y = [self.bindings[x].data for x in sorted(self.output_names)]  # 获取输出数据
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()  # 将图像从 PyTorch 转换为 NumPy 数组
            im = Image.fromarray((im[0] * 255).astype("uint8"))  # 将 NumPy 数组转换为图像格式
            # im = im.resize((192, 320), Image.BILINEAR)  # 可以选择调整图像大小
            y = self.model.predict({"image": im})  # coordinates are xywh normalized  # 使用 CoreML 模型进行推理，坐标为 xywh 归一化
            if "confidence" in y:  # 如果输出中包含置信度信息
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels  # 将归一化坐标转换为像素坐标
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)  # 获取最大置信度和对应的类别
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)  # 合并输出
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)  # 对于分割模型，反转输出
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)  # 将图像转换为 NumPy 数组并转换为 FP32 类型
            self.input_handle.copy_from_cpu(im)  # 将输入数据复制到 PaddlePaddle 输入句柄
            self.predictor.run()  # 执行推理
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]  # 获取输出数据
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)  # 使用 Triton 进行推理
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()  # 将图像从 PyTorch 转换为 NumPy 数组
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)  # 根据是否为 Keras 模型加载模型
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))  # 使用冻结的函数进行推理
            else:  # Lite or Edge TPU
                input = self.input_details[0]  # 获取输入细节
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model  # 检查是否为量化的 uint8 模型
                if int8:
                    scale, zero_point = input["quantization"]  # 获取量化参数
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale  # 反量化
                self.interpreter.set_tensor(input["index"], im)  # 设置输入张量
                self.interpreter.invoke()  # 执行推理
                y = []  # 初始化输出列表
                for output in self.output_details:  # 遍历输出细节
                    x = self.interpreter.get_tensor(output["index"])  # 获取输出张量
                    if int8:
                        scale, zero_point = output["quantization"]  # 获取输出的量化参数
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale  # 反量化
                    y.append(x)  # 将输出添加到列表中
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]  # 确保输出为 NumPy 数组
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels  # 将归一化坐标转换为像素坐标

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]  # 如果输出为列表或元组，返回转换后的结果
        else:
            return self.from_numpy(y)  # 否则直接返回转换后的结果

    def from_numpy(self, x):
        """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
        # 将 NumPy 数组转换为 PyTorch 张量，保持设备兼容性
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x  # 如果输入是 NumPy 数组，则转换为张量并移动到指定设备

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Performs a single inference warmup to initialize model weights, accepting an `imgsz` tuple for image size."""
        # 执行一次推理预热以初始化模型权重，接受一个 `imgsz` 元组作为图像大小
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton  # 初始化预热类型
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):  # 如果有任何预热类型并且设备不是 CPU 或者是 Triton
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input  # 创建一个空的输入张量
            for _ in range(2 if self.jit else 1):  # 如果使用 JIT，则预热两次，否则一次
                self.forward(im)  # warmup  # 执行前向推理进行预热

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines model type from file path or URL, supporting various export formats.

        Example: path='path/to/model.onnx' -> type=onnx
        """
        # 从文件路径或 URL 确定模型类型，支持各种导出格式
        from export import export_formats  # 从 export 导入导出格式
        from utils.downloads import is_url  # 从 utils.downloads 导入 is_url 函数

        sf = list(export_formats().Suffix)  # export suffixes  # 获取导出后缀列表
        if not is_url(p, check=False):  # 如果不是 URL
            check_suffix(p, sf)  # checks  # 检查后缀
        url = urlparse(p)  # if url may be Triton inference server  # 解析 URL
        types = [s in Path(p).name for s in sf]  # 检查文件名中是否包含导出后缀
        types[8] &= not types[9]  # tflite &= not edgetpu  # 确保 TFLite 不是 Edge TPU
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])  # 检查是否为 Triton 服务器
        return types + [triton]  # 返回模型类型和 Triton 标志

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """Loads metadata from a YAML file, returning strides and names if the file exists, otherwise `None`."""
        # 从 YAML 文件加载元数据，如果文件存在则返回步幅和名称，否则返回 None
        if f.exists():  # 如果文件存在
            d = yaml_load(f)  # 加载 YAML 文件
            return d["stride"], d["names"]  # assign stride, names  # 返回步幅和名称
        return None, None  # 如果文件不存在，返回 None



class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    # YOLOv5 输入稳健模型包装器，用于传递 cv2/np/PIL/torch 输入，包括预处理、推理和非极大值抑制（NMS）
    conf = 0.25  # NMS confidence threshold  # NMS 置信度阈值
    iou = 0.45  # NMS IoU threshold  # NMS IoU 阈值
    agnostic = False  # NMS class-agnostic  # NMS 类别无关
    multi_label = False  # NMS multiple labels per box  # NMS 每个框多个标签
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs  # （可选列表）按类过滤，例如 COCO 中的人员、猫和狗
    max_det = 1000  # maximum number of detections per image  # 每张图像的最大检测数量
    amp = False  # Automatic Mixed Precision (AMP) inference  # 自动混合精度（AMP）推理

    def __init__(self, model, verbose=True):
        """Initializes YOLOv5 model for inference, setting up attributes and preparing model for evaluation."""
        # 初始化 YOLOv5 模型以进行推理，设置属性并准备模型进行评估
        super().__init__()  # 调用父类构造函数
        if verbose:
            LOGGER.info("Adding AutoShape... ")  # 记录日志，表示正在添加 AutoShape
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # 复制属性
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() 实例
        self.pt = not self.dmb or model.pt  # PyTorch 模型
        self.model = model.eval()  # 将模型设置为评估模式
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference  # 为安全的多线程推理设置为 False
            m.export = True  # do not output loss values  # 不输出损失值

    def _apply(self, fn):
        """
        Applies to(), cpu(), cuda(), half() etc.

        to model tensors excluding parameters or registered buffers.
        """
        # 应用 to()、cpu()、cuda()、half() 等方法
        self = super()._apply(fn)  # 调用父类的 _apply 方法
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)  # 应用函数到 stride
            m.grid = list(map(fn, m.grid))  # 应用函数到 grid
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))  # 应用函数到 anchor_grid
        return self  # 返回修改后的对象

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        """
        Performs inference on inputs with optional augment & profiling.

        Supports various formats including file, URI, OpenCV, PIL, numpy, torch.
        """
        # 对输入执行推理，支持可选的增强和性能分析
        # 支持多种格式，包括文件、URI、OpenCV、PIL、numpy、torch
        # For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())  # 初始化性能分析器
        with dt[0]:  # 开始第一个性能分析
            if isinstance(size, int):  # expand  # 如果 size 是整数
                size = (size, size)  # 扩展为元组
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param  # 获取模型参数
            autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference  # 自动混合精度推理
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):  # 如果输入是 PyTorch 张量
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference  # 执行推理

            # Pre-process  # 预处理
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images  # 获取图像数量和列表
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames  # 初始化图像和推理形状、文件名
            for i, im in enumerate(ims):  # 遍历每个输入图像
                f = f"image{i}"  # filename  # 生成文件名
                if isinstance(im, (str, Path)):  # filename or uri  # 如果输入是文件名或 URI
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im  # 打开图像
                    im = np.asarray(exif_transpose(im))  # 转换为 NumPy 数组并处理 EXIF 信息
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f  # 转换为 NumPy 数组
                files.append(Path(f).with_suffix(".jpg").name)  # 将文件名添加到列表中
                if im.shape[0] < 5:  # image in CHW  # 如果图像维度小于 5
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)  # 反转数据加载器的维度
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input  # 确保输入为 3 通道
                s = im.shape[:2]  # HWC  # 获取图像的高度和宽度
                shape0.append(s)  # image shape  # 添加图像形状
                g = max(size) / max(s)  # gain  # 计算增益
                shape1.append([int(y * g) for y in s])  # 更新推理形状
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update  # 更新图像
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape  # 计算推理形状
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad  # 填充图像
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW  # 堆叠并调整维度
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32  # 转换为 PyTorch 张量并归一化

        with amp.autocast(autocast):  # 自动混合精度推理
            # Inference  # 推理
            with dt[1]:  # 开始第二个性能分析
                y = self.model(x, augment=augment)  # forward  # 执行前向推理

            # Post-process  # 后处理
            with dt[2]:  # 开始第三个性能分析
                y = non_max_suppression(  # NMS
                    y if self.dmb else y[0],  # 如果是多后端模型，使用 y，否则使用 y[0]
                    self.conf,  # 置信度阈值
                    self.iou,  # IoU 阈值
                    self.classes,  # 类别过滤
                    self.agnostic,  # 类别无关
                    self.multi_label,  # 多标签
                    max_det=self.max_det,  # 最大检测数量
                )
                for i in range(n):  # 遍历每个图像
                    scale_boxes(shape1, y[i][:, :4], shape0[i])  # 缩放边界框

            return Detections(ims, y, files, dt, self.names, x.shape)  # 返回检测结果


class Detections:
    # YOLOv5 detections class for inference results  # YOLOv5 检测类，用于推理结果

    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        """Initializes the YOLOv5 Detections class with image info, predictions, filenames, timing and normalization."""
        # 初始化 YOLOv5 Detections 类，包含图像信息、预测、文件名、时间和归一化
        super().__init__()  # 调用父类构造函数
        d = pred[0].device  # device  # 获取预测结果的设备
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations  # 计算归一化因子
        self.ims = ims  # list of images as numpy arrays  # 图像列表，作为 NumPy 数组
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)  # 预测结果列表，包含 (xyxy, 置信度, 类别)
        self.names = names  # class names  # 类别名称
        self.files = files  # image filenames  # 图像文件名
        self.times = times  # profiling times  # 性能分析时间
        self.xyxy = pred  # xyxy pixels  # xyxy 像素坐标
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels  # xywh 像素坐标
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized  # xyxy 归一化
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized  # xywh 归一化
        self.n = len(self.pred)  # number of images (batch size)  # 图像数量（批次大小）
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)  # 时间戳（毫秒）
        self.s = tuple(shape)  # inference BCHW shape  # 推理时的 BCHW 形状

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        """Executes model predictions, displaying and/or saving outputs with optional crops and labels."""
        # 执行模型预测，显示和/或保存输出，支持可选的裁剪和标签
        s, crops = "", []  # 初始化字符串和裁剪列表
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):  # 遍历每张图像和对应的预测结果
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string  # 添加图像信息到字符串
            if pred.shape[0]:  # 如果存在预测结果
                for c in pred[:, -1].unique():  # 遍历每个唯一的类别
                    n = (pred[:, -1] == c).sum()  # detections per class  # 计算每个类别的检测数量
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string  # 将检测数量和类别名称添加到字符串
                s = s.rstrip(", ")  # 去掉字符串末尾的逗号
                if show or save or render or crop:  # 如果需要显示、保存、渲染或裁剪
                    annotator = Annotator(im, example=str(self.names))  # 创建注释器
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class  # 遍历预测结果
                        label = f"{self.names[int(cls)]} {conf:.2f}"  # 创建标签
                        if crop:  # 如果需要裁剪
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None  # 设置裁剪文件路径
                            crops.append(  # 添加裁剪信息到列表
                                {
                                    "box": box,  # 边界框
                                    "conf": conf,  # 置信度
                                    "cls": cls,  # 类别
                                    "label": label,  # 标签
                                    "im": save_one_box(box, im, file=file, save=save),  # 保存裁剪图像
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))  # 添加边界框和标签
                    im = annotator.im  # 更新图像
            else:
                s += "(no detections)"  # 如果没有检测结果，添加提示信息

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np  # 将 NumPy 数组转换为图像格式
            if show:  # 如果需要显示图像
                if is_jupyter():  # 如果在 Jupyter 环境中
                    from IPython.display import display  # 从 IPython 导入显示函数

                    display(im)  # 显示图像
                else:
                    im.show(self.files[i])  # 在其他环境中显示图像
            if save:  # 如果需要保存图像
                f = self.files[i]  # 获取文件名
                im.save(save_dir / f)  # 保存图像
                if i == self.n - 1:  # 如果是最后一张图像
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")  # 记录保存信息
            if render:  # 如果需要渲染
                self.ims[i] = np.asarray(im)  # 更新图像列表
        if pprint:  # 如果需要打印信息
            s = s.lstrip("\n")  # 去掉字符串开头的换行符
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t  # 返回处理速度信息
        if crop:  # 如果需要裁剪
            if save:  # 如果需要保存裁剪结果
                LOGGER.info(f"Saved results to {save_dir}\n")  # 记录保存信息
            return crops  # 返回裁剪信息

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays detection results with optional labels.

        Usage: show(labels=True)
        """
        self._run(show=True, labels=labels)  # show results  # 执行模型预测并显示结果

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves detection results with optional labels to a specified directory.

        Usage: save(labels=True, save_dir='runs/detect/exp', exist_ok=False)
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir  # 增加保存目录路径
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results  # 执行模型预测并保存结果

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detection results, optionally saves them to a directory.

        Args: save (bool), save_dir (str), exist_ok (bool).
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None  # 如果需要保存，则增加保存目录路径
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results  # 执行模型预测并裁剪结果

    def render(self, labels=True):
        """Renders detection results with optional labels on images; args: labels (bool) indicating label inclusion."""
        self._run(render=True, labels=labels)  # render results  # 执行模型预测并渲染结果
        return self.ims  # 返回处理后的图像列表

    def pandas(self):
        """
        Returns detections as pandas DataFrames for various box formats (xyxy, xyxyn, xywh, xywhn).

        Example: print(results.pandas().xyxy[0]).
        """
        new = copy(self)  # return copy  # 创建当前对象的副本
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns  # 定义 xyxy 格式的列名
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns  # 定义 xywh 格式的列名
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):  # 遍历不同的格式
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update  # 更新检测结果
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])  # 将更新后的结果转换为 DataFrame
        return new  # 返回新的 Detections 对象

    def tolist(self):
        """
        Converts a Detections object into a list of individual detection results for iteration.

        Example: for result in results.tolist():
        """
        r = range(self.n)  # iterable  # 创建可迭代的范围
        return [
            Detections(
                [self.ims[i]],  # 图像列表
                [self.pred[i]],  # 预测结果列表
                [self.files[i]],  # 文件名列表
                self.times,  # 性能分析时间
                self.names,  # 类别名称
                self.s,  # 推理时的形状
            )
            for i in r  # 遍历每个索引
        ]

def print(self):
    """Logs the string representation of the current object's state via the LOGGER."""
    # 通过 LOGGER 记录当前对象状态的字符串表示
    LOGGER.info(self.__str__())  # 记录对象的字符串表示

def __len__(self):
    """Returns the number of results stored, overrides the default len(results)."""
    # 返回存储的结果数量，重写默认的 len(results)
    return self.n  # 返回结果的数量

def __str__(self):
    """Returns a string representation of the model's results, suitable for printing, overrides default
    print(results).
    """
    # 返回模型结果的字符串表示，适合打印，重写默认的 print(results)
    return self._run(pprint=True)  # 执行模型预测并打印结果

def __repr__(self):
    """Returns a string representation of the YOLOv5 object, including its class and formatted results."""
    # 返回 YOLOv5 对象的字符串表示，包括其类和格式化结果
    return f"YOLOv5 {self.__class__} instance\n" + self.__str__()  # 返回类名和结果的字符串表示



class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models  # YOLOv5 掩码原型模块，用于分割模型

    def __init__(self, c1, c_=256, c2=32):
        """Initializes YOLOv5 Proto module for segmentation with input, proto, and mask channels configuration."""
        # 初始化 YOLOv5 Proto 模块，用于分割，配置输入、原型和掩码通道
        super().__init__()  # 调用父类构造函数
        self.cv1 = Conv(c1, c_, k=3)  # 第一个卷积层，输入通道 c1，输出通道 c_，卷积核大小为 3
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # 上采样层，缩放因子为 2，使用最近邻插值
        self.cv2 = Conv(c_, c_, k=3)  # 第二个卷积层，输入和输出通道均为 c_
        self.cv3 = Conv(c_, c2)  # 第三个卷积层，输入通道 c_，输出通道 c2

    def forward(self, x):
        """Performs a forward pass using convolutional layers and upsampling on input tensor `x`."""
        # 对输入张量 x 执行前向传播，使用卷积层和上采样
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))  # 依次通过 cv1、上采样、cv2 和 cv3 层

class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)  # YOLOv5 分类头，将输入 x(b,c1,20,20) 转换为 x(b,c2)

    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        # 初始化 YOLOv5 分类头，配置输入通道、输出通道、卷积核、步幅、填充、分组和 dropout 概率
        super().__init__()  # 调用父类构造函数
        c_ = 1280  # efficientnet_b0 size  # 设置中间通道大小为 1280（efficientnet_b0 的大小）
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)  # 卷积层，配置输入通道、输出通道、卷积核大小、步幅、填充和分组
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)  # 自适应平均池化层，输出大小为 (1, 1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)  # Dropout 层，设置丢弃概率
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)  # 全连接层，将输入通道 c_ 转换为输出通道 c2

    def forward(self, x):
        """Processes input through conv, pool, drop, and linear layers; supports list concatenation input."""
        # 通过卷积、池化、Dropout 和线性层处理输入，支持列表拼接输入
        if isinstance(x, list):  # 如果输入是列表
            x = torch.cat(x, 1)  # 将列表中的张量在通道维度上拼接
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))  # 执行前向传播，返回最终结果