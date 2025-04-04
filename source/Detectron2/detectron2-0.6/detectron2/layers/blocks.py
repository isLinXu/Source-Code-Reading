# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import fvcore.nn.weight_init as weight_init
from torch import nn

from .batch_norm import FrozenBatchNorm2d, get_norm
from .wrappers import Conv2d


"""
CNN building blocks.
CNN构建块。
"""


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    CNN块假定具有输入通道、输出通道和步幅。
    The input and output of `forward()` method must be NCHW tensors.
    `forward()`方法的输入和输出必须是NCHW格式的张量。
    The method can perform arbitrary computation but must match the given
    channels and stride specification.
    该方法可以执行任意计算，但必须匹配给定的通道和步幅规范。

    Attribute:
        in_channels (int):
        # 输入通道数（整数）
        out_channels (int):
        # 输出通道数（整数）
        stride (int):
        # 步幅（整数）
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        任何子类的`__init__`方法也应包含这些参数。

        Args:
            in_channels (int):
            # 输入通道数
            out_channels (int):
            # 输出通道数
            stride (int):
            # 步幅
        """
        super().__init__()
        self.in_channels = in_channels  # 保存输入通道数
        self.out_channels = out_channels  # 保存输出通道数
        self.stride = stride  # 保存步幅

    def freeze(self):
        """
        Make this block not trainable.
        使该块不可训练。
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm
        此方法将所有参数设置为`requires_grad=False`，
        并将所有BatchNorm层转换为FrozenBatchNorm

        Returns:
            the block itself
            块本身
        """
        for p in self.parameters():
            p.requires_grad = False  # 设置所有参数为不需要梯度
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)  # 将所有BN层转换为FrozenBatchNorm
        return self  # 返回块本身，便于链式调用


class DepthwiseSeparableConv2d(nn.Module):
    """
    A kxk depthwise convolution + a 1x1 convolution.
    一个kxk深度卷积 + 一个1x1卷积。

    In :paper:`xception`, norm & activation are applied on the second conv.
    在:paper:`xception`中，归一化和激活应用于第二个卷积。
    :paper:`mobilenet` uses norm & activation on both convs.
    :paper:`mobilenet`在两个卷积上都使用归一化和激活。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        *,
        norm1=None,
        activation1=None,
        norm2=None,
        activation2=None,
    ):
        """
        Args:
            norm1, norm2 (str or callable): normalization for the two conv layers.
            # norm1, norm2（字符串或可调用对象）：两个卷积层的归一化。
            activation1, activation2 (callable(Tensor) -> Tensor): activation
                function for the two conv layers.
            # activation1, activation2（可调用(Tensor) -> Tensor）：两个卷积层的激活函数。
        """
        super().__init__()
        # 深度卷积层：对每个输入通道单独进行空间卷积
        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 分组卷积，每组只有一个通道
            bias=not norm1,  # 如果使用归一化，则不使用偏置
            norm=get_norm(norm1, in_channels),  # 获取归一化层
            activation=activation1,  # 设置激活函数
        )
        # 逐点卷积层：使用1x1卷积进行通道间的信息融合
        self.pointwise = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,  # 1x1卷积
            bias=not norm2,  # 如果使用归一化，则不使用偏置
            norm=get_norm(norm2, out_channels),  # 获取归一化层
            activation=activation2,  # 设置激活函数
        )

        # default initialization
        # 默认初始化
        weight_init.c2_msra_fill(self.depthwise)  # 使用MSRA方法初始化深度卷积权重
        weight_init.c2_msra_fill(self.pointwise)  # 使用MSRA方法初始化逐点卷积权重

    def forward(self, x):
        # 先应用深度卷积，再应用逐点卷积
        return self.pointwise(self.depthwise(x))
