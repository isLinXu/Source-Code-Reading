#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


# class SiLU(nn.Module):
#     """export-friendly version of nn.SiLU()"""

#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    # 适用于导出的nn.SiLU()的版本

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)  # 计算SiLU激活函数的输出


def get_activation(name="silu", inplace=True):
    # 根据名称返回相应的激活函数
    if name == "silu":
        module = nn.SiLU(inplace=inplace)  # 如果名称为silu，返回SiLU激活函数
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)  # 如果名称为relu，返回ReLU激活函数
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)  # 如果名称为lrelu，返回LeakyReLU激活函数
    else:
        raise AttributeError("Unsupported act type: {}".format(name))  # 如果名称不支持，抛出异常
    return module  # 返回激活函数模块


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    # 一个包含Conv2d -> Batchnorm -> silu/leaky relu的模块

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()  # 调用父类的初始化方法
        # same padding
        pad = (ksize - 1) // 2  # 计算填充大小，使得输入输出大小相同
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )  # 创建卷积层
        self.bn = nn.BatchNorm2d(out_channels)  # 创建批归一化层
        self.act = get_activation(act, inplace=True)  # 获取激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # 前向传播，依次通过卷积、批归一化和激活函数

    def fuseforward(self, x):
        return self.act(self.conv(x))  # 融合前向传播，仅通过卷积和激活函数


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    # 深度卷积 + 卷积

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()  # 调用父类的初始化方法
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,  # 使用深度卷积，groups等于输入通道数
            act=act,
        )  # 创建深度卷积层
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )  # 创建逐点卷积层

    def forward(self, x):
        x = self.dconv(x)  # 通过深度卷积层处理输入
        return self.pconv(x)  # 通过逐点卷积层处理输出


class Bottleneck(nn.Module):
    # Standard bottleneck
    # 标准瓶颈结构
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()  # 调用父类的初始化方法
        hidden_channels = int(out_channels * expansion)  # 计算隐藏层通道数
        Conv = DWConv if depthwise else BaseConv  # 根据是否使用深度卷积选择卷积层
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)  # 创建1x1卷积层
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)  # 创建3x3卷积层
        self.use_add = shortcut and in_channels == out_channels  # 判断是否使用shortcut连接

    def forward(self, x):
        y = self.conv2(self.conv1(x))  # 先通过conv1，再通过conv2
        if self.use_add:
            y = y + x  # 如果使用shortcut连接，将输入加到输出上
        return y  # 返回最终的输出

# class ResLayer(nn.Module):
#     "Residual layer with `in_channels` inputs."

#     def __init__(self, in_channels: int):
#         super().__init__()
#         mid_channels = in_channels // 2
#         self.layer1 = BaseConv(
#             in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
#         )
#         self.layer2 = BaseConv(
#             mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
#         )

#     def forward(self, x):
#         out = self.layer2(self.layer1(x))
#         return x + out


# class SPPBottleneck(nn.Module):
#     """Spatial pyramid pooling layer used in YOLOv3-SPP"""

#     def __init__(
#         self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
#     ):
#         super().__init__()
#         hidden_channels = in_channels // 2
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
#         self.m = nn.ModuleList(
#             [
#                 nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
#                 for ks in kernel_sizes
#             ]
#         )
#         conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
#         self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.cat([x] + [m(x) for m in self.m], dim=1)
#         x = self.conv2(x)
#         return x


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."  # 残差层，具有 `in_channels` 输入通道。

    def __init__(self, in_channels: int):
        super().__init__()  # 调用父类的构造函数
        mid_channels = in_channels // 2  # 中间通道数为输入通道数的一半
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"  # 第一层卷积，输入通道为in_channels，输出通道为mid_channels，卷积核大小为1，步幅为1，激活函数为lrelu
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"  # 第二层卷积，输入通道为mid_channels，输出通道为in_channels，卷积核大小为3，步幅为1，激活函数为lrelu
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))  # 先通过layer1处理输入x，再通过layer2处理输出
        return x + out  # 返回输入x与输出的和，形成残差连接


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""  # 用于YOLOv3-SPP的空间金字塔池化层

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()  # 调用父类的构造函数
        hidden_channels = in_channels // 2  # 隐藏通道数为输入通道数的一半
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)  # 第一层卷积，输入通道为in_channels，输出通道为hidden_channels，卷积核大小为1，步幅为1，激活函数为指定的activation
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)  # 创建一个最大池化层列表，池化核大小由kernel_sizes指定，步幅为1，填充为核大小的一半
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)  # 第二层卷积的输出通道数为隐藏通道数乘以池化层数量加1
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)  # 第二层卷积，输入通道为conv2_channels，输出通道为out_channels，卷积核大小为1，步幅为1，激活函数为指定的activation

    def forward(self, x):
        x = self.conv1(x)  # 通过第一层卷积处理输入x
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)  # 将x与所有池化层的输出在通道维度上进行拼接
        x = self.conv2(x)  # 通过第二层卷积处理拼接后的输出
        return x  # 返回最终输出


# class CSPLayer(nn.Module):
#     """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         n=1,
#         shortcut=True,
#         expansion=0.5,
#         depthwise=False,
#         act="silu",
#     ):
#         """
#         Args:
#             in_channels (int): input channels.
#             out_channels (int): output channels.
#             n (int): number of Bottlenecks. Default value: 1.
#         """
#         # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         hidden_channels = int(out_channels * expansion)  # hidden channels
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
#         module_list = [
#             Bottleneck(
#                 hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
#             )
#             for _ in range(n)
#         ]
#         self.m = nn.Sequential(*module_list)

#     def forward(self, x):
#         x_1 = self.conv1(x)
#         x_2 = self.conv2(x)
#         x_1 = self.m(x_1)
#         x = torch.cat((x_1, x_2), dim=1)
#         return self.conv3(x)


# class Focus(nn.Module):
#     """Focus width and height information into channel space."""

#     def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
#         super().__init__()
#         self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

#     def forward(self, x):
#         # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
#         patch_top_left = x[..., ::2, ::2]
#         patch_top_right = x[..., ::2, 1::2]
#         patch_bot_left = x[..., 1::2, ::2]
#         patch_bot_right = x[..., 1::2, 1::2]
#         x = torch.cat(
#             (
#                 patch_top_left,
#                 patch_bot_left,
#                 patch_top_right,
#                 patch_bot_right,
#             ),
#             dim=1,
#         )
#         return self.conv(x)

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""  # yolov5中的C3，具有3个卷积的CSP瓶颈

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.  # 输入通道数
            out_channels (int): output channels.  # 输出通道数
            n (int): number of Bottlenecks. Default value: 1.  # Bottleneck的数量，默认值为1
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # 调用父类的构造函数
        hidden_channels = int(out_channels * expansion)  # 隐藏通道数为输出通道数乘以扩展因子
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)  # 第一层卷积，输入通道为in_channels，输出通道为hidden_channels，卷积核大小为1，步幅为1，激活函数为指定的act
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)  # 第二层卷积，输入通道为in_channels，输出通道为hidden_channels，卷积核大小为1，步幅为1，激活函数为指定的act
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)  # 第三层卷积，输入通道为2倍的hidden_channels，输出通道为out_channels，卷积核大小为1，步幅为1，激活函数为指定的act
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act  # 创建Bottleneck模块，输入和输出通道均为hidden_channels，使用shortcut连接，扩展因子为1.0，是否使用深度卷积由depthwise决定，激活函数为指定的act
            )
            for _ in range(n)  # 根据n的值创建n个Bottleneck模块
        ]
        self.m = nn.Sequential(*module_list)  # 将所有Bottleneck模块放入顺序容器中

    def forward(self, x):
        x_1 = self.conv1(x)  # 通过第一层卷积处理输入x
        x_2 = self.conv2(x)  # 通过第二层卷积处理输入x
        x_1 = self.m(x_1)  # 通过Bottleneck模块处理x_1
        x = torch.cat((x_1, x_2), dim=1)  # 在通道维度上将x_1和x_2进行拼接
        return self.conv3(x)  # 通过第三层卷积处理拼接后的输出并返回


class Focus(nn.Module):
    """Focus width and height information into channel space."""  # 将宽度和高度信息聚焦到通道空间

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()  # 调用父类的构造函数
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)  # 创建卷积层，输入通道为in_channels的4倍，输出通道为out_channels，卷积核大小和步幅由参数指定，激活函数为指定的act

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)  # 输入x的形状为(b,c,w,h)，输出y的形状为(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]  # 获取左上角的补丁
        patch_top_right = x[..., ::2, 1::2]  # 获取右上角的补丁
        patch_bot_left = x[..., 1::2, ::2]  # 获取左下角的补丁
        patch_bot_right = x[..., 1::2, 1::2]  # 获取右下角的补丁
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,  # 在通道维度上拼接所有补丁
        )
        return self.conv(x)  # 通过卷积层处理拼接后的输出并返回
