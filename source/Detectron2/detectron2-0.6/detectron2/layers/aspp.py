# Copyright (c) Facebook, Inc. and its affiliates.
# 版权所有 (c) Facebook, Inc. 及其附属公司。

from copy import deepcopy
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .batch_norm import get_norm
from .blocks import DepthwiseSeparableConv2d
from .wrappers import Conv2d


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    空洞空间金字塔池化(ASPP)。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
        pool_kernel_size=None,
        dropout: float = 0.0,
        use_depthwise_separable_conv=False,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            # 输入通道数：ASPP模块的输入通道数量。
            out_channels (int): number of output channels.
            # 输出通道数：ASPP模块的输出通道数量。
            dilations (list): a list of 3 dilations in ASPP.
            # 空洞率列表：ASPP中使用的3个空洞卷积的扩张率。
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            # 归一化类型：应用于所有卷积层的归一化方法。
            # 可参考:func:`layers.get_norm`了解支持的格式。除了全局平均池化后的卷积外，其他卷积层都会应用此归一化。
            activation (callable): activation function.
            # 激活函数：用于各层的激活函数。
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            # 池化核大小：ASPP中图像池化层的平均池化大小(kh, kw)。
            # 如果设为None，则始终执行全局平均池化。
            # 如果不为None，则必须能够被forward()中输入的形状整除。
            # 建议在训练中使用固定的输入特征大小，并设置此选项以匹配此大小，
            # 这样在训练中执行全局平均池化，并在推理中保持池化窗口大小一致。
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
            # dropout比率：应用于ASPP输出的dropout比率。
            # 在官方DeepLab实现中使用0.1的比率。
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                for 3x3 convs in ASPP, proposed in :paper:`DeepLabV3+`.
            # 使用深度可分离卷积：在ASPP中的3x3卷积使用DepthwiseSeparableConv2d，
            # 这是在DeepLabV3+论文中提出的方法。
        """
        super(ASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        # 断言空洞率列表必须包含3个值
        self.pool_kernel_size = pool_kernel_size  # 保存池化核大小
        self.dropout = dropout  # 保存dropout比率
        use_bias = norm == ""  # 如果没有指定归一化方法，则使用偏置项
        self.convs = nn.ModuleList()  # 创建模块列表存储所有卷积层
        # conv 1x1
        # 1x1卷积层
        self.convs.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels),
                activation=deepcopy(activation),
            )
        )
        weight_init.c2_xavier_fill(self.convs[-1])  # 使用xavier初始化卷积权重
        # atrous convs
        # 空洞卷积层
        for dilation in dilations:
            if use_depthwise_separable_conv:
                # 如果使用深度可分离卷积
                self.convs.append(
                    DepthwiseSeparableConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        norm1=norm,
                        activation1=deepcopy(activation),
                        norm2=norm,
                        activation2=deepcopy(activation),
                    )
                )
            else:
                # 普通的带空洞的3x3卷积
                self.convs.append(
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                        activation=deepcopy(activation),
                    )
                )
                weight_init.c2_xavier_fill(self.convs[-1])  # 使用xavier初始化卷积权重
        # image pooling
        # 图像池化分支
        # We do not add BatchNorm because the spatial resolution is 1x1,
        # the original TF implementation has BatchNorm.
        # 我们不添加BatchNorm，因为空间分辨率是1x1，
        # 原始的TensorFlow实现中有BatchNorm。
        if pool_kernel_size is None:
            # 如果没有指定池化核大小，则使用全局平均池化
            image_pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 自适应全局平均池化到1x1
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),  # 1x1卷积
            )
        else:
            # 使用指定大小的平均池化
            image_pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),  # 指定核大小的平均池化
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),  # 1x1卷积
            )
        weight_init.c2_xavier_fill(image_pooling[1])  # 初始化池化后的1x1卷积
        self.convs.append(image_pooling)  # 将池化分支添加到模块列表

        # 最终的投影层：将所有分支的输出连接后通过1x1卷积融合
        self.project = Conv2d(
            5 * out_channels,  # 5个分支，每个分支out_channels个通道
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=deepcopy(activation),
        )
        weight_init.c2_xavier_fill(self.project)  # 初始化投影层权重

    def forward(self, x):
        size = x.shape[-2:]  # 获取输入特征图的高宽
        if self.pool_kernel_size is not None:
            # 如果指定了池化核大小，检查输入尺寸是否能被池化核整除
            if size[0] % self.pool_kernel_size[0] or size[1] % self.pool_kernel_size[1]:
                raise ValueError(
                    "`pool_kernel_size` must be divisible by the shape of inputs. "
                    "Input size: {} `pool_kernel_size`: {}".format(size, self.pool_kernel_size)
                )
        res = []  # 存储各分支的输出结果
        for conv in self.convs:
            res.append(conv(x))  # 将每个分支的输出添加到结果列表
        
        # 将图像池化分支的输出上采样到原始大小
        res[-1] = F.interpolate(res[-1], size=size, mode="bilinear", align_corners=False)
        
        res = torch.cat(res, dim=1)  # 在通道维度上连接所有分支的输出
        res = self.project(res)  # 通过投影层融合特征
        
        # 如果dropout率大于0，则应用dropout
        res = F.dropout(res, self.dropout, training=self.training) if self.dropout > 0 else res
        return res  # 返回最终输出
