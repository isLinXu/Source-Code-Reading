#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )
#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    YOLOv3模型。Darknet 53是该模型的默认主干网络。
    """

    def __init__(
        self,
        depth=1.0,  # 模型深度，默认为1.0
        width=1.0,  # 模型宽度，默认为1.0
        in_features=("dark3", "dark4", "dark5"),  # 输入特征层，默认为("dark3", "dark4", "dark5")
        in_channels=[256, 512, 1024],  # 输入通道数，默认为[256, 512, 1024]
        depthwise=False,  # 是否使用深度可分离卷积，默认为False
        act="silu",  # 激活函数，默认为"silu"
    ):
        super().__init__()  # 调用父类的初始化方法
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)  # 初始化主干网络，使用CSPDarknet
        self.in_features = in_features  # 设置输入特征层
        self.in_channels = in_channels  # 设置输入通道数
        Conv = DWConv if depthwise else BaseConv  # 根据depthwise的值选择卷积类型

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # 定义上采样层，缩放因子为2，使用最近邻插值
        self.lateral_conv0 = BaseConv(  # 创建侧向卷积层
            int(in_channels[2] * width),  # 输入通道数
            int(in_channels[1] * width),  # 输出通道数
            1,  # 卷积核大小
            1,  # 步幅
            act=act  # 激活函数
        )
        self.C3_p4 = CSPLayer(  # 创建CSP层
            int(2 * in_channels[1] * width),  # 输入通道数
            int(in_channels[1] * width),  # 输出通道数
            round(3 * depth),  # 深度
            False,  # 是否使用深度可分离卷积
            depthwise=depthwise,  # 使用深度可分离卷积的标志
            act=act,  # 激活函数
        )  # cat

        self.reduce_conv1 = BaseConv(  # 创建减少通道数的卷积层
            int(in_channels[1] * width),  # 输入通道数
            int(in_channels[0] * width),  # 输出通道数
            1,  # 卷积核大小
            1,  # 步幅
            act=act  # 激活函数
        )
        self.C3_p3 = CSPLayer(  # 创建CSP层
            int(2 * in_channels[0] * width),  # 输入通道数
            int(in_channels[0] * width),  # 输出通道数
            round(3 * depth),  # 深度
            False,  # 是否使用深度可分离卷积
            depthwise=depthwise,  # 使用深度可分离卷积的标志
            act=act,  # 激活函数
        )

        # bottom-up conv
        self.bu_conv2 = Conv(  # 创建底部向上的卷积层
            int(in_channels[0] * width),  # 输入通道数
            int(in_channels[0] * width),  # 输出通道数
            3,  # 卷积核大小
            2,  # 步幅
            act=act  # 激活函数
        )
        self.C3_n3 = CSPLayer(  # 创建CSP层
            int(2 * in_channels[0] * width),  # 输入通道数
            int(in_channels[1] * width),  # 输出通道数
            round(3 * depth),  # 深度
            False,  # 是否使用深度可分离卷积
            depthwise=depthwise,  # 使用深度可分离卷积的标志
            act=act,  # 激活函数
        )

        # bottom-up conv
        self.bu_conv1 = Conv(  # 创建另一个底部向上的卷积层
            int(in_channels[1] * width),  # 输入通道数
            int(in_channels[1] * width),  # 输出通道数
            3,  # 卷积核大小
            2,  # 步幅
            act=act  # 激活函数
        )
        self.C3_n4 = CSPLayer(  # 创建CSP层
            int(2 * in_channels[1] * width),  # 输入通道数
            int(in_channels[2] * width),  # 输出通道数
            round(3 * depth),  # 深度
            False,  # 是否使用深度可分离卷积
            depthwise=depthwise,  # 使用深度可分离卷积的标志
            act=act,  # 激活函数
        )


    def forward(self, input):
        """
        Args:
            inputs: input images.
            inputs: 输入图像。
    
        Returns:
            Tuple[Tensor]: FPN feature.
            Tuple[Tensor]: FPN特征。
        """
    
        #  backbone
        out_features = self.backbone(input)  # 通过主干网络处理输入，得到特征输出
        features = [out_features[f] for f in self.in_features]  # 从输出特征中提取指定的特征层
        [x2, x1, x0] = features  # 解构特征层
    
        fpn_out0 = self.lateral_conv0(x0)  # 通过侧向卷积层处理x0，输出通道从1024变为512，空间分辨率为32
        f_out0 = self.upsample(fpn_out0)  # 对fpn_out0进行上采样，空间分辨率变为16
        f_out0 = torch.cat([f_out0, x1], 1)  # 将上采样后的特征与x1在通道维度上拼接，通道数从512变为1024，空间分辨率为16
        f_out0 = self.C3_p4(f_out0)  # 通过CSP层处理拼接后的特征，输出通道从1024变为512，空间分辨率为16
    
        fpn_out1 = self.reduce_conv1(f_out0)  # 通过减少通道数的卷积层处理f_out0，输出通道从512变为256，空间分辨率为16
        f_out1 = self.upsample(fpn_out1)  # 对fpn_out1进行上采样，空间分辨率变为8
        f_out1 = torch.cat([f_out1, x2], 1)  # 将上采样后的特征与x2在通道维度上拼接，通道数从256变为512，空间分辨率为8
        pan_out2 = self.C3_p3(f_out1)  # 通过CSP层处理拼接后的特征，输出通道从512变为256，空间分辨率为8
    
        p_out1 = self.bu_conv2(pan_out2)  # 通过底部向上的卷积层处理pan_out2，输出通道保持为256，空间分辨率为16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 将p_out1与fpn_out1在通道维度上拼接，通道数从256变为512，空间分辨率为16
        pan_out1 = self.C3_n3(p_out1)  # 通过CSP层处理拼接后的特征，输出通道保持为512，空间分辨率为16
    
        p_out0 = self.bu_conv1(pan_out1)  # 通过另一个底部向上的卷积层处理pan_out1，输出通道保持为512，空间分辨率为32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 将p_out0与fpn_out0在通道维度上拼接，通道数从512变为1024，空间分辨率为32
        pan_out0 = self.C3_n4(p_out0)  # 通过CSP层处理拼接后的特征，输出通道保持为1024，空间分辨率为32
    
        outputs = (pan_out2, pan_out1, pan_out0)  # 将三个输出特征打包成元组
        return outputs  # 返回输出特征
