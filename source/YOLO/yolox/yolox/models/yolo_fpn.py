#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import Darknet
from .network_blocks import BaseConv


# class YOLOFPN(nn.Module):
#     """
#     YOLOFPN module. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=53,
#         in_features=["dark3", "dark4", "dark5"],
#     ):
#         super().__init__()

#         self.backbone = Darknet(depth)
#         self.in_features = in_features

#         # out 1
#         self.out1_cbl = self._make_cbl(512, 256, 1)
#         self.out1 = self._make_embedding([256, 512], 512 + 256)

#         # out 2
#         self.out2_cbl = self._make_cbl(256, 128, 1)
#         self.out2 = self._make_embedding([128, 256], 256 + 128)

#         # upsample
#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

#     def _make_cbl(self, _in, _out, ks):
#         return BaseConv(_in, _out, ks, stride=1, act="lrelu")

#     def _make_embedding(self, filters_list, in_filters):
#         m = nn.Sequential(
#             *[
#                 self._make_cbl(in_filters, filters_list[0], 1),
#                 self._make_cbl(filters_list[0], filters_list[1], 3),
#                 self._make_cbl(filters_list[1], filters_list[0], 1),
#                 self._make_cbl(filters_list[0], filters_list[1], 3),
#                 self._make_cbl(filters_list[1], filters_list[0], 1),
#             ]
#         )
#         return m

#     def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
#         with open(filename, "rb") as f:
#             state_dict = torch.load(f, map_location="cpu")
#         print("loading pretrained weights...")
#         self.backbone.load_state_dict(state_dict)

class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    YOLOFPN模块。Darknet 53是该模型的默认主干网络。
    """

    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
    ):
        super().__init__()

        self.backbone = Darknet(depth)  # 初始化主干网络，使用Darknet，深度为53
        self.in_features = in_features  # 输入特征，默认为["dark3", "dark4", "dark5"]

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)  # 创建第一个输出的卷积块
        self.out1 = self._make_embedding([256, 512], 512 + 256)  # 创建第一个输出的嵌入层

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)  # 创建第二个输出的卷积块
        self.out2 = self._make_embedding([128, 256], 256 + 128)  # 创建第二个输出的嵌入层

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # 定义上采样层，缩放因子为2，使用最近邻插值

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")  # 创建卷积-批归一化-激活层的组合

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),  # 第一个卷积块
                self._make_cbl(filters_list[0], filters_list[1], 3),  # 第二个卷积块
                self._make_cbl(filters_list[1], filters_list[0], 1),  # 第三个卷积块
                self._make_cbl(filters_list[0], filters_list[1], 3),  # 第四个卷积块
                self._make_cbl(filters_list[1], filters_list[0], 1),  # 第五个卷积块
            ]
        )
        return m  # 返回嵌入层的序列

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:  # 以二进制模式打开预训练模型文件
            state_dict = torch.load(f, map_location="cpu")  # 加载模型的状态字典到CPU
        print("loading pretrained weights...")  # 打印加载预训练权重的提示信息
        self.backbone.load_state_dict(state_dict)  # 将加载的状态字典应用到主干网络


    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.
            inputs (Tensor): 输入图像。

        Returns:
            Tuple[Tensor]: FPN output features..
            Tuple[Tensor]: FPN输出特征。
        """
        #  backbone
        out_features = self.backbone(inputs)  # 通过主干网络处理输入，得到特征输出
        x2, x1, x0 = [out_features[f] for f in self.in_features]  # 从输出特征中提取指定的特征层

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)  # 将特征x0输入到第一个YOLO分支的卷积块中
        x1_in = self.upsample(x1_in)  # 对输出进行上采样
        x1_in = torch.cat([x1_in, x1], 1)  # 将上采样后的特征与x1在通道维度上拼接
        out_dark4 = self.out1(x1_in)  # 通过第一个YOLO分支生成输出特征

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)  # 将第一个分支的输出特征输入到第二个YOLO分支的卷积块中
        x2_in = self.upsample(x2_in)  # 对输出进行上采样
        x2_in = torch.cat([x2_in, x2], 1)  # 将上采样后的特征与x2在通道维度上拼接
        out_dark3 = self.out2(x2_in)  # 通过第二个YOLO分支生成输出特征

        outputs = (out_dark3, out_dark4, x0)  # 将两个YOLO分支的输出和x0打包成元组
        return outputs  # 返回输出特征