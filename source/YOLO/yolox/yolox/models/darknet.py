#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


# class Darknet(nn.Module):
#     # number of blocks from dark2 to dark5.
#     depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

#     def __init__(
#         self,
#         depth,
#         in_channels=3,
#         stem_out_channels=32,
#         out_features=("dark3", "dark4", "dark5"),
#     ):
#         """
#         Args:
#             depth (int): depth of darknet used in model, usually use [21, 53] for this param.
#             in_channels (int): number of input channels, for example, use 3 for RGB image.
#             stem_out_channels (int): number of output channels of darknet stem.
#                 It decides channels of darknet layer2 to layer5.
#             out_features (Tuple[str]): desired output layer name.
#         """
#         super().__init__()
#         assert out_features, "please provide output features of Darknet"
#         self.out_features = out_features
#         self.stem = nn.Sequential(
#             BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
#             *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
#         )
#         in_channels = stem_out_channels * 2  # 64

#         num_blocks = Darknet.depth2blocks[depth]
#         # create darknet with `stem_out_channels` and `num_blocks` layers.
#         # to make model structure more clear, we don't use `for` statement in python.
#         self.dark2 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[0], stride=2)
#         )
#         in_channels *= 2  # 128
#         self.dark3 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[1], stride=2)
#         )
#         in_channels *= 2  # 256
#         self.dark4 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[2], stride=2)
#         )
#         in_channels *= 2  # 512

#         self.dark5 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[3], stride=2),
#             *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
#         )

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    # dark2到dark5的块数
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}  # 定义不同深度对应的块数

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            # depth (int): 模型中使用的darknet深度，通常使用[21, 53]作为参数。
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            # in_channels (int): 输入通道的数量，例如，对于RGB图像使用3。
            stem_out_channels (int): number of output channels of darknet stem.
            # stem_out_channels (int): darknet stem的输出通道数量。
                It decides channels of darknet layer2 to layer5.
                # 它决定darknet层2到层5的通道数。
            out_features (Tuple[str]): desired output layer name.
            # out_features (Tuple[str]): 期望的输出层名称。
        """
        super().__init__()  # 调用父类的初始化方法
        assert out_features, "please provide output features of Darknet"  # 确保提供输出特征
        self.out_features = out_features  # 保存输出特征
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),  # 创建stem层
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),  # 创建组层
        )
        in_channels = stem_out_channels * 2  # 64，更新输入通道数

        num_blocks = Darknet.depth2blocks[depth]  # 获取当前深度对应的块数
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # 使用`stem_out_channels`和`num_blocks`层创建darknet。
        # to make model structure more clear, we don't use [for](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolox/yolox/models/yolo_head.py:213:4-348:42) statement in python.
        # 为了使模型结构更清晰，我们不使用Python中的[for](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolox/yolox/models/yolo_head.py:213:4-348:42)语句。
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)  # 创建dark2层
        )
        in_channels *= 2  # 128，更新输入通道数
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)  # 创建dark3层
        )
        in_channels *= 2  # 256，更新输入通道数
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)  # 创建dark4层
        )
        in_channels *= 2  # 512，更新输入通道数

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),  # 创建dark5层
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),  # 创建SPP块
        )

    # def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
    #     "starts with conv layer then has `num_blocks` `ResLayer`"
    #     return [
    #         BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
    #         *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
    #     ]

    # def make_spp_block(self, filters_list, in_filters):
    #     m = nn.Sequential(
    #         *[
    #             BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
    #             BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
    #             SPPBottleneck(
    #                 in_channels=filters_list[1],
    #                 out_channels=filters_list[0],
    #                 activation="lrelu",
    #             ),
    #             BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
    #             BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
    #         ]
    #     )
    #     return m

    # def forward(self, x):
    #     outputs = {}
    #     x = self.stem(x)
    #     outputs["stem"] = x
    #     x = self.dark2(x)
    #     outputs["dark2"] = x
    #     x = self.dark3(x)
    #     outputs["dark3"] = x
    #     x = self.dark4(x)
    #     outputs["dark4"] = x
    #     x = self.dark5(x)
    #     outputs["dark5"] = x
    #     return {k: v for k, v in outputs.items() if k in self.out_features}
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        # 从卷积层开始，然后有`num_blocks`个`ResLayer`
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),  # 创建卷积层
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],  # 创建指定数量的残差层
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),  # 创建1x1卷积层
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),  # 创建3x3卷积层
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",  # 创建SPP瓶颈层
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),  # 创建3x3卷积层
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),  # 创建1x1卷积层
            ]
        )
        return m  # 返回构建的模块

    def forward(self, x):
        outputs = {}  # 初始化输出字典
        x = self.stem(x)  # 通过stem层处理输入
        outputs["stem"] = x  # 保存stem层的输出
        x = self.dark2(x)  # 通过dark2层处理
        outputs["dark2"] = x  # 保存dark2层的输出
        x = self.dark3(x)  # 通过dark3层处理
        outputs["dark3"] = x  # 保存dark3层的输出
        x = self.dark4(x)  # 通过dark4层处理
        outputs["dark4"] = x  # 保存dark4层的输出
        x = self.dark5(x)  # 通过dark5层处理
        outputs["dark5"] = x  # 保存dark5层的输出
        return {k: v for k, v in outputs.items() if k in self.out_features}  # 返回所需的输出


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()  # 调用父类的初始化方法
        assert out_features, "please provide output features of Darknet"  # 确保提供输出特征
        self.out_features = out_features  # 保存输出特征
        Conv = DWConv if depthwise else BaseConv  # 根据是否使用深度卷积选择卷积层

        base_channels = int(wid_mul * 64)  # 计算基础通道数，乘以宽度因子
        base_depth = max(round(dep_mul * 3), 1)  # 计算基础深度，至少为1

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)  # 创建stem层，输入为3通道，输出为基础通道数

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),  # 创建dark2层的卷积层
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),  # 创建dark2层的CSPLayer
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),  # 创建dark3层的卷积层
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),  # 创建dark3层的CSPLayer
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),  # 创建dark4层的卷积层
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),  # 创建dark4层的CSPLayer
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),  # 创建dark5层的卷积层
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),  # 创建SPP瓶颈层
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),  # 创建dark5层的CSPLayer
        )

    def forward(self, x):
        outputs = {}  # 初始化输出字典
        x = self.stem(x)  # 通过stem层处理输入
        outputs["stem"] = x  # 保存stem层的输出
        x = self.dark2(x)  # 通过dark2层处理
        outputs["dark2"] = x  # 保存dark2层的输出
        x = self.dark3(x)  # 通过dark3层处理
        outputs["dark3"] = x  # 保存dark3层的输出
        x = self.dark4(x)  # 通过dark4层处理
        outputs["dark4"] = x  # 保存dark4层的输出
        x = self.dark5(x)  # 通过dark5层处理
        outputs["dark5"] = x  # 保存dark5层的输出
        return {k: v for k, v in outputs.items() if k in self.out_features}  # 返回所需的输出