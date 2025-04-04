# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from collections import namedtuple


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    一个包含张量基本形状规格的简单结构。
    
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    它通常用作模型的辅助输入/输出，
    用于补充PyTorch模块中缺乏的形状推断能力。

    Attributes:
        channels:
        # channels: 特征通道数
        height:
        # height: 特征图高度
        width:
        # width: 特征图宽度
        stride:
        # stride: 特征图相对于原始图像的步幅
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        # 创建一个新的ShapeSpec实例
        # 所有参数都是可选的，可以为None
        return super().__new__(cls, channels, height, width, stride)
