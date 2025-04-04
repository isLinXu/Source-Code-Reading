# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
import torch.nn as nn

from detectron2.layers import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    网络骨干的抽象基类，所有具体的backbone网络都需要继承这个类。
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        子类的`__init__`方法可以指定自己的参数集。这里提供了一个基础的初始化实现。
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        
        子类必须重写这个方法，但需要保持相同的返回类型。
        这个方法定义了backbone的前向传播过程，处理输入特征并生成特征图。
        
        返回值：
            dict[str->Tensor]: 从特征名称（例如"res2"）到对应特征张量的映射字典
        """
        pass

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        
        某些backbone要求输入的高度和宽度必须能被特定的整数整除。
        这种情况在具有横向连接的编码器/解码器类型网络（如FPN）中很常见，
        因为特征图需要在"自下而上"和"自上而下"的路径中匹配尺寸。
        如果不需要特定的输入尺寸整除要求，则设置为0。
        """
        return 0

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        
        返回一个字典，描述每个输出特征图的形状规格。
        ShapeSpec包含了通道数(channels)和步长(stride)等信息。
        这是一个向后兼容的默认实现。
        """
        # this is a backward-compatible default
        # 这是一个向后兼容的默认实现
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
