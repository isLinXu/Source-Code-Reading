# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from .backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    从配置文件中的MODEL.BACKBONE.NAME构建backbone网络

    Returns:
        an instance of :class:`Backbone`
        返回一个Backbone类的实例
    """
    # 如果没有指定输入形状，则根据配置文件中的像素均值通道数创建默认的输入形状
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    # 从配置文件中获取backbone的名称
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # 通过BACKBONE_REGISTRY注册表获取对应的backbone构建函数，并传入配置和输入形状来构建backbone实例
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    # 确保返回的实例是Backbone类型
    assert isinstance(backbone, Backbone)
    return backbone
