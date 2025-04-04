# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.logger import _log_api_usage  # 导入API使用记录工具
from detectron2.utils.registry import Registry  # 导入注册表工具类

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
# 创建元架构(meta-architectures)的注册表，用于管理不同类型的模型架构
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
用于注册元架构（即整个模型）的注册表

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
注册的对象将使用`obj(cfg)`调用，并期望返回一个`nn.Module`对象
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    构建由``cfg.MODEL.META_ARCHITECTURE``定义的整个模型架构。
    注意：这个函数不会从``cfg``中加载任何权重。
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE  # 从配置中获取元架构名称
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)  # 通过注册表获取并实例化对应的模型类
    model.to(torch.device(cfg.MODEL.DEVICE))  # 将模型移动到指定的设备（CPU/GPU）上
    _log_api_usage("modeling.meta_arch." + meta_arch)  # 记录API使用情况
    return model  # 返回构建好的模型
