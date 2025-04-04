# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.utils.registry import Registry

PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")
PROPOSAL_GENERATOR_REGISTRY.__doc__ = """
Registry for proposal generator, which produces object proposals from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

from . import rpn, rrpn  # noqa F401 isort:skip


def build_proposal_generator(cfg, input_shape):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    从`cfg.MODEL.PROPOSAL_GENERATOR.NAME`构建一个候选框生成器。
    如果名称为"PrecomputedProposals"，则表示不使用候选框生成器。
    """
    # 从配置中获取候选框生成器的名称
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    # 如果使用预计算的候选框，则不需要生成器，返回None
    if name == "PrecomputedProposals":
        return None

    # 通过注册表获取对应的候选框生成器类，并使用配置和输入形状进行实例化
    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, input_shape)
