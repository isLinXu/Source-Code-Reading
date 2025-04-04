# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
import torch
from fvcore.common.param_scheduler import CosineParamScheduler, MultiStepParamScheduler

from detectron2.config import CfgNode

from .lr_scheduler import LRMultiplier, WarmupParamScheduler

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


# 梯度裁剪类型枚举
class GradientClipType(Enum):
    VALUE = "value"  # 按值裁剪
    NORM = "norm"    # 按范数裁剪


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    根据配置创建梯度裁剪闭包，支持按值或按范数裁剪
    """
    cfg = copy.deepcopy(cfg)  # 深拷贝配置防止修改原配置

    # 定义按范数裁剪的函数
    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    # 定义按值裁剪的函数
    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    # 创建类型到裁剪函数的映射
    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    动态创建继承自给定优化器的新类型，重写step方法实现梯度裁剪
    """
    # 检查参数有效性（不能同时使用两种裁剪方式）
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    # 定义新的step方法
    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            # 逐参数裁剪
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            # 全局裁剪
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)  # 调用父类step方法

    # 动态创建新的优化器类
    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",  # 新类名
        (optimizer,),  # 继承自原优化器
        {"step": optimizer_wgc_step},  # 重写step方法
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    根据配置决定是否添加梯度裁剪功能
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer  # 未启用裁剪直接返回原优化器
    
    # 获取优化器类型
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    # 创建梯度裁剪器并生成新优化器类
    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )

    # 处理实例或类的情况
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # 直接修改实例的类（hack方式）
        return optimizer
    else:
        return OptimizerWithGradientClip


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    根据配置构建优化器
    """
    # 获取默认优化器参数
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )
    # 创建带梯度裁剪的SGD优化器
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    获取优化器默认参数列表，支持多种覆盖设置，无覆盖时等效于model.parameters()
    """
    if overrides is None:
        overrides = {}  # 初始化参数覆盖字典
    defaults = {}  # 默认参数容器
    if base_lr is not None:
        defaults["lr"] = base_lr  # 设置基础学习率
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay  # 设置权重衰减系数
    
    # 处理偏置项的超参数覆盖
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        # NOTE: 与Detectron v1不同，现在默认偏置参数与常规权重参数使用相同超参
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor  # 计算偏置项学习率
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias  # 设置偏置项权重衰减
    
    # 合并偏置项覆盖设置
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides  # 将偏置覆盖加入总覆盖字典

    # 定义归一化层类型集合
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    
    params: List[Dict[str, Any]] = []  # 参数组列表
    memo: Set[torch.nn.parameter.Parameter] = set()  # 参数记忆集合防重复
    
    # 遍历模型所有模块
    for module in model.modules():
        # 遍历模块的直接参数（不递归）
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue  # 跳过不需要梯度的参数
            # 避免重复参数
            if value in memo:
                continue
            memo.add(value)  # 记录已处理参数

            # 初始化当前参数的超参配置
            hyperparams = copy.copy(defaults)
            # 处理归一化层的权重衰减覆盖
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            # 应用参数特定覆盖设置
            hyperparams.update(overrides.get(module_param_name, {}))
            # 将参数及其配置加入列表
            params.append({"params": [value], **hyperparams})
    
    return reduce_param_groups(params)  # 优化参数组结构


def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Transform parameter groups into per-parameter structure.
    # Later items in `params` can overwrite parameters set in previous items.
    # 将参数组展开为单参数结构，后续项可以覆盖前面设置
    ret = defaultdict(dict)
    for item in params:
        assert "params" in item
        cur_params = {x: y for x, y in item.items() if x != "params"}
        for param in item["params"]:
            # 为每个参数创建独立配置项
            ret[param].update({"params": [param], **cur_params})
    return list(ret.values())


def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reorganize the parameter groups and merge duplicated groups.
    # The number of parameter groups needs to be as small as possible in order
    # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    # of using a parameter_group per single parameter, we reorganize the
    # parameter groups and merge duplicated groups. This approach speeds
    # up multi-tensor optimizer significantly.
    # 重组参数组，合并相同超参配置的组
    params = _expand_param_groups(params)  # 先展开为单参数结构
    groups = defaultdict(list)  # 按超参配置分组
    
    for item in params:
        # 提取非params的配置项作为分组键
        cur_params = tuple((x, y) for x, y in item.items() if x != "params")
        groups[cur_params].extend(item["params"])  # 合并相同配置的参数
    
    # 重新构建优化器需要的参数组格式
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}  # 还原超参字典
        cur["params"] = param_values  # 添加参数列表
        ret.append(cur)
    return ret


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    根据配置构建学习率调度器
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME  # 获取调度器类型名称

    if name == "WarmupMultiStepLR":
        # 过滤超过最大迭代次数的step点
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        # 创建多步学习率调度器
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        # 创建余弦退火调度器（从1到0）
        sched = CosineParamScheduler(1, 0)
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    # 添加预热阶段
    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,  # 预热起始系数
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),  # 预热比例
        cfg.SOLVER.WARMUP_METHOD,  # 预热方法（如linear/constant）
    )
    # 返回最终调度器
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
