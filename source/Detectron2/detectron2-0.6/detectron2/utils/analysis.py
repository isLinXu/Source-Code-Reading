# Copyright (c) Facebook, Inc. and its affiliates.
# -*- coding: utf-8 -*-

import typing  # 导入类型提示模块
from typing import Any, List  # 导入类型提示的Any和List类型
import fvcore  # 导入fvcore库
from fvcore.nn import activation_count, flop_count, parameter_count, parameter_count_table  # 从fvcore.nn导入计数函数
from torch import nn  # 从torch导入nn模块

from detectron2.export import TracingAdapter  # 从detectron2.export导入追踪适配器

__all__ = [  # 定义模块公开的函数列表
    "activation_count_operators",
    "flop_count_operators",
    "parameter_count_table",
    "parameter_count",
    "FlopCountAnalysis",
]

FLOPS_MODE = "flops"  # 定义FLOPS计数模式常量
ACTIVATIONS_MODE = "activations"  # 定义激活计数模式常量


# Some extra ops to ignore from counting, including elementwise and reduction ops
# 一些需要从计数中忽略的额外操作，包括元素级操作和归约操作
_IGNORED_OPS = {
    "aten::add",  # 加法操作
    "aten::add_",  # 原地加法操作
    "aten::argmax",  # 最大值索引操作
    "aten::argsort",  # 排序索引操作
    "aten::batch_norm",  # 批量归一化操作
    "aten::constant_pad_nd",  # 常数填充操作
    "aten::div",  # 除法操作
    "aten::div_",  # 原地除法操作
    "aten::exp",  # 指数操作
    "aten::log2",  # 对数操作
    "aten::max_pool2d",  # 最大池化操作
    "aten::meshgrid",  # 网格生成操作
    "aten::mul",  # 乘法操作
    "aten::mul_",  # 原地乘法操作
    "aten::neg",  # 取负操作
    "aten::nonzero_numpy",  # 非零元素查找操作
    "aten::reciprocal",  # 倒数操作
    "aten::rsub",  # 右减操作
    "aten::sigmoid",  # sigmoid激活函数
    "aten::sigmoid_",  # 原地sigmoid激活函数
    "aten::softmax",  # softmax操作
    "aten::sort",  # 排序操作
    "aten::sqrt",  # 平方根操作
    "aten::sub",  # 减法操作
    "torchvision::nms",  # TODO estimate flop for nms  # 非极大值抑制操作（尚未实现FLOP估计）
}


class FlopCountAnalysis(fvcore.nn.FlopCountAnalysis):
    """
    Same as :class:`fvcore.nn.FlopCountAnalysis`, but supports detectron2 models.
    
    与:class:`fvcore.nn.FlopCountAnalysis`相同，但支持detectron2模型。
    """

    def __init__(self, model, inputs):
        """
        Args:
            model (nn.Module):
            inputs (Any): inputs of the given model. Does not have to be tuple of tensors.
            
        参数:
            model (nn.Module): 要分析的模型
            inputs (Any): 给定模型的输入。不必是张量的元组。
        """
        wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)  # 创建追踪适配器，允许非张量输入
        super().__init__(wrapper, wrapper.flattened_inputs)  # 调用父类初始化方法
        self.set_op_handle(**{k: None for k in _IGNORED_OPS})  # 设置忽略的操作处理器为None


def flop_count_operators(model: nn.Module, inputs: list) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level flops counting using jit.
    This is a wrapper of :func:`fvcore.nn.flop_count` and adds supports for standard
    detection models in detectron2.
    Please use :class:`FlopCountAnalysis` for more advanced functionalities.
    
    使用jit实现操作符级别的FLOP计数。
    这是:func:`fvcore.nn.flop_count`的包装器，增加了对detectron2中标准检测模型的支持。
    请使用:class:`FlopCountAnalysis`获取更高级的功能。

    Note:
        The function runs the input through the model to compute flops.
        The flops of a detection model is often input-dependent, for example,
        the flops of box & mask head depends on the number of proposals &
        the number of detected objects.
        Therefore, the flops counting using a single input may not accurately
        reflect the computation cost of a model. It's recommended to average
        across a number of inputs.
        
    注意:
        该函数通过模型运行输入来计算FLOP。
        检测模型的FLOP通常依赖于输入，例如，
        框和掩码头的FLOP取决于建议的数量和
        检测到的对象的数量。
        因此，使用单个输入进行FLOP计数可能无法准确
        反映模型的计算成本。建议在多个输入上取平均值。

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
            Only "image" key will be used.
        supported_ops (dict[str, Handle]): see documentation of :func:`fvcore.nn.flop_count`
        
    参数:
        model: 接受`list[dict]`作为输入的detectron2模型。
        inputs (list[dict]): 模型输入，采用detectron2的标准格式。
            只会使用"image"键。
        supported_ops (dict[str, Handle]): 参见:func:`fvcore.nn.flop_count`的文档

    Returns:
        Counter: Gflop count per operator
        
    返回:
        Counter: 每个操作符的Gflop计数
    """
    old_train = model.training  # 保存模型当前的训练状态
    model.eval()  # 将模型设置为评估模式
    ret = FlopCountAnalysis(model, inputs).by_operator()  # 使用FlopCountAnalysis分析模型并按操作符分类结果
    model.train(old_train)  # 恢复模型原来的训练状态
    return {k: v / 1e9 for k, v in ret.items()}  # 将结果转换为Gflop（除以10^9）并返回


def activation_count_operators(
    model: nn.Module, inputs: list, **kwargs
) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level activations counting using jit.
    This is a wrapper of fvcore.nn.activation_count, that supports standard detection models
    in detectron2.
    
    使用jit实现操作符级别的激活计数。
    这是fvcore.nn.activation_count的包装器，支持detectron2中的标准检测模型。

    Note:
        The function runs the input through the model to compute activations.
        The activations of a detection model is often input-dependent, for example,
        the activations of box & mask head depends on the number of proposals &
        the number of detected objects.
        
    注意:
        该函数通过模型运行输入来计算激活。
        检测模型的激活通常依赖于输入，例如，
        框和掩码头的激活取决于建议的数量和
        检测到的对象的数量。

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
            Only "image" key will be used.
            
    参数:
        model: 接受`list[dict]`作为输入的detectron2模型。
        inputs (list[dict]): 模型输入，采用detectron2的标准格式。
            只会使用"image"键。

    Returns:
        Counter: activation count per operator
        
    返回:
        Counter: 每个操作符的激活计数
    """
    return _wrapper_count_operators(model=model, inputs=inputs, mode=ACTIVATIONS_MODE, **kwargs)  # 调用通用包装函数进行激活计数


def _wrapper_count_operators(
    model: nn.Module, inputs: list, mode: str, **kwargs
) -> typing.DefaultDict[str, float]:
    # ignore some ops
    # 忽略一些操作
    supported_ops = {k: lambda *args, **kwargs: {} for k in _IGNORED_OPS}  # 为忽略的操作创建空处理函数
    supported_ops.update(kwargs.pop("supported_ops", {}))  # 更新用户提供的支持操作
    kwargs["supported_ops"] = supported_ops  # 将支持的操作添加到kwargs中

    assert len(inputs) == 1, "Please use batch size=1"  # 确保输入批次大小为1
    tensor_input = inputs[0]["image"]  # 获取图像张量输入
    inputs = [{"image": tensor_input}]  # 移除其他键，以防有任何其他键存在

    old_train = model.training  # 保存模型的训练状态
    if isinstance(model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)):
        model = model.module  # 如果模型是并行包装的，获取其原始模块
    wrapper = TracingAdapter(model, inputs)  # 创建模型的追踪适配器
    wrapper.eval()  # 将包装器设置为评估模式
    if mode == FLOPS_MODE:
        ret = flop_count(wrapper, (tensor_input,), **kwargs)  # 如果是FLOPS模式，调用flop_count
    elif mode == ACTIVATIONS_MODE:
        ret = activation_count(wrapper, (tensor_input,), **kwargs)  # 如果是激活模式，调用activation_count
    else:
        raise NotImplementedError("Count for mode {} is not supported yet.".format(mode))  # 不支持其他模式时抛出异常
    # compatible with change in fvcore
    # 兼容fvcore中的更改
    if isinstance(ret, tuple):
        ret = ret[0]  # 如果返回值是元组，取第一个元素
    model.train(old_train)  # 恢复模型的训练状态
    return ret  # 返回计数结果


def find_unused_parameters(model: nn.Module, inputs: Any) -> List[str]:
    """
    Given a model, find parameters that do not contribute
    to the loss.
    
    给定一个模型，找出不参与损失计算的参数。

    Args:
        model: a model in training mode that returns losses
        inputs: argument or a tuple of arguments. Inputs of the model
        
    参数:
        model: 处于训练模式并返回损失的模型
        inputs: 参数或参数的元组。模型的输入

    Returns:
        list[str]: the name of unused parameters
        
    返回:
        list[str]: 未使用参数的名称
    """
    assert model.training  # 确保模型处于训练模式
    for _, prm in model.named_parameters():
        prm.grad = None  # 清除所有参数的梯度

    if isinstance(inputs, tuple):
        losses = model(*inputs)  # 如果输入是元组，解包后传入模型
    else:
        losses = model(inputs)  # 否则直接传入模型

    if isinstance(losses, dict):
        losses = sum(losses.values())  # 如果损失是字典，将所有损失值相加
    losses.backward()  # 反向传播计算梯度

    unused: List[str] = []  # 初始化未使用参数列表
    for name, prm in model.named_parameters():
        if prm.grad is None:
            unused.append(name)  # 如果参数的梯度为None，将其添加到未使用列表中
        prm.grad = None  # 清除梯度
    return unused  # 返回未使用参数的名称列表
