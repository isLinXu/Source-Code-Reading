#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# The code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py
import math  # 导入math模块，用于数学计算
from copy import deepcopy  # 从copy模块导入deepcopy函数，用于深拷贝
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    # ModelEMA类用于实现模型的指数移动平均（EMA）

    def __init__(self, model, decay=0.9999, updates=0):
        # 初始化ModelEMA实例
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # 深拷贝模型并设置为评估模式，确保EMA模型不参与梯度计算
        self.updates = updates  # 初始化更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 定义衰减函数
        for param in self.ema.parameters():
            param.requires_grad_(False)  # 将EMA模型的参数设置为不需要梯度

    def update(self, model):
        # 更新EMA模型
        with torch.no_grad():  # 禁用梯度计算
            self.updates += 1  # 更新次数加一
            decay = self.decay(self.updates)  # 计算当前的衰减值

            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()  # 获取模型的状态字典
            for k, item in self.ema.state_dict().items():  # 遍历EMA模型的状态字典
                if item.dtype.is_floating_point:  # 如果参数是浮点类型
                    item *= decay  # 应用衰减
                    item += (1 - decay) * state_dict[k].detach()  # 更新EMA参数

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # 更新模型的属性
        copy_attr(self.ema, model, include, exclude)  # 复制属性

def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from one instance and set them to another instance."""
    # 从一个实例复制属性并设置到另一个实例
    for k, item in b.__dict__.items():  # 遍历实例b的所有属性
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue  # 如果属性不在include中，或者以'_'开头，或者在exclude中，跳过
        else:
            setattr(a, k, item)  # 将属性设置到实例a

def is_parallel(model):
    '''Return True if model's type is DP or DDP, else False.'''
    # 如果模型的类型是DataParallel或DistributedDataParallel，返回True，否则返回False
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    '''De-parallelize a model. Return single-GPU model if model's type is DP or DDP.'''
    # 将模型去并行化。如果模型的类型是DP或DDP，返回单GPU模型
    return model.module if is_parallel(model) else model  # 如果是并行模型，返回其module属性