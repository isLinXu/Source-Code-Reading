#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os  # 导入os模块，用于处理文件和目录
import math  # 导入math模块，用于数学计算

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

from yolov6.utils.events import LOGGER  # 从yolov6.utils.events导入日志记录器

def build_optimizer(cfg, model):
    """ Build optimizer from cfg file."""
    # 从配置文件构建优化器
    g_bnw, g_w, g_b = [], [], []  # 初始化三个列表，分别用于存储BatchNorm权重、权重和偏置
    for v in model.modules():  # 遍历模型中的所有模块
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # 如果模块有偏置并且是nn.Parameter类型
            g_b.append(v.bias)  # 将偏置添加到g_b列表
        if isinstance(v, nn.BatchNorm2d):  # 如果模块是BatchNorm2d类型
            g_bnw.append(v.weight)  # 将BatchNorm的权重添加到g_bnw列表
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # 如果模块有权重并且是nn.Parameter类型
            g_w.append(v.weight)  # 将权重添加到g_w列表

    assert cfg.solver.optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
    # 检查优化器类型，如果不是SGD或Adam则抛出错误
    if cfg.solver.optim == 'SGD':  # 如果优化器是SGD
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
        # 创建SGD优化器，使用BatchNorm权重、学习率、动量和Nesterov加速
    elif cfg.solver.optim == 'Adam':  # 如果优化器是Adam
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
        # 创建Adam优化器，使用BatchNorm权重、学习率和动量参数

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})  # 添加权重参数组
    optimizer.add_param_group({'params': g_b})  # 添加偏置参数组

    del g_bnw, g_w, g_b  # 删除临时变量以释放内存
    return optimizer  # 返回构建的优化器

def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    # 从配置文件构建学习率调度器
    if cfg.solver.lr_scheduler == 'Cosine':  # 如果学习率调度器为余弦调度
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
        # 定义余弦调度的学习率函数
    elif cfg.solver.lr_scheduler == 'Constant':  # 如果学习率调度器为常量
        lf = lambda x: 1.0  # 学习率始终为1.0
    else:
        LOGGER.error('unknown lr scheduler, use Cosine defaulted')  # 记录错误信息，未知的学习率调度器

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 创建学习率调度器
    return scheduler, lf  # 返回调度器和学习率函数