#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import time  # 导入时间模块
from contextlib import contextmanager  # 从上下文管理模块导入上下文管理器
from copy import deepcopy  # 从复制模块导入深拷贝
import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入分布式训练模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络功能模块
from yolov6.utils.events import LOGGER  # 从YOLOv6的事件模块导入记录器

try:
    import thop  # for FLOPs computation  # 尝试导入thop库，用于计算FLOPs
except ImportError:
    thop = None  # 如果导入失败，则将thop设置为None


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    装饰器，确保在分布式训练中，所有进程等待每个local_master完成某些操作。
    """
    if local_rank not in [-1, 0]:  # 如果local_rank不是-1或0
        dist.barrier(device_ids=[local_rank])  # 在指定的设备上设置障碍
    yield  # 让出控制权
    if local_rank == 0:  # 如果local_rank是0
        dist.barrier(device_ids=[0])  # 在设备0上设置障碍


def time_sync():
    '''Waits for all kernels in all streams on a CUDA device to complete if cuda is available.'''
    # 如果CUDA可用，等待所有CUDA设备上的所有流中的所有内核完成
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 同步CUDA设备
    return time.time()  # 返回当前时间


def initialize_weights(model):
    # 初始化模型的权重
    for m in model.modules():  # 遍历模型中的所有模块
        t = type(m)  # 获取模块的类型
        if t is nn.Conv2d:  # 如果模块是卷积层
            pass  # 不进行初始化
        elif t is nn.BatchNorm2d:  # 如果模块是批归一化层
            m.eps = 1e-3  # 设置epsilon值
            m.momentum = 0.03  # 设置动量值
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:  # 如果模块是激活函数
            m.inplace = True  # 设置为就地操作


def fuse_conv_and_bn(conv, bn):
    '''Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/.'''
    # 融合卷积层和批归一化层
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,  # 输入通道数
            conv.out_channels,  # 输出通道数
            kernel_size=conv.kernel_size,  # 卷积核大小
            stride=conv.stride,  # 步幅
            padding=conv.padding,  # 填充
            groups=conv.groups,  # 组数
            bias=True,  # 使用偏置
        )
        .requires_grad_(False)  # 不需要梯度
        .to(conv.weight.device)  # 将其移动到卷积层的设备上
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)  # 准备卷积权重
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))  # 准备批归一化权重
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))  # 复制融合权重

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)  # 初始化卷积偏置
        if conv.bias is None  # 如果没有偏置
        else conv.bias  # 否则使用卷积的偏置
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(  # 准备批归一化的偏置
        torch.sqrt(bn.running_var + bn.eps)  # 计算偏置
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)  # 复制融合偏置

    return fusedconv  # 返回融合后的卷积层


def fuse_model(model):
    '''Fuse convolution and batchnorm layers of the model.'''
    # 融合模型中的卷积层和批归一化层
    from yolov6.layers.common import ConvModule  # 从YOLOv6的通用层导入ConvModule

    for m in model.modules():  # 遍历模型中的所有模块
        if type(m) is ConvModule and hasattr(m, "bn"):  # 如果模块是ConvModule并且有bn属性
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
            delattr(m, "bn")  # 删除批归一化层
            m.forward = m.forward_fuse  # 更新前向传播方法
    return model  # 返回融合后的模型


def get_model_info(model, img_size=640):
    """Get model Params and GFlops.
    Code base on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/model_utils.py
    获取模型的参数和GFlops。
    代码基于 https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/model_utils.py
    """
    from thop import profile  # 从thop导入profile函数
    stride = 64  # 设置步幅
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)  # 创建输入图像

    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)  # 计算FLOPs和参数
    params /= 1e6  # 将参数转换为百万
    flops /= 1e9  # 将FLOPs转换为十亿
    img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # 确保图像大小是列表
    flops *= img_size[0] * img_size[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)  # 格式化输出信息
    return info  # 返回模型信息