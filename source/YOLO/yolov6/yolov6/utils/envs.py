#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os  # 导入os模块，用于处理操作系统相关的功能
import random  # 导入random模块，用于生成随机数
import numpy as np  # 导入numpy库，用于数值计算

import torch  # 导入PyTorch库
import torch.backends.cudnn as cudnn  # 导入cudnn模块，用于设置CUDA后端
from yolov6.utils.events import LOGGER  # 从yolov6.utils.events导入日志记录器

def get_envs():
    """Get PyTorch needed environments from system environments."""
    # 从系统环境中获取PyTorch所需的环境变量
    local_rank = int(os.getenv('LOCAL_RANK', -1))  # 获取本地进程的rank，默认为-1
    rank = int(os.getenv('RANK', -1))  # 获取全局进程的rank，默认为-1
    world_size = int(os.getenv('WORLD_SIZE', 1))  # 获取全局进程的总数，默认为1
    return local_rank, rank, world_size  # 返回本地rank、全局rank和总进程数

def select_device(device):
    """Set devices' information to the program.
    Args:
        device: a string, like 'cpu' or '1,2,3,4'
    Returns:
        torch.device
    """
    # 设置设备信息
    if device == 'cpu':  # 如果设备为CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 设置CUDA不可用
        LOGGER.info('Using CPU for training... ')  # 记录使用CPU进行训练的信息
    elif device:  # 如果指定了设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置可见的CUDA设备
        assert torch.cuda.is_available()  # 确保CUDA可用
        nd = len(device.strip().split(','))  # 计算可用的GPU数量
        LOGGER.info(f'Using {nd} GPU for training... ')  # 记录使用GPU进行训练的信息
    cuda = device != 'cpu' and torch.cuda.is_available()  # 判断是否使用CUDA
    device = torch.device('cuda:0' if cuda else 'cpu')  # 设置设备为CUDA或CPU
    return device  # 返回设备对象

def set_random_seed(seed, deterministic=False):
    """ Set random state to random library, numpy, torch and cudnn.
    Args:
        seed: int value.
        deterministic: bool value.
    """
    # 设置随机种子
    random.seed(seed)  # 设置Python内置random模块的种子
    np.random.seed(seed)  # 设置numpy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    if deterministic:  # 如果需要确定性结果
        cudnn.deterministic = True  # 设置cudnn为确定性模式
        cudnn.benchmark = False  # 禁用cudnn的基准测试
    else:  # 如果不需要确定性结果
        cudnn.deterministic = False  # 设置cudnn为非确定性模式
        cudnn.benchmark = True  # 启用cudnn的基准测试