#!/usr/bin/env python  # 指定脚本的解释器路径
from __future__ import print_function  # 导入print_function以支持Python 2和3的兼容性
from itertools import count  # 从itertools导入count，用于生成无限计数器

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 从PyTorch导入功能性模块

POLY_DEGREE = 4  # 多项式的度数
W_target = torch.randn(POLY_DEGREE, 1) * 5  # 随机生成目标权重，范围乘以5
b_target = torch.randn(1) * 5  # 随机生成目标偏置，范围乘以5


def make_features(x):  # 定义生成特征的函数
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""  # 构建特征矩阵，包含[x, x^2, x^3, x^4]
    x = x.unsqueeze(1)  # 将x的维度扩展为2维
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)  # 生成特征矩阵并返回


def f(x):  # 定义近似函数
    """Approximated function."""  # 近似函数
    return x.mm(W_target) + b_target.item()  # 计算输出


def poly_desc(W, b):  # 定义生成多项式描述的函数
    """Creates a string description of a polynomial."""  # 创建多项式的字符串描述
    result = 'y = '  # 初始化结果字符串
    for i, w in enumerate(W):  # 遍历权重
        result += '{:+.2f} x^{} '.format(w, i + 1)  # 将每个权重添加到结果字符串中
    result += '{:+.2f}'.format(b[0])  # 添加偏置到结果字符串
    return result  # 返回多项式描述


def get_batch(batch_size=32):  # 定义生成批次的函数
    """Builds a batch i.e. (x, f(x)) pair."""  # 构建批次，即(x, f(x))对
    random = torch.randn(batch_size)  # 生成随机输入
    x = make_features(random)  # 生成特征
    y = f(x)  # 计算目标输出
    return x, y  # 返回特征和目标输出


# Define model
fc = torch.nn.Linear(W_target.size(0), 1)  # 定义线性模型，输入维度与目标权重相同，输出为1

for batch_idx in count(1):  # 使用count生成无限计数器
    # Get data
    batch_x, batch_y = get_batch()  # 获取数据批次

    # Reset gradients
    fc.zero_grad()  # 清零梯度

    # Forward pass
    output = F.smooth_l1_loss(fc(batch_x), batch_y)  # 计算模型输出与目标的平滑L1损失
    loss = output.item()  # 获取损失值

    # Backward pass
    output.backward()  # 反向传播计算梯度

    # Apply gradients
    for param in fc.parameters():  # 遍历模型参数
        param.data.add_(-0.1 * param.grad)  # 更新参数

    # Stop criterion
    if loss < 1e-3:  # 如果损失小于阈值
        break  # 退出循环

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))  # 打印损失和批次数量
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))  # 打印学习到的函数
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))  # 打印实际函数