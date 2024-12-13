import torch  # 导入PyTorch主要库
import numpy as np  # 导入numpy用于数值计算
import torch.nn.functional as F  # 导入PyTorch函数式接口


def transI_fusebn(kernel, bn):
    # 用于融合卷积核和BN层的参数
    # kernel: 卷积核参数
    # bn: BatchNorm层
    gamma = bn.weight  # 获取BN层的缩放因子gamma
    std = (bn.running_var + bn.eps).sqrt()  # 计算标准差
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std  # 返回融合后的卷积核和偏置


def transII_addbranch(kernels, biases):
    # 用于将多个分支的卷积核和偏置相加
    # kernels: 多个卷积核的列表
    # biases: 多个偏置的列表
    return sum(kernels), sum(biases)  # 返回所有分支的卷积核和和偏置和


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    # 用于融合1x1卷积和kxk卷积
    # k1, b1: 1x1卷积的参数
    # k2, b2: kxk卷积的参数
    # groups: 分组卷积的组数
    if groups == 1:
        # 标准卷积的情况
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))  # 计算等效卷积核
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))  # 计算等效偏置
    else:
        # 分组卷积的情况
        k_slices = []  # 存储每个分组的卷积核
        b_slices = []  # 存储每个分组的偏置
        k1_T = k1.permute(1, 0, 2, 3)  # 转置k1的维度
        k1_group_width = k1.size(0) // groups  # 计算每组的通道数
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            # 对每个分组进行处理
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]  # 提取当前分组的k1参数
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]  # 提取当前分组的k2参数
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))  # 计算当前分组的等效卷积核
            b_slices.append((k2_slice * b1[g * k1_group_width:(g+1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))  # 计算当前分组的等效偏置
        k, b_hat = transIV_depthconcat(k_slices, b_slices)  # 将所有分组的结果拼接
    return k, b_hat + b2  # 返回最终的等效卷积核和偏置


def transIV_depthconcat(kernels, biases):
    # 用于在通道维度上拼接卷积核和偏置
    # kernels: 卷积核列表
    # biases: 偏置列表
    return torch.cat(kernels, dim=0), torch.cat(biases)  # 返回拼接后的结果


def transV_avg(channels, kernel_size, groups):
    # 创建平均池化的等效卷积核
    # channels: 通道数
    # kernel_size: 核大小
    # groups: 分组数
    input_dim = channels // groups  # 计算输入维度
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))  # 创建零张量
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2  # 设置权重
    return k  # 返回等效卷积核


#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
# 注意：此函数未经过非方形卷积核(高!=宽)和偶数大小卷积核的测试
def transVI_multiscale(kernel, target_kernel_size):
    # 将卷积核调整到目标大小
    # kernel: 原始卷积核
    # target_kernel_size: 目标核大小
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2  # 计算高度方向需要填充的像素数
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2  # 计算宽度方向需要填充的像素数
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])  # 返回填充后的卷积核
