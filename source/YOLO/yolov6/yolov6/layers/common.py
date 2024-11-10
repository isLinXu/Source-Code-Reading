#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import warnings
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from yolov6.utils.general import download_ckpt


activation_table = {'relu':nn.ReLU(),
                    'silu':nn.SiLU(),
                    'hardswish':nn.Hardswish()
                    }

class SiLU(nn.Module):
    '''Activation of SiLU'''
    '''
    SiLU激活函数类

    SiLU（Sigmoid Linear Unit）是一种激活函数，其定义为 f(x) = x * sigmoid(x)。
    这种激活函数在某些深度学习模型中可以提供更好的性能。
    '''
    @staticmethod
    def forward(x):
        '''
        前向传播函数

        参数:
        x (torch.Tensor): 输入张量

        返回:
        torch.Tensor: 经过SiLU激活后的输出张量
        '''
        # 计算sigmoid(x)并与输入张量x相乘，得到激活后的输出张量
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    '''A combination of Conv + BN + Activation'''
    '''
    ConvModule类是一个组合了卷积层(Conv)、批归一化层(BatchNorm)和激活函数(Activation)的模块。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核大小。
        stride (int): 步长。
        activation_type (str): 激活函数的类型。
        padding (int, optional): 填充大小，默认为kernel_size // 2。
        groups (int, optional): 卷积组数，默认为1。
        bias (bool, optional): 是否使用偏置，默认为False。
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        # 如果padding未指定，则默认设置为kernel_size的一半
        if padding is None:
            padding = kernel_size // 2
        # 定义卷积层
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        # 定义批归一化层
        self.bn = nn.BatchNorm2d(out_channels)
        # 如果指定了激活函数类型，则获取对应的激活函数
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        '''
        前向传播函数。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 经过卷积、批归一化和激活后的输出张量。
        '''
        # 如果没有指定激活函数，则直接返回批归一化和卷积的结果
        if self.activation_type is None:
            return self.bn(self.conv(x))
        # 否则，返回激活函数、批归一化和卷积的结果
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        '''
        前向融合传播函数，用于融合卷积和激活函数。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 经过融合后的卷积和激活函数的输出张量。
        '''
        # 如果没有指定激活函数，则直接返回卷积的结果
        if self.activation_type is None:
            return self.conv(x)
        # 否则，返回激活函数和卷积的结果
        return self.act(self.conv(x))


# 定义一个卷积、批归一化和ReLU激活的组合模块
class ConvBNReLU(nn.Module):
    '''Conv and BN with ReLU activation'''
    '''
    Conv和BN（批归一化）以及ReLU激活的组合模块

    参数:
    in_channels (int): 输入通道数
    out_channels (int): 输出通道数
    kernel_size (int): 卷积核大小，默认为3
    stride (int): 步长，默认为1
    padding (int, optional): 填充大小，默认为None，表示使用kernel_size//2进行填充
    groups (int): 分组卷积的组数，默认为1
    bias (bool): 是否使用偏置，默认为False
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        # 创建一个ConvModule实例，该实例包含了卷积、批归一化和ReLU激活
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'relu', padding, groups, bias)

    # 前向传播函数，输入x通过self.block处理后返回结果
    def forward(self, x):
        return self.block(x)


# 定义一个包含卷积层、批归一化层和SiLU激活函数的模块
class ConvBNSiLU(nn.Module):
    '''Conv and BN with SiLU activation'''
    '''
    Conv和BN与SiLU激活的组合模块

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认为3
        stride (int): 步长，默认为1
        padding (int, optional): 填充大小，默认为None
        groups (int): 卷积组数，默认为1
        bias (bool): 是否使用偏置，默认为False
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__() # 调用父类构造函数
        # 创建一个卷积模块，包含卷积层、批归一化层和SiLU激活函数
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'silu', padding, groups, bias)

    def forward(self, x):
        # 将输入传递给卷积模块并返回结果
        return self.block(x)


# 定义一个名为ConvBN的类，继承自nn.Module
class ConvBN(nn.Module):
    '''Conv and BN without activation'''
    '''
    ConvBN类是一个包含卷积层(Convolutional Layer)和批归一化层(Batch Normalization Layer)的模块，
    但不包含激活函数。这个类通常用于深度学习模型的构建中。
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        '''
        初始化ConvBN类。

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小，默认为3
        :param stride: 步长，默认为1
        :param padding: 填充大小，如果为None，则根据kernel_size自动计算
        :param groups: 分组卷积的组数，默认为1
        :param bias: 是否使用偏置，默认为False
        '''
        super().__init__() # 调用父类的初始化方法
        # 创建一个ConvModule实例，包含卷积层和批归一化层
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, None, padding, groups, bias)

    def forward(self, x):
        '''
        定义前向传播过程。

        :param x: 输入数据
        :return: 经过卷积层和批归一化层后的输出数据
        '''
        # 将输入数据传递给self.block进行处理并返回结果
        return self.block(x)


# 定义一个包含卷积、批归一化和硬希夫激活的模块
class ConvBNHS(nn.Module):
    '''Conv and BN with Hardswish activation'''
    '''
    Conv和BN与Hardswish激活的组合

    参数:
    in_channels (int): 输入通道数
    out_channels (int): 输出通道数
    kernel_size (int): 卷积核大小，默认为3
    stride (int): 步长，默认为1
    padding (int or tuple): 填充大小，如果为None则根据kernel_size自动计算，默认为None
    groups (int): 分组卷积的组数，默认为1
    bias (bool): 是否使用偏置，默认为False
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__() # 调用父类的构造函数
        # 创建一个ConvModule实例，该实例包含了卷积、批归一化和硬希夫激活
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'hardswish', padding, groups, bias)

    def forward(self, x):
        # 将输入x传递给block进行处理，并返回处理后的结果
        return self.block(x)


class SPPFModule(nn.Module):
    """
    SPPFModule 是一个继承自 nn.Module 的自定义神经网络模块。
    它实现了特定的卷积和池化操作，用于特征提取。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int, optional): 池化操作的核大小，默认为5。
        block (callable, optional): 用于构建卷积块的函数，默认为 ConvBNReLU。
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        super().__init__()
        c_ = in_channels // 2  # hidden channels # 隐藏层通道数
        self.cv1 = block(in_channels, c_, 1, 1) # 第一个卷积层
        self.cv2 = block(c_ * 4, out_channels, 1, 1) # 第二个卷积层
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2) # 最大池化层

    def forward(self, x):
        """
        定义前向传播过程。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过模块处理后的输出张量。
        """
        x = self.cv1(x) # 通过第一个卷积层
        with warnings.catch_warnings(): # 忽略警告
            warnings.simplefilter('ignore')
            y1 = self.m(x)      # 第一次池化
            y2 = self.m(y1)     # 第二次池化
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1)) # 拼接结果并通过第二个卷积层


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    '''
    简化版的SPPF（空间金字塔池化特征），使用ReLU激活函数。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核大小，默认为5。
        block (nn.Module): 包含卷积、批归一化和ReLU的模块，默认为ConvBNReLU。
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        super().__init__()                                                    # 调用父类的构造函数
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size, block) # 初始化SPPF模块

    def forward(self, x):
        '''前向传播函数

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 经过SPPF模块处理后的输出张量。
        '''
        # 将输入张量传递给SPPF模块并返回结果
        return self.sppf(x)


# 定义SPPF类，继承自nn.Module
class SPPF(nn.Module):
    '''SPPF with SiLU activation'''
    '''
    SPPF类，带有SiLU激活函数的SPPF模块

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认为5
        block (type): 块类型，默认为ConvBNSiLU
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNSiLU):
        super().__init__()                                                    # 调用父类构造函数
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size, block) # 初始化SPPFModule

    def forward(self, x):
        '''前向传播函数

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 经过SPPF模块处理后的输出张量
        '''
        # 返回SPPF模块的输出
        return self.sppf(x)

# 定义一个名为CSPSPPF的神经网络模块，该模块使用了SiLU激活函数
class CSPSPPFModule(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    CSPSPPFModule 类实现了 Cross Stage Partial Networks (CSP) 的一个模块。
    参考: https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        """
        初始化 CSPSPPFModule。

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 池化操作的核大小，默认为5
        :param e: 隐藏通道的比例因子，默认为0.5
        :param block: 使用的块类型，默认为 ConvBNReLU
        """
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels # 计算隐藏通道数
        self.cv1 = block(in_channels, c_, 1, 1)       # 第一个卷积块
        self.cv2 = block(in_channels, c_, 1, 1)       # 第二个卷积块
        self.cv3 = block(c_, c_, 3, 1)                # 第三个卷积块
        self.cv4 = block(c_, c_, 1, 1)                # 第四个卷积块

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2) # 最大池化层
        self.cv5 = block(4 * c_, c_, 1, 1)            # 第五个卷积块
        self.cv6 = block(c_, c_, 3, 1)                # 第六个卷积块
        self.cv7 = block(2 * c_, out_channels, 1, 1)  # 第七个卷积块

    def forward(self, x):
        """
        此函数定义了前向传播的过程。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过一系列卷积和激活操作后的输出张量。
        """
        # 通过一系列卷积层处理输入x，并返回处理后的结果
        x1 = self.cv4(self.cv3(self.cv1(x)))
        # 对输入x应用另一个卷积层
        y0 = self.cv2(x)
        # 忽略警告信息
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # 应用激活函数m
            y1 = self.m(x1)
            y2 = self.m(y1)
            # 将x1, y1, y2和经过两次激活函数m的y2进行拼接，并通过卷积层处理
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        # 将y0和处理后的y3拼接，并通过最后一个卷积层得到最终输出
        return self.cv7(torch.cat((y0, y3), dim=1))


class SimCSPSPPF(nn.Module):
    '''CSPSPPF with ReLU activation'''
    '''
    CSPSPPF 类，带有 ReLU 激活函数。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核大小，默认为 5。
        e (float): 一个参数，用于 CSPSPPFModule，默认为 0.5。
        block (nn.Module): 一个包含卷积、批归一化和 ReLU 的模块，默认为 ConvBNReLU。
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        super().__init__()                                                             # 调用父类的构造函数
        self.cspsppf = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block) # 初始化 CSPSPPFModule

    def forward(self, x):
        '''前向传播函数

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过 CSPSPPFModule 处理后的输出张量。
        '''
        return self.cspsppf(x) # 将输入传递给 CSPSPPFModule 并返回结果

# 定义一个名为CSPSPPF的神经网络模块，该模块使用SiLU激活函数
class CSPSPPF(nn.Module):
    '''CSPSPPF with SiLU activation'''
    '''
    CSPSPPF模块，带有SiLU激活函数

    参数:
    in_channels (int): 输入通道数
    out_channels (int): 输出通道数
    kernel_size (int): 卷积核大小，默认为5
    e (float): 一个参数，默认为0.5
    block (class): 一个类，用于定义卷积块，默认为ConvBNSiLU
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNSiLU):
        super().__init__() # 调用父类的构造函数
        # 初始化CSPSPPF模块，传入相应的参数
        self.cspsppf = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)

    # 定义前向传播函数
    def forward(self, x):
        # 将输入x传递给CSPSPPF模块，并返回结果
        return self.cspsppf(x)


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    '''
    正常的转置操作，默认用于上采样
    '''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        '''
        初始化函数

        参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认为2
        stride (int): 步长，默认为2
        '''
        super().__init__()
        # 定义一个转置卷积层，用于上采样
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        '''
        前向传播函数

        参数:
        x (Tensor): 输入张量

        返回:
        Tensor: 经过转置卷积层处理后的张量
        '''
        return self.upsample_transpose(x)


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    RepVGGBlock 是一个基本的 rep-style 块，包括训练和部署状态
    该代码基于 https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image                       输入图像中的通道数
            out_channels (int): Number of channels produced by the convolution             卷积产生的通道数
            kernel_size (int or tuple): Size of the convolving kernel                      卷积核的大小
            stride (int or tuple, optional): Stride of the convolution. Default: 1         卷积的步长。默认: 1
            padding (int or tuple, optional): Zero-padding added to both sides of          添加到输入两侧的零填充。默认: 1
                the input. Default: 1                                                      核元素之间的间距。默认: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1 从输入通道到输出通道的阻塞连接数。默认: 1
            groups (int, optional): Number of blocked connections from input               从输入通道到输出通道的阻塞连接数。默认: 1
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'                              默认: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False         默认: False
            use_se: Whether to use se. Default: False                                      默认: False
        """
        self.deploy = deploy                    # 是否为部署状态
        self.groups = groups                    # 连接组数
        self.in_channels = in_channels          # 输入通道数
        self.out_channels = out_channels        # 输出通道数

        assert kernel_size == 3                 # 断言卷积核大小为3
        assert padding == 1                     # 断言填充为1

        padding_11 = padding - kernel_size // 2 # 计算1x1卷积的填充

        self.nonlinearity = nn.ReLU()           # 激活函数

        if use_se:
            raise NotImplementedError("se block not supported yet") # 如果使用se块，抛出未实现错误
        else:
            self.se = nn.Identity()             # 否则使用恒等映射

        if deploy:
            # 如果是部署状态，使用重参数化的卷积
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            # 如果是训练状态，使用批归一化、3x3卷积和1x1卷积
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, activation_type=None, padding=padding, groups=groups)
            self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, activation_type=None, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        '''
        前向传播过程
        如果存在重参数化线性层(rbr_reparam)，则返回经过非线性激活函数处理的输出
        否则，计算并返回经过非线性激活函数处理的密集连接、1x1卷积和恒等映射的输出之和
        '''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        '''
        获取等效的卷积核和偏置
        将密集连接、1x1卷积和恒等映射的分支融合为一个3x3的卷积核和偏置
        '''
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        '''
        将平均池化层转换为等效的3x3卷积核
        '''
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        '''
        将1x1卷积核填充为3x3大小
        如果kernel1x1为None，则返回0
        否则，使用torch.nn.functional.pad进行填充
        '''
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        '''
        融合批归一化层和卷积层，返回等效的卷积核和偏置
        如果branch为None，则返回(0, 0)
        如果branch是ConvModule实例，则直接返回其卷积核和偏置
        如果branch是nn.BatchNorm2d实例，则计算并返回等效的卷积核和偏置
        '''
        # 如果分支为None，则返回0, 0
        if branch is None:
            return 0, 0
        # 如果分支是ConvModule类型，则获取卷积层的权重和偏置
        elif isinstance(branch, ConvModule):
            kernel = branch.conv.weight  # 获取卷积核权重
            bias = branch.conv.bias  # 获取偏置项
            return kernel, bias
        # 如果分支是nn.BatchNorm2d类型，则处理批归一化层的相关参数
        elif isinstance(branch, nn.BatchNorm2d):
            # 如果没有id_tensor属性，则创建一个
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                # 初始化一个全零的张量作为id_tensor
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                # 设置id_tensor的对角线元素为1
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                # 将numpy数组转换为torch张量，并移动到branch.weight所在的设备
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor  # 获取id_tensor作为卷积核
            running_mean = branch.running_mean  # 获取批归一化层的运行均值
            running_var = branch.running_var  # 获取批归一化层的运行方差
            gamma = branch.weight  # 获取批归一化层的权重参数
            beta = branch.bias  # 获取批归一化层的偏置参数
            eps = branch.eps  # 获取批归一化层的epsilon值
            std = (running_var + eps).sqrt()  # 计算标准差
            t = (gamma / std).reshape(-1, 1, 1, 1)  # 计算缩放因子
            # 返回处理后的卷积核和偏置项
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        '''
        切换到部署模式
        如果存在重参数化线性层(rbr_reparam)，则直接返回
        否则，计算等效的卷积核和偏置，并创建新的重参数化线性层
        删除原有的密集连接、1x1卷积和恒等映射分支，以及id_tensor属性
        设置deploy标志为True
        '''
        # 如果对象具有属性'rbr_reparam'，则直接返回，不进行后续操作
        if hasattr(self, 'rbr_reparam'):
            return

        # 获取等效的卷积核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()

        # 创建一个新的卷积层'rbr_reparam'，并设置其参数
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,  # 输入通道数
            out_channels=self.rbr_dense.conv.out_channels,  # 输出通道数
            kernel_size=self.rbr_dense.conv.kernel_size,  # 卷积核大小
            stride=self.rbr_dense.conv.stride,  # 步长
            padding=self.rbr_dense.conv.padding,  # 填充大小
            dilation=self.rbr_dense.conv.dilation,  # 膨胀率
            groups=self.rbr_dense.conv.groups,  # 组数
            bias=True  # 是否使用偏置
        )

        # 将获取到的卷积核和偏置赋值给新创建的卷积层
        self.rbr_reparam.weight.data = kernel
        self.rbr_redaram.bias.data = bias

        # 遍历所有参数并分离它们，防止反向传播时更新这些参数
        for para in self.parameters():
            para.detach_()

        # 删除原有的属性
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')

        # 如果存在其他特定属性，也一并删除
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

        # 设置部署标志为True，表示模型已转换为部署模式
        self.deploy = True


class QARepVGGBlock(RepVGGBlock):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    QARepVGGBlock是基于RepVGGBlock的一个扩展，增加了量化感知训练（QAT）的支持。
    这个类继承自RepVGGBlock，并在其基础上添加了量化相关的操作。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        """
        初始化QARepVGGBlock。

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding: 填充大小
        :param dilation: 膨胀率
        :param groups: 分组卷积的组数
        :param padding_mode: 填充模式
        :param deploy: 是否为部署模式
        :param use_se: 是否使用Squeeze-and-Excitation模块
        """
        super(QARepVGGBlock, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se)
        # 初始化BatchNorm和1x1卷积层
        if not deploy:
            # 初始化一个2D批归一化层，参数为输出通道数
            self.bn = nn.BatchNorm2d(out_channels)
            # 初始化一个1x1卷积层，参数包括输入通道数、输出通道数、卷积核大小、步长、分组数以及是否使用偏置
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            # 如果输出通道数等于输入通道数且步长为1，则初始化一个恒等映射层，否则设置为None
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        # 初始化一个私有变量_id_tensor，用于存储张量数据
        self._id_tensor = None

    def forward(self, inputs):
        """
        前向传播。

        :param inputs: 输入数据
        :return: 前向传播的结果
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.bn(self.se(self.rbr_reparam(inputs))))

        # 如果'rbr_identity'属性不存在，则将'id_out'设置为0
        # 如果'rbr_identity'属性存在，则对输入数据应用'rbr_identity'函数，并将结果赋值给'id_out'
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def get_equivalent_kernel_bias(self):
        """
        获取等效的卷积核和偏置。

        :return: 等效的卷积核和偏置
        """
        # 融合批归一化张量，获取3x3卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # 将1x1卷积核填充到3x3大小，并与3x3卷积核相加
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3  # 偏置保持不变

        # 如果存在恒等映射分支
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            # 初始化一个全零的卷积核张量
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            # 设置恒等映射卷积核的值
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            # 将numpy数组转换为torch张量，并移动到与1x1卷积核相同的设备上
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            # 将恒等映射卷积核与之前的卷积核相加
            kernel = kernel + id_tensor
        return kernel, bias  # 返回等效的卷积核和偏置

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        """
        融合额外的BatchNorm张量。

        :param kernel: 卷积核，通常是一个四维张量，形状为(out_channels, in_channels, kernel_height, kernel_width)
        :param bias: 偏置，一个一维张量，长度为out_channels
        :param branch: BatchNorm层，一个nn.BatchNorm2d实例，包含BatchNorm的参数
        :return: 融合后的卷积核和偏置，返回一个元组(kernel, bias)，其中kernel是融合了BatchNorm权重的卷积核，bias是新的偏置项

        该函数的作用是将卷积层的输出与BatchNorm层进行融合，以减少计算量和内存占用。
        """
        # 确保branch是nn.BatchNorm2d的实例
        assert isinstance(branch, nn.BatchNorm2d)

        # 移除偏置：BatchNorm的running_mean减去卷积层的bias
        running_mean = branch.running_mean - bias

        # 获取BatchNorm层的参数
        running_var = branch.running_var  # 方差
        gamma = branch.weight  # 权重
        beta = branch.bias  # 偏置
        eps = branch.eps  # 防止除零的小常数

        # 计算标准差
        std = (running_var + eps).sqrt()

        # 计算新的卷积核，将BatchNorm的权重gamma除以标准差std，并调整形状以匹配卷积核
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # 返回融合后的卷积核和偏置
        # 新的卷积核是原卷积核乘以t
        # 新的偏置是BatchNorm层的beta减去running_mean乘以gamma再除以std
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
        切换到部署模式。

        该函数用于将模型切换到部署模式，主要步骤包括：
        1. 检查是否存在 rbr_reparam 属性，如果存在则直接返回。
        2. 获取等效的卷积核和偏置。
        3. 创建一个新的 nn.Conv2d 层，并将获取到的卷积核和偏置赋值给该层。
        4. 断开所有参数的梯度计算。
        5. 删除原有的 rbr_dense, rbr_1x1 属性，以及可能存在的 rbr_identity 和 id_tensor 属性。
        6. 设置 deploy 属性为 True，表示模型已切换到部署模式。
        """
        # 检查是否存在 rbr_reparam 属性，如果存在则直接返回
        if hasattr(self, 'rbr_reparam'):
            return

        # 获取等效的卷积核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()

        # 创建一个新的 nn.Conv2d 层，并将获取到的卷积核和偏置赋值给该层
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # 断开所有参数的梯度计算
        for para in self.parameters():
            para.detach_()

        # 删除原有的 rbr_dense, rbr_1x1 属性
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        # keep post bn for QAT
        # if hasattr(self, 'bn'):
        #     self.__delattr__('bn')
        # 删除可能存在的 rbr_identity 和 id_tensor 属性
        self.deploy = True


class QARepVGGBlockV2(RepVGGBlock):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    QARepVGGBlockV2类继承自RepVGGBlock，用于实现一个基本的rep-style块，支持训练和部署状态。

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认为3
        stride (int): 步长，默认为1
        padding (int): 填充大小，默认为1
        dilation (int): 膨胀率，默认为1
        groups (int): 分组卷积的组数，默认为1
        padding_mode (str): 填充模式，默认为'zeros'
        deploy (bool): 是否为部署模式，默认为False
        use_se (bool): 是否使用Squeeze-and-Excitation，默认为False
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(QARepVGGBlockV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se)
        # 初始化其他属性
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.rbr_avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding) if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        """
        前向传播方法，定义数据通过网络的流程。

        参数:
            inputs (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        # 前向传播逻辑
        # 如果对象具有属性'rbr_reparam'，则执行以下操作：
        # 1. 对输入数据应用'rbr_reparam'方法
        # 2. 对结果应用'se'方法
        # 3. 对上一步的结果应用'bn'方法
        # 4. 对最终结果应用'nonlinearity'方法并返回
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.bn(self.se(self.rbr_reparam(inputs))))

        # 如果'rbr_identity'属性不存在，则将'id_out'设置为0
        # 否则，对输入数据应用'rbr_identity'方法并将结果赋值给'id_out'
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        # 如果'rbr_avg'属性不存在，则将'avg_out'设置为0
        # 否则，对输入数据应用'rbr_avg'方法并将结果赋值给'avg_out'
        if self.rbr_avg is None:
            avg_out = 0
        else:
            avg_out = self.rbr_avg(inputs)

        # 对输入数据依次应用'rbr_dense'、'rbr_1x1'方法，并加上'id_out'和'avg_out'
        # 然后对结果应用'se'、'bn'方法，最后应用'nonlinearity'方法并返回
        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out + avg_out)))

    def get_equivalent_kernel_bias(self):
        """
        计算等效的卷积核和偏置项。

        该方法用于将不同类型的卷积操作（如3x3卷积、1x1卷积、平均池化等）融合为一个等效的3x3卷积操作，
        并计算出相应的卷积核和偏置项。

        Returns:
            tuple: 包含两个元素的元组，第一个元素是等效的卷积核，第二个元素是等效的偏置项。
        """
        # 融合批归一化层到3x3卷积层
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # 将1x1卷积核扩展到3x3大小并加到3x3卷积核上
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)

        # 如果存在平均池化层，将其转换为3x3大小并加到卷积核上
        if self.rbr_avg is not None:
            kernelavg = self._avg_to_3x3_tensor(self.rbr_avg)
            kernel = kernel + kernelavg.to(self.rbr_1x1.weight.device)

        # 偏置项直接取自融合后的3x3卷积层
        bias = bias3x3

        # 如果存在恒等映射路径，计算其等效卷积核并加到总卷积核上
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        # 返回等效的卷积核和偏置项
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        """
        将卷积层和批归一化层的权重融合。

        :param kernel: 卷积层的权重张量
        :param bias: 卷积层的偏置张量
        :param branch: 批归一化层对象
        :return: 融合后的卷积层权重张量和偏置张量
        """
        # 确保branch是nn.BatchNorm2d的实例
        assert isinstance(branch, nn.BatchNorm2d)

        # 移除偏置的影响
        running_mean = branch.running_mean - bias
        # 获取批归一化层的参数
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        # 计算标准差
        std = (running_var + eps).sqrt()
        # 计算新的权重张量
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 返回融合后的卷积层权重和偏置
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        # 如果已经融合过，则直接返回
        if hasattr(self, 'rbr_reparam'):
            return

        # 获取等效的卷积层权重和偏置
        kernel, bias = self.get_equivalent_kernel_bias()

        # 创建新的卷积层，用于部署模式
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size,
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding,
                                     dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups,
                                     bias=True)
        # 设置新卷积层的权重和偏置
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # 断开所有参数的梯度
        for para in self.parameters():
            para.detach_()

        # 删除不再需要的层和参数
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'rbr_avg'):
            self.__delattr__('rbr_avg')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

        # 保留QAT后的批归一化层（可选）
        # if hasattr(self, 'bn'):
        #     self.__delattr__('bn')

        # 标记为部署模式
        self.deploy = True


class RealVGGBlock(nn.Module):
    """
    实现了一个VGG块的类，包含卷积、批归一化和激活函数。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int, optional): 卷积核大小，默认为3。
        stride (int, optional): 卷积步长，默认为1。
        padding (int, optional): 填充大小，默认为1。
        dilation (int, optional): 膨胀率，默认为1。
        groups (int, optional): 卷积分组数，默认为1。
        padding_mode (str, optional): 填充模式，默认为'zeros'。
        use_se (bool, optional): 是否使用Squeeze-and-Excitation模块，默认为False。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False,
    ):
        super(RealVGGBlock, self).__init__()
        self.relu = nn.ReLU() # 激活函数
        # 卷积层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # 批归一化层
        self.bn = nn.BatchNorm2d(out_channels)

        if use_se:
            raise NotImplementedError("se block not supported yet")  # 如果使用Squeeze-and-Excitation模块，则抛出未实现错误
        else:
            self.se = nn.Identity()  # 否则使用恒等映射

    def forward(self, inputs):
        """
        该函数定义了一个前向传播过程，它接收输入数据并经过一系列的神经网络层处理后返回输出结果。

        Args:
            inputs (Tensor): 输入数据张量。

        Returns:
            Tensor: 经过神经网络层处理后的输出数据张量。
        """
        out = self.relu(self.se(self.bn(self.conv(inputs))))
        return out


class ScaleLayer(torch.nn.Module):

    def __init__(self, num_features, use_bias=True, scale_init=1.0):
        """
        初始化ScaleLayer模块。

        :param num_features: 输入特征的数量。
        :param use_bias: 是否使用偏置项，默认为True。
        :param scale_init: 权重的初始值，默认为1.0。
        """
        super(ScaleLayer, self).__init__()
        # 初始化权重参数
        self.weight = Parameter(torch.Tensor(num_features))
        init.constant_(self.weight, scale_init)
        self.num_features = num_features
        # 如果使用偏置项，则初始化偏置参数，否则设置为None
        if use_bias:
            self.bias = Parameter(torch.Tensor(num_features))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        # 如果没有偏置项，则只进行缩放操作
        if self.bias is None:
            return inputs * self.weight.view(1, self.num_features, 1, 1)
        else:
            # 如果有偏置项，则进行缩放并添加偏置
            return inputs * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)


#   A CSLA block is a LinearAddBlock with is_csla=True
# 定义一个线性加法块类，继承自nn.Module，用于实现线性加法块。
class LinearAddBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False, is_csla=False, conv_scale_init=1.0):
        super(LinearAddBlock, self).__init__()
        """
        初始化函数，设置各个层的参数。

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding: 填充大小
        :param dilation: 膨胀率
        :param groups: 分组卷积的组数
        :param padding_mode: 填充模式
        :param use_se: 是否使用SE模块
        :param is_csla: 是否为CSLA模式
        :param conv_scale_init: 卷积层缩放因子的初始值
        """
        super(LinearAddBlock, self).__init__()  # 调用父类构造函数
        self.in_channels = in_channels  # 输入通道数
        self.relu = nn.ReLU()  # 激活函数层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)  # 主卷积层
        self.scale_conv = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)  # 卷积层后的缩放层
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)  # 1x1卷积层
        self.scale_1x1 = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)  # 1x1卷积层后的缩放层
        if in_channels == out_channels and stride == 1:  # 如果输入输出通道数相同且步长为1，则添加一个恒等缩放层
            self.scale_identity = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=1.0)
        self.bn = nn.BatchNorm2d(out_channels)  # 批量归一化层
        if is_csla:  # 如果是CSLA模式，则将1x1卷积层和主卷积层的缩放因子设置为不可训练
            self.scale_1x1.requires_grad_(False)
            self.scale_conv.requires_grad_(False)
        if use_se:  # 如果使用SE模块，则抛出未实现错误
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()  # 否则使用恒等映射作为SE模块

    def forward(self, inputs):
        """
        前向传播函数。

        :param inputs: 输入张量
        :return: 输出张量
        """
        out = self.scale_conv(self.conv(inputs)) + self.scale_1x1(self.conv_1x1(inputs))  # 主卷积和1x1卷积的输出分别经过缩放后相加
        if hasattr(self, 'scale_identity'):  # 如果存在恒等缩放层，则将其输出也加入结果中
            out += self.scale_identity(inputs)
        out = self.relu(self.se(self.bn(out)))  # 经过SE模块、批量归一化层和激活函数层后输出
        return out


class DetectBackend(nn.Module):
    """
    DetectBackend 类是一个用于目标检测的后端模型。
    它加载预训练的权重文件，并在前向传播时输出检测结果。
    """
    def __init__(self, weights='yolov6s.pt', device=None, dnn=True):
        """
        初始化 DetectBackend 类。

        :param weights: 预训练权重文件的路径，默认为 'yolov6s.pt'。
        :param device: 指定模型运行的设备，默认为 None，表示使用 CPU。
        :param dnn: 是否使用 DNN 模式，默认为 True。
        """
        super().__init__()
        # 如果权重文件不存在，则尝试从 GitHub 自动下载
        if not os.path.exists(weights):
            download_ckpt(weights)
        # 断言权重文件路径为字符串且后缀为 '.pt'
        assert isinstance(weights, str) and Path(weights).suffix == '.pt', f'{Path(weights).suffix} 格式不支持。'
        from yolov6.utils.checkpoint import load_checkpoint
        # 加载权重文件
        model = load_checkpoint(weights, map_location=device)
        # 获取模型的最大步幅
        stride = int(model.stride.max())
        # 将所有变量赋值给 self
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        y, _ = self.model(im)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y


class RepBlock(nn.Module):
    '''
    RepBlock is a stage block with rep-style basic block'
    RepBlock 是一个具有 rep-style 基础块的阶段块
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        # 第一个卷积层
        self.conv1 = block(in_channels, out_channels)

        # 如果 n 大于1，则创建一个由 n-1 个 block 组成的序列
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

        # 如果 block 是 BottleRep，则重新定义 self.conv1 和 self.block
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in
                  range(n - 1))) if n > 1 else None

    def forward(self, x):
        '''
        前向传播函数

        :param x: 输入数据
        :return: 处理后的数据
        '''
        x = self.conv1(x)  # 通过第一个卷积层
        if self.block is not None:  # 如果有额外的块，则通过这些块
            x = self.block(x)
        return x  # 返回处理后的数据


class BottleRep(nn.Module):
    """
    BottleRep类是一个自定义的神经网络模块，继承自nn.Module。
    它包含两个卷积层和一个可选的快捷连接（shortcut）。
    如果输入通道数和输出通道数不同，则不使用快捷连接。
    如果weight参数为True，则引入一个可学习的参数alpha。
    """
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        """
        初始化BottleRep类。

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param basic_block: 使用的基本卷积块类型，默认为RepVGGBlock
        :param weight: 是否引入可学习的参数alpha，默认为False
        """
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)  # 第一个卷积层
        self.conv2 = basic_block(out_channels, out_channels)  # 第二个卷积层
        if in_channels != out_channels:
            self.shortcut = False  # 如果输入输出通道数不同，则不使用快捷连接
        else:
            self.shortcut = True  # 否则使用快捷连接
        if weight:
            self.alpha = Parameter(torch.ones(1))  # 引入可学习的参数alpha
        else:
            self.alpha = 1.0  # 否则alpha为常数1.0

    def forward(self, x):
        """
        定义前向传播过程。

        :param x: 输入张量
        :return: 经过网络层处理后的输出张量
        """
        outputs = self.conv1(x)  # 通过第一个卷积层
        outputs = self.conv2(outputs)  # 通过第二个卷积层
        # 如果存在快捷连接，则将输入x乘以alpha后与卷积结果相加；否则直接返回卷积结果
        return outputs + self.alpha * x if self.shortcut else outputs


class BottleRep3(nn.Module):
    """
    BottleRep3类是一个继承自nn.Module的神经网络模块，用于构建特定的网络结构。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        basic_block (type): 基本块的类型，默认为RepVGGBlock。
        weight (bool): 是否使用权重参数。
    """
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        # 定义三个卷积层
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        self.conv3 = basic_block(out_channels, out_channels)

        # 如果输入通道数不等于输出通道数，则不使用shortcut连接
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True

        # 如果使用权重参数，则初始化alpha为Parameter类型，否则为常数1.0
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        """
        前向传播函数，定义了数据通过网络层的流动方式。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过网络层处理后的输出张量。
        """
        # 依次通过三个卷积层，然后将结果相加，最后加上alpha乘以输入x
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        # 如果存在shortcut连接，则将输入张量与输出张量相加，否则直接返回输出张量
        return outputs + self.alpha * x if self.shortcut else outputs


class BepC3(nn.Module):
    '''CSPStackRep Block'''
    '''
    CSPStackRep Block
    该类定义了一个CSPStackRep模块，它是一种用于深度学习模型中的卷积神经网络块。
    '''
    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        '''
        初始化BepC3模块。

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param n: RepBlock中的重复次数
        :param e: 隐藏通道的比例因子
        :param block: 基本块的类型，默认为RepVGGBlock
        '''
        super().__init__()
        c_ = int(out_channels * e)  # 计算隐藏通道数
        # 定义三个卷积层，分别具有不同的输入和输出通道数
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNReLU(2 * c_, out_channels, 1, 1)
        # 如果block是ConvBNSiLU，则替换上述卷积层为ConvBNSiLU
        if block == ConvBNSiLU:
            self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv2 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv3 = ConvBNSiLU(2 * c_, out_channels, 1, 1)

        # 定义一个RepBlock，用于进一步处理特征图
        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)

    def forward(self, x):
        '''
        定义前向传播过程。

        :param x: 输入特征图
        :return: 处理后的特征图
        '''
        # 将通过RepBlock处理后的特征图与另一个分支的特征图拼接，并通过最后一个卷积层
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class MBLABlock(nn.Module):
    ''' Multi Branch Layer Aggregation Block'''
    ''' 多分支层聚合块
    该类定义了一个多分支的神经网络层，用于聚合多个分支的输出。
    '''
    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        '''
        初始化MBLABlock类

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param n: 分支数量，默认为1
        :param e: 隐藏层通道数的缩放因子，默认为0.5
        :param block: 基本块的类型，默认为RepVGGBlock
        '''

        super().__init__()
        n = n // 2  # 将分支数量减半
        if n <= 0:
            n = 1  # 如果分支数量小于等于0，则设置为1

        # 最多增加一个分支
        if n == 1:
            n_list = [0, 1]
        else:
            extra_branch_steps = 1
            # 计算额外的分支步数
            while extra_branch_steps * 2 < n:
                extra_branch_steps *= 2
            n_list = [0, extra_branch_steps, n]
        branch_num = len(n_list)  # 分支数量

        c_ = int(out_channels * e)  # 隐藏层通道数
        self.c = c_
        # 第一个卷积层，将输入通道数转换为分支数乘以隐藏层通道数
        self.cv1 = ConvModule(in_channels, branch_num * self.c, 1, 1, 'relu', bias=False)
        # 第二个卷积层，将所有分支的输出合并后转换为输出通道数
        self.cv2 = ConvModule((sum(n_list) + branch_num) * self.c, out_channels, 1, 1, 'relu', bias=False)

        # 如果基本块类型为ConvBNSiLU，则使用silu激活函数
        if block == ConvBNSiLU:
            self.cv1 = ConvModule(in_channels, branch_num * self.c, 1, 1, 'silu', bias=False)
            self.cv2 = ConvModule((sum(n_list) + branch_num) * self.c, out_channels, 1, 1, 'silu', bias=False)

        self.m = nn.ModuleList()  # 存储所有分支的模块列表
        for n_list_i in n_list[1:]:
            # 为每个分支添加一系列的BottleRep3模块
            self.m.append(
                nn.Sequential(*(BottleRep3(self.c, self.c, basic_block=block, weight=True) for _ in range(n_list_i))))

        self.split_num = tuple([self.c] * branch_num)  # 分割数量元组

    def forward(self, x):
        '''
        前向传播函数

        :param x: 输入张量
        :return: 输出张量
        '''
        y = list(self.cv1(x).split(self.split_num, 1))  # 通过第一个卷积层并分割结果
        all_y = [y[0]]  # 初始化所有分支的输出列表
        for m_idx, m_i in enumerate(self.m):
            all_y.append(y[m_idx + 1])  # 添加当前分支的输入到输出列表
            # 将当前分支的所有模块的输出添加到输出列表
            all_y.extend(m(all_y[-1]) for m in m_i)
        return self.cv2(torch.cat(all_y, 1))  # 通过第二个卷积层并返回结果


class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''
    '''
    PAN中的BiFusion模块
    该模块通过卷积和上采样/下采样操作来融合不同输入通道的特征。
    '''
    def __init__(self, in_channels, out_channels):
        '''
        初始化BiFusion模块

        :param in_channels: 输入通道的列表，表示每个输入特征图的通道数
        :param out_channels: 输出通道数，表示融合后特征图的通道数
        '''
        super().__init__()
        # 第一个卷积层，用于处理第一个输入特征图
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        # 第二个卷积层，用于处理第二个输入特征图
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        # 第三个卷积层，用于融合三个特征图
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        # 上采样层，用于将第一个输入特征图上采样到与融合后的特征图相同的尺寸
        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        # 下采样层，用于将第二个输入特征图下采样到与融合后的特征图相同的尺寸
        self.downsample = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        '''
        前向传播函数

        :param x: 输入特征图的列表，包含三个特征图
        :return: 融合后的特征图
        '''
        # 对第一个输入特征图进行上采样
        x0 = self.upsample(x[0])
        # 对第二个输入特征图进行卷积操作
        x1 = self.cv1(x[1])
        # 对第二个输入特征图进行卷积和下采样操作
        x2 = self.downsample(self.cv2(x[2]))
        # 将三个特征图在通道维度上拼接，并进行卷积操作得到最终的融合特征图
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


def get_block(mode):
    """
    根据传入的模式返回对应的Repblock类。

    参数:
    mode (str): 模式名称，决定了返回哪个Repblock类。

    返回:
    class: 对应模式的Repblock类。

    异常:
    NotImplementedError: 如果传入的模式未定义，则抛出此异常。
    """
    if mode == 'repvgg':
        return RepVGGBlock  # 返回RepVGGBlock类
    elif mode == 'qarepvgg':
        return QARepVGGBlock  # 返回QARepVGGBlock类
    elif mode == 'qarepvggv2':
        return QARepVGGBlockV2  # 返回QARepVGGBlockV2类
    elif mode == 'hyper_search':
        return LinearAddBlock  # 返回LinearAddBlock类
    elif mode == 'repopt':
        return RealVGGBlock  # 返回RealVGGBlock类
    elif mode == 'conv_relu':
        return ConvBNReLU  # 返回ConvBNReLU类
    elif mode == 'conv_silu':
        return ConvBNSiLU  # 返回ConvBNSiLU类
    else:
        raise NotImplementedError("Undefied Repblock choice for mode {}".format(mode))  # 抛出未定义模式的异常


class SEBlock(nn.Module):
    """
    SEBlock 类是一个简单的自编码块，用于在卷积神经网络中进行特征图的通道维度上的自适应缩放。
    该模块通过全局平均池化来获取每个通道的全局信息，并通过两个卷积层来调整这些信息。
    最后，使用 hardsigmoid 函数将调整后的信息缩放到 [0, 1] 范围内，并与原始输入相乘。

    Args:
        channel (int): 输入特征图的通道数。
        reduction (int): 降维因子，用于减少通道数以计算通道注意力。
    """

    def __init__(self, channel, reduction=4):
        super().__init__()
        # 全局平均池化层，用于获取每个通道的全局信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 第一个卷积层，用于降维
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        # ReLU 激活函数
        self.relu = nn.ReLU()
        # 第二个卷积层，用于升维回原始通道数
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        # hardsigmoid 函数，用于将输出缩放到 [0, 1] 范围内
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        # 保存原始输入
        identity = x
        # 全局平均池化
        x = self.avg_pool(x)
        # 第一个卷积层
        x = self.conv1(x)
        # ReLU 激活
        x = self.relu(x)
        # 第二个卷积层
        x = self.conv2(x)
        # hardsigmoid 函数
        x = self.hardsigmoid(x)
        # 将调整后的信息与原始输入相乘
        out = identity * x
        return out


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """
    对输入张量x进行通道混洗。

    Args:
        x (Tensor): 输入的张量，形状为(batchsize, num_channels, height, width)。
        groups (int): 要分成的组数。

    Returns:
        Tensor: 混洗后的张量，形状为(batchsize, -1, height, width)。

    """
    # 获取输入张量的各个维度大小
    batchsize, num_channels, height, width = x.data.size()

    # 计算每组包含的通道数
    channels_per_group = num_channels // groups

    # 将输入张量重塑为(batchsize, groups, channels_per_group, height, width)的形状
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # 对张量的维度进行转置，将groups和channels_per_group两个维度互换
    x = torch.transpose(x, 1, 2).contiguous()

    # 将张量重塑为(batchsize, -1, height, width)的形状，-1表示自动计算该维度的大小
    x = x.view(batchsize, -1, height, width)

    return x  # 返回混洗后的张量


class Lite_EffiBlockS1(nn.Module):
    """
    Lite_EffiBlockS1是一个轻量级的神经网络模块，用于执行卷积和通道混洗操作。

    Args:
        in_channels (int): 输入通道数。
        mid_channels (int): 中间通道数。
        out_channels (int): 输出通道数。
        stride (int): 卷积步长。
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride):
        super().__init__()
        # 定义第一个1x1点卷积层
        self.conv_pw_1 = ConvBNHS(
            in_channels=in_channels // 2,  # 输入通道数为输入总通道数的一半
            out_channels=mid_channels,  # 输出通道数为中间通道数
            kernel_size=1,  # 卷积核大小为1x1
            stride=1,  # 步长为1
            padding=0,  # 填充为0
            groups=1)  # 分组卷积的组数为1
        # 定义第一个深度可分离卷积层
        self.conv_dw_1 = ConvBN(
            in_channels=mid_channels,  # 输入通道数为中间通道数
            out_channels=mid_channels,  # 输出通道数为中间通道数
            kernel_size=3,  # 卷积核大小为3x3
            stride=stride,  # 步长为用户定义的步长
            padding=1,  # 填充为1以保持特征图大小不变
            groups=mid_channels)  # 分组卷积的组数为中间通道数，实现深度可分离卷积
        # 定义一个SEBlock模块，用于通道注意力机制
        self.se = SEBlock(mid_channels)
        # 定义第二个1x1点卷积层
        self.conv_1 = ConvBNHS(
            in_channels=mid_channels,  # 输入通道数为中间通道数
            out_channels=out_channels // 2,  # 输出通道数为输出总通道数的一半
            kernel_size=1,  # 卷积核大小为1x1
            stride=1,  # 步长为1
            padding=0,  # 填充为0
            groups=1)  # 分组卷积的组数为1

    def forward(self, inputs):
        """
        前向传播函数
        :param inputs: 输入张量
        :return: 处理后的输出张量
        """
        # 将输入张量沿着维度1平均分成两部分
        x1, x2 = torch.split(
            inputs,
            split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            dim=1)

        # 对x2进行卷积操作
        x2 = self.conv_pw_1(x2)

        # 对x2进行深度可分离卷积
        x3 = self.conv_dw_1(x2)

        # 应用通道注意力机制
        x3 = self.se(x3)

        # 对x3进行卷积操作
        x3 = self.conv_1(x3)

        # 将x1和处理后的x3在维度1上拼接
        out = torch.cat([x1, x3], axis=1)

        # 对拼接后的张量进行通道混洗
        return channel_shuffle(out, 2)

# class Lite_EffiBlockS2(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  mid_channels,
#                  out_channels,
#                  stride):
#         super().__init__()
#         # branch1
#         self.conv_dw_1 = ConvBN(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             groups=in_channels)
#         self.conv_1 = ConvBNHS(
#             in_channels=in_channels,
#             out_channels=out_channels // 2,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1)
#         # branch2
#         self.conv_pw_2 = ConvBNHS(
#             in_channels=in_channels,
#             out_channels=mid_channels // 2,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1)
#         self.conv_dw_2 = ConvBN(
#             in_channels=mid_channels // 2,
#             out_channels=mid_channels // 2,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             groups=mid_channels // 2)
#         self.se = SEBlock(mid_channels // 2)
#         self.conv_2 = ConvBNHS(
#             in_channels=mid_channels // 2,
#             out_channels=out_channels // 2,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1)
#         self.conv_dw_3 = ConvBNHS(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             groups=out_channels)
#         self.conv_pw_3 = ConvBNHS(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1)
#
#     def forward(self, inputs):
#         x1 = self.conv_dw_1(inputs)
#         x1 = self.conv_1(x1)
#         x2 = self.conv_pw_2(inputs)
#         x2 = self.conv_dw_2(x2)
#         x2 = self.se(x2)
#         x2 = self.conv_2(x2)
#         out = torch.cat([x1, x2], axis=1)
#         out = self.conv_dw_3(out)
#         out = self.conv_pw_3(out)
#         return out

class Lite_EffiBlockS2(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride):
        """
        初始化Lite_EffiBlockS2模块。

        :param in_channels: 输入通道数
        :param mid_channels: 中间通道数
        :param out_channels: 输出通道数
        :param stride: 步长
        """
        super().__init__()
        # branch1
        # 深度可分离卷积，输入通道数等于输出通道数，用于降低计算量
        self.conv_dw_1 = ConvBN(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels)
        # 1x1卷积，用于改变通道数
        self.conv_1 = ConvBNHS(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        # branch2
        # 1x1卷积，用于降低通道数
        self.conv_pw_2 = ConvBNHS(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        # 深度可分离卷积，用于进一步降低计算量
        self.conv_dw_2 = ConvBN(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2)
        # SEBlock，用于通道注意力机制
        self.se = SEBlock(mid_channels // 2)
        # 1x1卷积，用于改变通道数
        self.conv_2 = ConvBNHS(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        # 深度可分离卷积，用于增加通道数的同时保持计算量低
        self.conv_dw_3 = ConvBNHS(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels)
        # 1x1卷积，用于改变通道数
        self.conv_pw_3 = ConvBNHS(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)

    def forward(self, inputs):
        """
        前向传播函数。

        :param inputs: 输入张量
        :return: 输出张量
        """
        x1 = self.conv_dw_1(inputs)  # 第一个分支的深度可分离卷积
        x1 = self.conv_1(x1)  # 第一个分支的1x1卷积
        x2 = self.conv_pw_2(inputs)  # 第二个分支的1x1卷积
        x2 = self.conv_dw_2(x2)  # 第二个分支的深度可分离卷积
        x2 = self.se(x2)  # 第二个分支的通道注意力机制
        x2 = self.conv_2(x2)  # 第二个分支的1x1卷积
        out = torch.cat([x1, x2], axis=1)  # 将两个分支的结果拼接
        out = self.conv_dw_3(out)  # 深度可分离卷积，用于增加通道数
        out = self.conv_pw_3(out)  # 1x1卷积，用于最终输出
        return out

class DPBlock(nn.Module):

    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1):
        """
        初始化DPBlock模块。

        :param in_channel: 输入通道数，默认为96
        :param out_channel: 输出通道数，默认为96
        :param kernel_size: 卷积核大小，默认为3
        :param stride: 步长，默认为1
        """
        super().__init__()
        # 深度卷积层，使用当前输出通道数作为组数，实现深度可分离卷积
        self.conv_dw_1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride)
        # 批量归一化层
        self.bn_1 = nn.BatchNorm2d(out_channel)
        # 激活函数层，使用Hardswish激活函数
        self.act_1 = nn.Hardswish()
        # 逐点卷积层，用于调整通道数
        self.conv_pw_1 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0)
        # 第二个批量归一化层
        self.bn_2 = nn.BatchNorm2d(out_channel)
        # 第二个激活函数层
        self.act_2 = nn.Hardswish()

    def forward(self, x):
        """
        前向传播函数。

        :param x: 输入张量
        :return: 处理后的输出张量
        """
        x = self.act_1(self.bn_1(self.conv_dw_1(x)))  # 深度卷积 + 批量归一化 + 激活
        x = self.act_2(self.bn_2(self.conv_pw_1(x)))  # 逐点卷积 + 批量归一化 + 激活
        return x

    def forward_fuse(self, x):
        """
        融合前向传播函数，跳过批量归一化层以加速计算。

        :param x: 输入张量
        :return: 处理后的输出张量
        """
        x = self.act_1(self.conv_dw_1(x))  # 深度卷积 + 激活
        x = self.act_2(self.conv_pw_1(x))  # 逐点卷积 + 激活
        return x


# class DarknetBlock(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  expansion=0.5):
#         super().__init__()
#         hidden_channels = int(out_channels * expansion)
#         self.conv_1 = ConvBNHS(
#             in_channels=in_channels,
#             out_channels=hidden_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0)
#         self.conv_2 = DPBlock(
#             in_channel=hidden_channels,
#             out_channel=out_channels,
#             kernel_size=kernel_size,
#             stride=1)
#
#     def forward(self, x):
#         out = self.conv_1(x)
#         out = self.conv_2(out)
#         return out
# 定义一个DarknetBlock类，继承自nn.Module
class DarknetBlock(nn.Module):

    # 初始化方法，定义了DarknetBlock的参数和属性
    def __init__(self,
                 in_channels,  # 输入通道数
                 out_channels,  # 输出通道数
                 kernel_size=3,  # 卷积核大小，默认为3
                 expansion=0.5):  # 扩展因子，默认为0.5
        super().__init__()  # 调用父类的初始化方法
        hidden_channels = int(out_channels * expansion)  # 计算隐藏层的通道数
        # 定义第一个卷积层，输入通道数为in_channels，输出通道数为hidden_channels，卷积核大小为1
        self.conv_1 = ConvBNHS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        # 定义第二个卷积层，输入通道数为hidden_channels，输出通道数为out_channels，卷积核大小为kernel_size
        self.conv_2 = DPBlock(
            in_channel=hidden_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1)
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out


class CSPBlock(nn.Module):
    """
    CSPBlock 类定义了一个卷积块，它使用了两个卷积层和一个 DarknetBlock 来处理输入数据，
    然后将两个分支的结果拼接起来，并通过另一个卷积层输出。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核大小，默认为3。
        expand_ratio (float): 扩展比率，默认为0.5。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expand_ratio=0.5):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)  # 计算中间通道数
        self.conv_1 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)  # 第一个卷积层
        self.conv_2 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)  # 第二个卷积层
        self.conv_3 = ConvBNHS(2 * mid_channels, out_channels, 1, 1, 0)  # 第三个卷积层，用于拼接后的特征图
        self.blocks = DarknetBlock(mid_channels,  # DarknetBlock 用于处理第一个卷积层的输出
                                   mid_channels,
                                   kernel_size,
                                   1.0)

    def forward(self, x):
        """
        forward 方法定义了数据通过 CSPBlock 的流程。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 处理后的输出张量。
        """
        x_1 = self.conv_1(x)  # 通过第一个卷积层
        x_1 = self.blocks(x_1)  # 通过 DarknetBlock
        x_2 = self.conv_2(x)  # 通过第二个卷积层
        x = torch.cat((x_1, x_2), axis=1)  # 拼接两个分支的结果
        x = self.conv_3(x)  # 通过第三个卷积层
        return x  # 返回最终输出