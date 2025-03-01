# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""
# 卷积模块

import math  # 导入数学库

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
)
# 定义模块的公共接口

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    # 填充以保持相同的形状输出
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际的卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p  # 返回填充大小


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    # 标准卷积，参数包括输入通道数、输出通道数、卷积核、步幅、填充、组数、扩张和激活函数

    default_act = nn.SiLU()  # default activation
    # 默认激活函数为SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        # 初始化卷积层，给定参数，包括激活函数
        super().__init__()  # 调用父类构造函数
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 定义卷积层
        self.bn = nn.BatchNorm2d(c2)  # 定义批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 定义激活函数

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # 对输入张量应用卷积、批归一化和激活
        return self.act(self.bn(self.conv(x)))  # 返回经过激活的卷积结果

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        # 在不使用批归一化的情况下应用卷积和激活
        return self.act(self.conv(x))  # 返回经过激活的卷积结果


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""
    # 简化的RepConv模块，具有卷积融合

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        # 初始化卷积层，给定参数，包括激活函数
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)  # 调用父类构造函数
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # 添加1x1卷积层

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # 对输入张量应用卷积、批归一化和激活
        return self.act(self.bn(self.conv(x) + self.cv2(x)))  # 返回经过激活的卷积结果

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        # 对输入张量应用融合卷积、批归一化和激活
        return self.act(self.bn(self.conv(x)))  # 返回经过激活的卷积结果

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        # 融合并行卷积
        w = torch.zeros_like(self.conv.weight.data)  # 创建与卷积权重相同形状的零张量
        i = [x // 2 for x in w.shape[2:]]  # 计算索引
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()  # 将1x1卷积的权重复制到w
        self.conv.weight.data += w  # 将w添加到主卷积的权重
        self.__delattr__("cv2")  # 删除cv2属性
        self.forward = self.forward_fuse  # 将forward方法替换为forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    # 轻量卷积，参数包括输入通道、输出通道和卷积核

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        # 初始化卷积层，给定参数，包括激活函数
        super().__init__()  # 调用父类构造函数
        self.conv1 = Conv(c1, c2, 1, act=False)  # 定义第一个卷积层
        self.conv2 = DWConv(c2, c2, k, act=act)  # 定义深度卷积层

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # 对输入张量应用两个卷积
        return self.conv2(self.conv1(x))  # 返回第二个卷积的结果


class DWConv(Conv):
    """Depth-wise convolution."""
    # 深度卷积

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        # 初始化深度卷积，给定参数
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)  # 调用父类构造函数


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""
    # 深度转置卷积

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        # 初始化DWConvTranspose2d类，给定参数
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))  # 调用父类构造函数


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    # 2D卷积转置层

    default_act = nn.SiLU()  # default activation
    # 默认激活函数为SiLU

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        # 初始化ConvTranspose2d层，带有批归一化和激活函数
        super().__init__()  # 调用父类构造函数
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)  # 定义转置卷积层
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()  # 定义批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 定义激活函数

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        # 对输入应用转置卷积、批归一化和激活
        return self.act(self.bn(self.conv_transpose(x)))  # 返回经过激活的转置卷积结果

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        # 对输入应用激活和转置卷积操作
        return self.act(self.conv_transpose(x))  # 返回经过激活的转置卷积结果


class Focus(nn.Module):
    """Focus wh information into c-space."""
    # 将宽高信息集中到通道空间

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        # 初始化Focus对象，给定用户定义的通道、卷积、填充、组和激活值
        super().__init__()  # 调用父类构造函数
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)  # 定义卷积层
        # self.contract = Contract(gain=2)  # 注释掉的代码

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        # 对拼接的张量应用卷积并返回输出
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))  # 注释掉的代码


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""
    # Ghost卷积

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        # 初始化Ghost卷积模块，具有主要和廉价操作以实现高效特征学习
        super().__init__()  # 调用父类构造函数
        c_ = c2 // 2  # hidden channels
        # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)  # 定义第一个卷积层
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)  # 定义第二个卷积层

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        # 通过Ghost瓶颈层的前向传播，带有跳跃连接
        y = self.cv1(x)  # 通过第一个卷积层处理输入
        return torch.cat((y, self.cv2(y)), 1)  # 返回拼接后的结果


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    # RepConv是一个基本的重复样式块，包括训练和部署状态

    default_act = nn.SiLU()  # default activation
    # 默认激活函数为SiLU

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        # 初始化轻量卷积层，给定输入、输出和可选的激活函数
        super().__init__()  # 调用父类构造函数
        assert k == 3 and p == 1  # 确保卷积核大小为3且填充为1
        self.g = g  # 保存组数
        self.c1 = c1  # 保存输入通道数
        self.c2 = c2  # 保存输出通道数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 定义激活函数

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None  # 定义批归一化层
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)  # 定义第一个卷积层
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)  # 定义第二个卷积层

    def forward_fuse(self, x):
        """Forward process."""
        # 前向传播
        return self.act(self.conv(x))  # 返回经过激活的卷积结果

    def forward(self, x):
        """Forward process."""
        # 前向传播
        id_out = 0 if self.bn is None else self.bn(x)  # 如果没有批归一化，则id_out为0
        return self.act(self.conv1(x) + self.conv2(x) + id_out)  # 返回经过激活的卷积结果

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        # 返回等效的卷积核和偏置，通过添加3x3卷积核、1x1卷积核和恒等卷积核及其偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)  # 获取第一个卷积层的卷积核和偏置
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)  # 获取第二个卷积层的卷积核和偏置
        kernelid, biasid = self._fuse_bn_tensor(self.bn)  # 获取批归一化层的卷积核和偏置
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid  # 返回等效卷积核和偏置

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        # 将1x1张量填充为3x3张量
        if kernel1x1 is None:
            return 0  # 如果卷积核为空，返回0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])  # 对1x1张量进行填充

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        # 通过融合神经网络的分支生成适当的卷积核和偏置
        if branch is None:
            return 0, 0  # 如果分支为空，返回0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight  # 获取卷积核
            running_mean = branch.bn.running_mean  # 获取批归一化的均值
            running_var = branch.bn.running_var  # 获取批归一化的方差
            gamma = branch.bn.weight  # 获取批归一化的权重
            beta = branch.bn.bias  # 获取批归一化的偏置
            eps = branch.bn.eps  # 获取批归一化的epsilon
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g  # 计算输入维度
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)  # 创建卷积核的零张量
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1  # 设置恒等卷积核
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)  # 将零张量转换为PyTorch张量
            kernel = self.id_tensor  # 使用恒等卷积核
            running_mean = branch.running_mean  # 获取批归一化的均值
            running_var = branch.running_var  # 获取批归一化的方差
            gamma = branch.weight  # 获取批归一化的权重
            beta = branch.bias  # 获取批归一化的偏置
            eps = branch.eps  # 获取批归一化的epsilon
        std = (running_var + eps).sqrt()  # 计算标准差
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 计算缩放因子
        return kernel * t, beta - running_mean * gamma / std  # 返回卷积核和偏置

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        # 将两个卷积层合并为一个层，并删除类中未使用的属性
        if hasattr(self, "conv"):
            return  # 如果已经存在卷积层，返回
        kernel, bias = self.get_equivalent_kernel_bias()  # 获取等效卷积核和偏置
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)  # 创建新的卷积层，并设置为不需要梯度
        self.conv.weight.data = kernel  # 设置卷积层的权重
        self.conv.bias.data = bias  # 设置卷积层的偏置
        for para in self.parameters():
            para.detach_()  # 将所有参数分离
        self.__delattr__("conv1")  # 删除conv1属性
        self.__delattr__("conv2")  # 删除conv2属性
        if hasattr(self, "nm"):
            self.__delattr__("nm")  # 删除nm属性
        if hasattr(self, "bn"):
            self.__delattr__("bn")  # 删除bn属性
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")  # 删除id_tensor属性


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""
    # 通道注意力模块

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        # 初始化类并设置基本配置和实例变量
        super().__init__()  # 调用父类构造函数
        self.pool = nn.AdaptiveAvgPool2d(1)  # 定义自适应平均池化层
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)  # 定义1x1卷积层
        self.act = nn.Sigmoid()  # 定义Sigmoid激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        # 对输入的卷积应用激活，选择性使用批归一化
        return x * self.act(self.fc(self.pool(x)))  # 返回经过激活的结果


class SpatialAttention(nn.Module):
    """Spatial-attention module."""
    # 空间注意力模块

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        # 初始化空间注意力模块，给定卷积核大小参数
        super().__init__()  # 调用父类构造函数
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"  # 确保卷积核大小为3或7
        padding = 3 if kernel_size == 7 else 1  # 根据卷积核大小设置填充
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 定义卷积层
        self.act = nn.Sigmoid()  # 定义Sigmoid激活函数

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        # 对输入应用通道和空间注意力以重新校准特征
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))  # 返回经过激活的结果


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    # 卷积块注意力模块

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        # 初始化CBAM，给定输入通道和卷积核大小
        super().__init__()  # 调用父类构造函数
        self.channel_attention = ChannelAttention(c1)  # 定义通道注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)  # 定义空间注意力模块

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        # 通过C1模块执行前向传播
        return self.spatial_attention(self.channel_attention(x))  # 返回经过空间注意力的结果


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    # 沿指定维度连接张量列表

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        # 沿指定维度连接张量列表
        super().__init__()  # 调用父类构造函数
        self.d = dimension  # 保存维度

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        # YOLOv8掩码Proto模块的前向传播
        return torch.cat(x, self.d)  # 返回连接后的结果


class Index(nn.Module):
    """Returns a particular index of the input."""
    # 返回输入的特定索引

    def __init__(self, index=0):
        """Returns a particular index of the input."""
        # 返回输入的特定索引
        super().__init__()  # 调用父类构造函数
        self.index = index  # 保存索引

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        # 前向传播
        return x[self.index]  # 返回指定索引的张量