# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",  # 分布焦点损失模块
    "HGBlock",  # HGBlock模块
    "HGStem",  # HGStem模块
    "SPP",  # 空间金字塔池化模块
    "SPPF",  # 快速空间金字塔池化模块
    "C1",  # CSP瓶颈模块，1个卷积
    "C2",  # CSP瓶颈模块，2个卷积
    "C3",  # CSP瓶颈模块，3个卷积
    "C2f",  # CSP瓶颈模块，2个卷积，快速实现
    "C2fAttn",  # CSP瓶颈模块，2个卷积，带注意力机制
    "ImagePoolingAttn",  # 图像池化注意力模块
    "ContrastiveHead",  # 对比头模块
    "BNContrastiveHead",  # BN对比头模块
    "C3x",  # C3模块，带交叉卷积
    "C3TR",  # C3模块，带变换卷积
    "C3Ghost",  # C3模块，Ghost卷积
    "GhostBottleneck",  # Ghost瓶颈模块
    "Bottleneck",  # 瓶颈模块
    "BottleneckCSP",  # CSP瓶颈模块
    "Proto",  # YOLOv8掩膜原型模块
    "RepC3",  # 重复C3模块
    "ResNetLayer",  # ResNet层
    "RepNCSPELAN4",  # 重复NCSPELAN4模块
    "ELAN1",  # ELAN模块
    "ADown",  # ADown模块
    "AConv",  # AConv模块
    "SPPELAN",  # SPPELAN模块
    "CBFuse",  # CB融合模块
    "CBLinear",  # CB线性模块
    "C3k2",  # C3k2模块
    "C2fPSA",  # C2f PSA模块
    "C2PSA",  # C2 PSA模块
    "RepVGGDW",  # 重复VGG DW模块
    "CIB",  # CIB模块
    "C2fCIB",  # C2f CIB模块
    "Attention",  # 注意力模块
    "PSA",  # PSA模块
    "SCDown",  # SCDown模块
    "TorchVision",  # TorchVision模块
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    分布焦点损失（DFL）的整体模块。

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    在《广义焦点损失》中提出 https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        # 用给定数量的输入通道初始化卷积层
        super().__init__()  # 调用父类的初始化方法
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)  # 定义卷积层，不需要偏置，不需要梯度
        x = torch.arange(c1, dtype=torch.float)  # 创建一个从0到c1的张量
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))  # 将权重初始化为x的参数
        self.c1 = c1  # 保存输入通道数量

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        # 在输入张量'x'上应用变换层并返回张量
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # 将输入张量重塑并应用卷积，返回处理后的张量


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""
    # YOLOv8掩膜原型模块，用于分割模型

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        初始化YOLOv8掩膜原型模块，指定原型和掩膜的数量。

        输入参数为输入通道数、原型数量、掩膜数量。
        """
        super().__init__()  # 调用父类的初始化方法
        self.cv1 = Conv(c1, c_, k=3)  # 定义第一个卷积层
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # 定义转置卷积层进行上采样
        self.cv2 = Conv(c_, c_, k=3)  # 定义第二个卷积层
        self.cv3 = Conv(c_, c2)  # 定义第三个卷积层

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        # 使用上采样的输入图像执行前向传播
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))  # 依次通过卷积层和上采样层


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    PPHGNetV2的StemBlock，包含5个卷积和一个最大池化层。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        # 用输入/输出通道和指定的最大池化核大小初始化SPP层
        super().__init__()  # 调用父类的初始化方法
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())  # 定义第一个卷积层
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())  # 定义第二个卷积层
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())  # 定义第三个卷积层
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())  # 定义第四个卷积层
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())  # 定义第五个卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)  # 定义最大池化层

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        # PPHGNetV2主干层的前向传播
        x = self.stem1(x)  # 通过第一个卷积层
        x = F.pad(x, [0, 1, 0, 1])  # 对x进行填充
        x2 = self.stem2a(x)  # 通过第二个卷积层
        x2 = F.pad(x2, [0, 1, 0, 1])  # 对x2进行填充
        x2 = self.stem2b(x2)  # 通过第三个卷积层
        x1 = self.pool(x)  # 通过最大池化层
        x = torch.cat([x1, x2], dim=1)  # 在通道维度上连接x1和x2
        x = self.stem3(x)  # 通过第四个卷积层
        x = self.stem4(x)  # 通过第五个卷积层
        return x  # 返回输出


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    PPHGNetV2的HG_Block，包含2个卷积和LightConv。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        # 使用指定的输入和输出通道初始化CSP瓶颈，包含1个卷积
        super().__init__()  # 调用父类的初始化方法
        block = LightConv if lightconv else Conv  # 根据lightconv参数选择卷积类型
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))  # 创建模块列表
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze卷积
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation卷积
        self.add = shortcut and c1 == c2  # 判断是否使用shortcut连接

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        # PPHGNetV2主干层的前向传播
        y = [x]  # 初始化y为输入x
        y.extend(m(y[-1]) for m in self.m)  # 将每个模块应用于y的最后一个元素
        y = self.ec(self.sc(torch.cat(y, 1)))  # 通过squeeze和excitation卷积处理y
        return y + x if self.add else y  # 如果添加shortcut连接，则返回y + x，否则返回y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""
    # 空间金字塔池化（SPP）层

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        # 用输入/输出通道和池化核大小初始化SPP层
        super().__init__()  # 调用父类的初始化方法
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)  # 定义第二个卷积层
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])  # 创建最大池化层列表

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        # SPP层的前向传播，执行空间金字塔池化
        x = self.cv1(x)  # 通过第一个卷积层
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))  # 连接x和池化结果并通过第二个卷积层


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    # 快速空间金字塔池化（SPPF）层，适用于YOLOv5，由Glenn Jocher提出

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        用给定的输入/输出通道和核大小初始化SPPF层。

        此模块等效于SPP(k=(5, 9, 13))。
        """
        super().__init__()  # 调用父类的初始化方法
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 定义第二个卷积层
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 定义最大池化层

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        # 通过Ghost卷积块的前向传播
        y = [self.cv1(x)]  # 通过第一个卷积层
        y.extend(self.m(y[-1]) for _ in range(3))  # 进行3次最大池化
        return self.cv2(torch.cat(y, 1))  # 连接y并通过第二个卷积层


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""
    # CSP瓶颈，包含1个卷积

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        # 用输入通道、输出通道和数量初始化CSP瓶颈
        super().__init__()  # 调用父类的初始化方法
        self.cv1 = Conv(c1, c2, 1, 1)  # 定义第一个卷积层
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))  # 创建多个卷积层的序列

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        # 在C3模块中对输入应用交叉卷积
        y = self.cv1(x)  # 通过第一个卷积层
        return self.m(y) + y  # 返回卷积结果与输入的和


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    # CSP瓶颈，包含2个卷积

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        # 初始化一个包含2个卷积和可选shortcut连接的CSP瓶颈
        super().__init__()  # 调用父类的初始化方法
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(2 * self.c, c2, 1)  # 定义第二个卷积层
        # self.attention = ChannelAttention(2 * self.c)  # 或者使用空间注意力
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))  # 创建多个瓶颈层的序列

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        # 通过包含2个卷积的CSP瓶颈的前向传播
        a, b = self.cv1(x).chunk(2, 1)  # 将输出分成两部分
        return self.cv2(torch.cat((self.m(a), b), 1))  # 连接并通过第二个卷积层


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    # CSP瓶颈的更快实现，包含2个卷积

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        # 初始化一个包含2个卷积和n个瓶颈块的CSP瓶颈，以实现更快的处理
        super().__init__()  # 调用父类的初始化方法
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 定义第二个卷积层
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))  # 创建多个瓶颈块的模块列表

    def forward(self, x):
        """Forward pass through C2f layer."""
        # 通过C2f层的前向传播
        y = list(self.cv1(x).chunk(2, 1))  # 将输出分成两部分
        y.extend(m(y[-1]) for m in self.m)  # 将瓶颈块应用于最后一个部分
        return self.cv2(torch.cat(y, 1))  # 连接并通过第二个卷积层


    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        # 使用split()而不是chunk()的前向传播
        y = self.cv1(x).split((self.c, self.c), 1)  # 将输出分成两部分
        y = [y[0], y[1]]  # 创建一个列表
        y.extend(m(y[-1]) for m in self.m)  # 将瓶颈块应用于最后一个部分
        return self.cv2(torch.cat(y, 1))  # 连接并通过第二个卷积层


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    # CSP瓶颈，包含3个卷积

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        # 用给定的通道、数量、shortcut、组和扩展值初始化CSP瓶颈
        super().__init__()  # 调用父类的初始化方法
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c1, c_, 1, 1)  # 定义第二个卷积层
        self.cv3 = Conv(2 * c_, c2, 1)  # 定义第三个卷积层
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))  # 创建多个瓶颈层的序列

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        # 通过包含2个卷积的CSP瓶颈的前向传播
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))  # 连接并通过第三个卷积层


class C3x(C3):
    """C3 module with cross-convolutions."""
    # 带交叉卷积的C3模块

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        # 初始化C3TR实例并设置默认参数
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类的初始化方法
        self.c_ = int(c2 * e)  # 隐藏通道
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))  # 创建多个瓶颈层的序列

class RepC3(nn.Module):
    """Rep C3."""
    # 定义Rep C3类，继承自nn.Module

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        # 初始化CSP Bottleneck，使用输入通道、输出通道和数量进行单卷积初始化
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # hidden channels，计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c1, c_, 1, 1)  # 定义第二个卷积层
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])  # 定义n个RepConv的顺序容器
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()  # 定义第三个卷积层，如果隐藏通道数不等于输出通道数，则使用卷积，否则使用恒等映射

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        # RT-DETR颈部层的前向传播
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))  # 返回卷积结果和输入的和

class C3TR(C3):
    """C3 module with TransformerBlock()."""
    # 定义C3TR类，继承自C3，包含TransformerBlock

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        # 初始化C3Ghost模块，使用GhostBottleneck
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.m = TransformerBlock(c_, c_, 4, n)  # 定义TransformerBlock

class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""
    # 定义C3Ghost类，继承自C3，包含GhostBottleneck

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        # 初始化'SPP'模块，使用不同的池化大小进行空间金字塔池化
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))  # 定义n个GhostBottleneck的顺序容器

class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""
    # 定义GhostBottleneck类，继承自nn.Module

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        # 初始化GhostBottleneck模块，参数为输入通道、输出通道、卷积核大小和步幅
        super().__init__()  # 调用父类构造函数
        c_ = c2 // 2  # 计算中间通道数
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw，定义逐点卷积
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw，定义深度卷积，如果步幅为2则使用深度卷积，否则使用恒等映射
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear，定义逐点卷积，输出通道为c2
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )  # 定义shortcut，如果步幅为2则使用深度卷积和卷积层，否则使用恒等映射

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        # 应用跳跃连接和输入张量的拼接
        return self.conv(x) + self.shortcut(x)  # 返回卷积结果与shortcut的和

class Bottleneck(nn.Module):
    """Standard bottleneck."""
    # 定义标准瓶颈类，继承自nn.Module

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        # 初始化标准瓶颈模块，具有可选的shortcut连接和可配置参数
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, k[0], 1)  # 定义第一个卷积层
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # 定义第二个卷积层
        self.add = shortcut and c1 == c2  # 确定是否添加shortcut

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # 将YOLO FPN应用于输入数据
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  # 返回输入与卷积结果的和（如果添加shortcut）

class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""
    # 定义CSP Bottleneck类，继承自nn.Module

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        # 初始化CSP Bottleneck，给定输入通道、输出通道、数量、shortcut、组数和扩展参数
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 定义第一个卷积层
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # 定义第二个卷积层
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  # 定义第三个卷积层
        self.cv4 = Conv(2 * c_, c2, 1, 1)  # 定义第四个卷积层
        self.bn = nn.BatchNorm2d(2 * c_)  # 对拼接后的通道进行批归一化
        self.act = nn.SiLU()  # 定义激活函数
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))  # 定义n个Bottleneck的顺序容器

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        # 应用具有3个卷积的CSP瓶颈
        y1 = self.cv3(self.m(self.cv1(x)))  # 通过第一个卷积和Bottleneck序列计算y1
        y2 = self.cv2(x)  # 通过第二个卷积计算y2
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))  # 返回经过激活和批归一化后的拼接结果

class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""
    # 定义ResNetBlock类，继承自nn.Module

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        # 使用给定参数初始化卷积
        super().__init__()  # 调用父类构造函数
        c3 = e * c2  # 计算扩展后的通道数
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)  # 定义第一个卷积层
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)  # 定义第二个卷积层
        self.cv3 = Conv(c2, c3, k=1, act=False)  # 定义第三个卷积层
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()  # 定义shortcut

    def forward(self, x):
        """Forward pass through the ResNet block."""
        # 通过ResNet块的前向传播
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))  # 返回经过ReLU激活的结果

class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""
    # 定义ResNetLayer类，继承自nn.Module，表示具有多个ResNet块的层

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        # 初始化ResNetLayer，给定输入通道、输出通道、步幅、是否为第一层、块的数量和扩展因子
        super().__init__()  # 调用父类构造函数
        self.is_first = is_first  # 标记是否为第一层

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),  # 定义第一层卷积，卷积核大小为7，步幅为2，填充为3，激活函数为ReLU
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 定义最大池化层，池化核大小为3，步幅为2，填充为1
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]  # 创建第一个ResNetBlock
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])  # 创建n-1个ResNetBlock并扩展到blocks列表
            self.layer = nn.Sequential(*blocks)  # 将所有块组合成一个顺序容器

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        # 通过ResNet层的前向传播
        return self.layer(x)  # 返回层的输出

class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""
    # 定义MaxSigmoidAttnBlock类，继承自nn.Module，表示最大Sigmoid注意力块

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        # 初始化MaxSigmoidAttnBlock，给定输入通道、输出通道、头数、扩展通道、全局通道和缩放标志
        super().__init__()  # 调用父类构造函数
        self.nh = nh  # 头数
        self.hc = c2 // nh  # 每个头的通道数
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None  # 如果输入通道不等于扩展通道，则定义卷积层
        self.gl = nn.Linear(gc, ec)  # 定义线性层，将全局通道映射到扩展通道
        self.bias = nn.Parameter(torch.zeros(nh))  # 定义偏置参数
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)  # 定义卷积层，用于投影
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0  # 如果需要缩放，则定义缩放参数

    def forward(self, x, guide):
        """Forward process."""
        # 前向过程
        bs, _, h, w = x.shape  # 获取输入张量的批量大小、高度和宽度

        guide = self.gl(guide)  # 通过线性层处理引导张量
        guide = guide.view(bs, -1, self.nh, self.hc)  # 调整引导张量的形状
        embed = self.ec(x) if self.ec is not None else x  # 如果存在扩展卷积，则处理输入张量
        embed = embed.view(bs, self.nh, self.hc, h, w)  # 调整嵌入张量的形状

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)  # 计算注意力权重
        aw = aw.max(dim=-1)[0]  # 取最大值
        aw = aw / (self.hc**0.5)  # 进行缩放
        aw = aw + self.bias[None, :, None, None]  # 添加偏置
        aw = aw.sigmoid() * self.scale  # 应用Sigmoid激活并缩放

        x = self.proj_conv(x)  # 通过投影卷积处理输入张量
        x = x.view(bs, self.nh, -1, h, w)  # 调整投影结果的形状
        x = x * aw.unsqueeze(2)  # 应用注意力权重
        return x.view(bs, -1, h, w)  # 返回调整后的张量

class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""
    # 定义C2fAttn类，继承自nn.Module，表示具有附加注意力模块的C2f模块

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        # 初始化C2f模块，添加注意力机制以增强特征提取和处理
        super().__init__()  # 调用父类构造函数
        self.c = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # 定义第二个卷积层
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))  # 创建n个Bottleneck
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)  # 定义注意力块

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        # 通过C2f层的前向传播
        y = list(self.cv1(x).chunk(2, 1))  # 将输入分成两部分
        y.extend(m(y[-1]) for m in self.m)  # 通过Bottleneck处理最后一部分
        y.append(self.attn(y[-1], guide))  # 添加注意力块的输出
        return self.cv2(torch.cat(y, 1))  # 将所有输出拼接并通过第二个卷积层

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        # 使用split()进行前向传播
        y = list(self.cv1(x).split((self.c, self.c), 1))  # 将输入分成两部分
        y.extend(m(y[-1]) for m in self.m)  # 通过Bottleneck处理最后一部分
        y.append(self.attn(y[-1], guide))  # 添加注意力块的输出
        return self.cv2(torch.cat(y, 1))  # 将所有输出拼接并通过第二个卷积层

class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""
    # 定义ImagePoolingAttn类，继承自nn.Module，表示通过图像感知信息增强文本嵌入的模块

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        # 初始化ImagePoolingAttn，给定扩展通道、通道列表、文本通道、头数、池化大小和缩放标志
        super().__init__()  # 调用父类构造函数

        nf = len(ch)  # 获取通道数量
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))  # 定义查询层
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))  # 定义键层
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))  # 定义值层
        self.proj = nn.Linear(ec, ct)  # 定义投影层
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0  # 如果需要缩放，则定义缩放参数
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])  # 定义投影卷积层
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])  # 定义自适应最大池化层
        self.ec = ec  # 保存扩展通道数
        self.nh = nh  # 保存头数
        self.nf = nf  # 保存通道数量
        self.hc = ec // nh  # 每个头的通道数
        self.k = k  # 保存池化大小

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        # 在输入张量x和引导张量上执行注意力机制
        bs = x[0].shape[0]  # 获取批量大小
        assert len(x) == self.nf  # 确保输入张量数量与通道数量一致
        num_patches = self.k**2  # 计算每个图像的补丁数量
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]  # 处理每个输入张量
        x = torch.cat(x, dim=-1).transpose(1, 2)  # 拼接并转置张量
        q = self.query(text)  # 通过查询层处理文本
        k = self.key(x)  # 通过键层处理输入
        v = self.value(x)  # 通过值层处理输入

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)  # 调整查询张量的形状
        k = k.reshape(bs, -1, self.nh, self.hc)  # 调整键张量的形状
        v = v.reshape(bs, -1, self.nh, self.hc)  # 调整值张量的形状

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)  # 计算注意力权重
        aw = aw / (self.hc**0.5)  # 进行缩放
        aw = F.softmax(aw, dim=-1)  # 应用Softmax激活

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)  # 计算加权值
        x = self.proj(x.reshape(bs, -1, self.ec))  # 通过投影层处理
        return x * self.scale + text  # 返回经过缩放的结果与文本的和

class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""
    # 定义ContrastiveHead类，继承自nn.Module，实现对区域-文本相似度的对比学习头

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        # 初始化ContrastiveHead，给定区域-文本相似度参数
        super().__init__()  # 调用父类构造函数
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))  # 定义偏置参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())  # 定义logit缩放参数

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        # 对比学习的前向函数
        x = F.normalize(x, dim=1, p=2)  # 对输入进行L2归一化
        w = F.normalize(w, dim=-1, p=2)  # 对权重进行L2归一化
        x = torch.einsum("bchw,bkc->bkhw", x, w)  # 计算相似度
        return x * self.logit_scale.exp() + self.bias  # 返回经过缩放和偏置的结果

class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """
    # 定义BNContrastiveHead类，继承自nn.Module，使用批归一化的对比学习头

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        # 初始化ContrastiveHead，给定区域-文本相似度参数
        super().__init__()  # 调用父类构造函数
        self.norm = nn.BatchNorm2d(embed_dims)  # 定义批归一化层
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))  # 定义偏置参数
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))  # 定义logit缩放参数

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        # 对比学习的前向函数
        x = self.norm(x)  # 对输入进行批归一化
        w = F.normalize(w, dim=-1, p=2)  # 对权重进行L2归一化
        x = torch.einsum("bchw,bkc->bkhw", x, w)  # 计算相似度
        return x * self.logit_scale.exp() + self.bias  # 返回经过缩放和偏置的结果

class RepBottleneck(Bottleneck):
    """Rep bottleneck."""
    # 定义RepBottleneck类，继承自Bottleneck，表示重复瓶颈模块

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        # 初始化RepBottleneck模块，具有可定制的输入/输出通道、shortcut、组和扩展
        super().__init__(c1, c2, shortcut, g, k, e)  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = RepConv(c1, c_, k[0], 1)  # 定义第一个重复卷积层

class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""
    # 定义RepCSP类，继承自C3，表示可重复的跨阶段部分网络模块，用于高效特征提取

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        # 初始化RepCSP层，给定通道、重复次数、shortcut、组和扩展比例
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))  # 定义n个RepBottleneck的顺序容器

class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""
    # 定义RepNCSPELAN4类，继承自nn.Module，表示CSP-ELAN层

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        # 初始化CSP-ELAN层，给定通道大小、重复次数和卷积层
        super().__init__()  # 调用父类构造函数
        self.c = c3 // 2  # 计算隐藏通道数
        self.cv1 = Conv(c1, c3, 1, 1)  # 定义第一个卷积层
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))  # 定义第二个卷积层，包含RepCSP和卷积
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))  # 定义第三个卷积层，包含RepCSP和卷积
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)  # 定义第四个卷积层，输入通道为c3和c4的两倍之和

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        # 通过RepNCSPELAN4层的前向传播
        y = list(self.cv1(x).chunk(2, 1))  # 将输入通过第一个卷积层处理并分成两部分
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])  # 通过cv2和cv3处理最后一部分并扩展y
        return self.cv4(torch.cat(y, 1))  # 将所有输出拼接并通过第四个卷积层

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        # 使用split()进行前向传播
        y = list(self.cv1(x).split((self.c, self.c), 1))  # 将输入通过第一个卷积层处理并分成两部分
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])  # 通过cv2和cv3处理最后一部分并扩展y
        return self.cv4(torch.cat(y, 1))  # 将所有输出拼接并通过第四个卷积层

class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""
    # 定义ELAN1类，继承自RepNCSPELAN4，表示具有4个卷积的ELAN1模块

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        # 初始化ELAN1层，给定通道大小
        super().__init__(c1, c2, c3, c4)  # 调用父类构造函数
        self.c = c3 // 2  # 计算隐藏通道数
        self.cv1 = Conv(c1, c3, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c3 // 2, c4, 3, 1)  # 定义第二个卷积层
        self.cv3 = Conv(c4, c4, 3, 1)  # 定义第三个卷积层
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)  # 定义第四个卷积层，输入通道为c3和c4的两倍之和

class AConv(nn.Module):
    """AConv."""
    # 定义AConv类，继承自nn.Module，表示AConv模块

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        # 初始化AConv模块，给定输入和输出通道
        super().__init__()  # 调用父类构造函数
        self.cv1 = Conv(c1, c2, 3, 2, 1)  # 定义卷积层，卷积核大小为3，步幅为2，填充为1

    def forward(self, x):
        """Forward pass through AConv layer."""
        # 通过AConv层的前向传播
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)  # 对输入进行平均池化
        return self.cv1(x)  # 返回卷积层的输出

class ADown(nn.Module):
    """ADown."""
    # 定义ADown类，继承自nn.Module，表示ADown模块

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        # 初始化ADown模块，给定输入通道和输出通道
        super().__init__()  # 调用父类构造函数
        self.c = c2 // 2  # 计算隐藏通道数
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)  # 定义第二个卷积层

    def forward(self, x):
        """Forward pass through ADown layer."""
        # 通过ADown层的前向传播
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)  # 对输入进行平均池化
        x1, x2 = x.chunk(2, 1)  # 将输入分成两部分
        x1 = self.cv1(x1)  # 通过第一个卷积层处理第一部分
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)  # 对第二部分进行最大池化
        x2 = self.cv2(x2)  # 通过第二个卷积层处理第二部分
        return torch.cat((x1, x2), 1)  # 返回拼接后的结果

class SPPELAN(nn.Module):
    """SPP-ELAN."""
    # 定义SPPELAN类，继承自nn.Module，表示SPP-ELAN模块

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        # 初始化SPP-ELAN块，给定输入通道、输出通道、隐藏通道和池化大小
        super().__init__()  # 调用父类构造函数
        self.c = c3  # 保存隐藏通道数
        self.cv1 = Conv(c1, c3, 1, 1)  # 定义第一个卷积层
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 定义最大池化层
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 定义最大池化层
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 定义最大池化层
        self.cv5 = Conv(4 * c3, c2, 1, 1)  # 定义最后的卷积层

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        # 通过SPPELAN层的前向传播
        y = [self.cv1(x)]  # 通过第一个卷积层处理输入
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])  # 通过最大池化层处理并扩展y
        return self.cv5(torch.cat(y, 1))  # 将所有输出拼接并通过最后的卷积层

class CBLinear(nn.Module):
    """CBLinear."""
    # 定义CBLinear类，继承自nn.Module，表示CBLinear模块

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        # 初始化CBLinear模块，给定输入通道、输出通道、卷积核大小、步幅、填充和组数
        super().__init__()  # 调用父类构造函数
        self.c2s = c2s  # 保存输出通道列表
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)  # 定义卷积层

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        # 通过CBLinear层的前向传播
        return self.conv(x).split(self.c2s, dim=1)  # 返回分割后的输出

class CBFuse(nn.Module):
    """CBFuse."""
    # 定义CBFuse类，继承自nn.Module，表示CBFuse模块

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        # 初始化CBFuse模块，给定层索引以进行选择性特征融合
        super().__init__()  # 调用父类构造函数
        self.idx = idx  # 保存索引

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        # 通过CBFuse层的前向传播
        target_size = xs[-1].shape[2:]  # 获取目标大小
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]  # 对每个特征图进行插值
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)  # 返回融合后的结果

class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    # 定义C3f类，继承自nn.Module，表示具有两个卷积的CSP瓶颈的更快实现

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        # 初始化CSP瓶颈层，给定输入通道、输出通道、重复次数、shortcut、组和扩展因子
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c1, c_, 1, 1)  # 定义第二个卷积层
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # 定义第三个卷积层
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))  # 创建n个Bottleneck

    def forward(self, x):
        """Forward pass through C2f layer."""
        # 通过C2f层的前向传播
        y = [self.cv2(x), self.cv1(x)]  # 通过两个卷积层处理输入
        y.extend(m(y[-1]) for m in self.m)  # 通过Bottleneck处理最后一部分
        return self.cv3(torch.cat(y, 1))  # 将所有输出拼接并通过第三个卷积层

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    # 定义C3k2类，继承自C2f，表示具有两个卷积的CSP瓶颈的更快实现

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        # 初始化C3k2模块，给定输入通道、输出通道、重复次数、是否使用C3k块、扩展因子、组数和shortcut
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类构造函数
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )  # 创建n个C3k或Bottleneck模块

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    # 定义C3k类，继承自C3，表示具有可定制卷积核大小的CSP瓶颈模块

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        # 初始化C3k模块，给定输入通道、输出通道、层数和配置
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))  # 创建n个Bottleneck模块

class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""
    # 定义RepVGGDW类，继承自nn.Module，表示RepVGG架构中的深度可分离卷积块

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        # 初始化RepVGGDW，使用深度可分离卷积层进行高效处理
        super().__init__()  # 调用父类构造函数
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)  # 定义第一个卷积层
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)  # 定义第二个卷积层
        self.dim = ed  # 保存输入通道数
        self.act = nn.SiLU()  # 定义激活函数

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        # 通过RepVGGDW块的前向传播
        return self.act(self.conv(x) + self.conv1(x))  # 返回经过激活的卷积结果

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        # 通过RepVGGDW块的前向传播，不融合卷积
        return self.act(self.conv(x))  # 返回经过激活的卷积结果

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        # 融合RepVGGDW块中的卷积层
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)  # 融合第一个卷积层
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)  # 融合第二个卷积层

        conv_w = conv.weight  # 获取第一个卷积层的权重
        conv_b = conv.bias  # 获取第一个卷积层的偏置
        conv1_w = conv1.weight  # 获取第二个卷积层的权重
        conv1_b = conv1.bias  # 获取第二个卷积层的偏置

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])  # 对第二个卷积层的权重进行填充

        final_conv_w = conv_w + conv1_w  # 合并权重
        final_conv_b = conv_b + conv1_b  # 合并偏置

        conv.weight.data.copy_(final_conv_w)  # 更新第一个卷积层的权重
        conv.bias.data.copy_(final_conv_b)  # 更新第一个卷积层的偏置

        self.conv = conv  # 更新卷积层
        del self.conv1  # 删除第二个卷积层

class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """
    # 定义CIB类，继承自nn.Module，表示条件身份块模块

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        # 初始化自定义模型，给定可选的shortcut、缩放因子和RepVGGDW层
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),  # 定义第一个卷积层
            Conv(c1, 2 * c_, 1),  # 定义第二个卷积层
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),  # 定义第三个卷积层，使用RepVGGDW或常规卷积
            Conv(2 * c_, c2, 1),  # 定义第四个卷积层
            Conv(c2, c2, 3, g=c2),  # 定义第五个卷积层
        )

        self.add = shortcut and c1 == c2  # 确定是否添加shortcut

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        # 通过CIB模块的前向传播
        return x + self.cv1(x) if self.add else self.cv1(x)  # 返回经过shortcut的结果或仅卷积结果

class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """
    # 定义C2fCIB类，继承自C2f，表示具有C2f和CIB模块的卷积块

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        # 初始化模块，给定通道、shortcut、局部键、组和扩展参数
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类构造函数
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))  # 创建n个CIB模块

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """
    # 定义Attention类，继承自nn.Module，表示自注意力模块

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        # 初始化多头注意力模块，给定查询、键和值的卷积和位置编码
        super().__init__()  # 调用父类构造函数
        self.num_heads = num_heads  # 保存头数
        self.head_dim = dim // num_heads  # 计算每个头的维度
        self.key_dim = int(self.head_dim * attn_ratio)  # 计算注意力键的维度
        self.scale = self.key_dim**-0.5  # 计算缩放因子
        nh_kd = self.key_dim * num_heads  # 计算总的键维度
        h = dim + nh_kd * 2  # 计算输入维度
        self.qkv = Conv(dim, h, 1, act=False)  # 定义卷积层，用于计算查询、键和值
        self.proj = Conv(dim, dim, 1, act=False)  # 定义投影层
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # 定义位置编码层

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        # 通过注意力模块的前向传播
        B, C, H, W = x.shape  # 获取输入张量的批量大小、通道数、高度和宽度
        N = H * W  # 计算总的patch数量
        qkv = self.qkv(x)  # 通过qkv卷积层计算qkv
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )  # 将qkv分割成q、k和v

        attn = (q.transpose(-2, -1) @ k) * self.scale  # 计算注意力权重
        attn = attn.softmax(dim=-1)  # 应用Softmax激活
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))  # 计算加权值并添加位置编码
        x = self.proj(x)  # 通过投影层处理
        return x  # 返回输出


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """
    # 定义PSABlock类，继承自nn.Module，表示位置敏感注意力块

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        # 初始化PSABlock，给定通道数、注意力比率、头数和shortcut标志
        super().__init__()  # 调用父类构造函数

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)  # 定义多头注意力模块
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))  # 定义前馈神经网络模块
        self.add = shortcut  # 保存shortcut标志

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        # 通过PSABlock的前向传播，应用注意力和前馈层
        x = x + self.attn(x) if self.add else self.attn(x)  # 如果添加shortcut，则返回输入与注意力的和
        x = x + self.ffn(x) if self.add else self.ffn(x)  # 如果添加shortcut，则返回输入与前馈网络的和
        return x  # 返回输出

class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """
    # 定义PSA类，继承自nn.Module，表示位置敏感注意力模块

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        # 初始化PSA模块，给定输入/输出通道和特征提取的注意力机制
        super().__init__()  # 调用父类构造函数
        assert c1 == c2  # 确保输入和输出通道数相等
        self.c = int(c1 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(2 * self.c, c1, 1)  # 定义第二个卷积层

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)  # 定义位置敏感注意力模块
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))  # 定义前馈网络

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        # 在PSA模块中执行前向传播，应用注意力和前馈层
        a, b = self.cv1(x).split((self.c, self.c), dim=1)  # 将输入通过第一个卷积层处理并分成两部分
        b = b + self.attn(b)  # 通过注意力模块处理第二部分
        b = b + self.ffn(b)  # 通过前馈网络处理第二部分
        return self.cv2(torch.cat((a, b), 1))  # 将两部分拼接并通过第二个卷积层返回结果

class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """
    # 定义C2PSA类，继承自nn.Module，表示具有注意力机制的C2PSA模块

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        # 初始化C2PSA模块，给定输入/输出通道、层数和扩展比例
        super().__init__()  # 调用父类构造函数
        assert c1 == c2  # 确保输入和输出通道数相等
        self.c = int(c1 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(2 * self.c, c1, 1)  # 定义第二个卷积层

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))  # 创建n个PSABlock模块

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        # 通过一系列PSA块处理输入张量'x'并返回变换后的张量
        a, b = self.cv1(x).split((self.c, self.c), dim=1)  # 将输入通过第一个卷积层处理并分成两部分
        b = self.m(b)  # 通过PSABlock模块处理第二部分
        return self.cv2(torch.cat((a, b), 1))  # 将两部分拼接并通过第二个卷积层返回结果

class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """
    # 定义C2fPSA类，继承自C2f，表示具有PSA块的C2fPSA模块

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        # 初始化C2fPSA模块，给定输入通道、输出通道、重复次数和扩展比例
        assert c1 == c2  # 确保输入和输出通道数相等
        super().__init__(c1, c2, n=n, e=e)  # 调用父类构造函数
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))  # 创建n个PSABlock模块

class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """
    # 定义SCDown类，继承自nn.Module，表示用于下采样的SCDown模块

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        # 初始化SCDown模块，给定输入通道、输出通道、卷积核大小和步幅
        super().__init__()  # 调用父类构造函数
        self.cv1 = Conv(c1, c2, 1, 1)  # 定义第一个卷积层
        self.cv2 = Conv(c2, c2, k, s, g=c2, act=False)  # 定义第二个卷积层

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        # 在SCDown模块中对输入张量应用卷积和下采样
        return self.cv2(self.cv1(x))  # 返回经过卷积和下采样的结果

class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """
    # 定义TorchVision类，继承自nn.Module，表示加载任何torchvision模型的模块

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        # 从torchvision加载模型和权重
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()  # 调用父类构造函数
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)  # 使用get_model方法加载模型
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))  # 使用字典加载模型
        if unwrap:
            layers = list(self.m.children())  # 获取模型的所有子层
            if isinstance(layers[0], nn.Sequential):  # 对于某些模型（如EfficientNet、Swin），第二级别的处理
                layers = [*list(layers[0].children()), *layers[1:]]  # 拆分子层
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))  # 创建顺序容器
            self.split = split  # 保存是否拆分的标志
        else:
            self.split = False  # 不拆分
            self.m.head = self.m.heads = nn.Identity()  # 将头部设置为恒等映射

    def forward(self, x):
        """Forward pass through the model."""
        # 通过模型的前向传播
        if self.split:
            y = [x]  # 初始化输出列表
            y.extend(m(y[-1]) for m in self.m)  # 通过每个子模块处理
        else:
            y = self.m(x)  # 直接通过模型处理
        return y  # 返回输出