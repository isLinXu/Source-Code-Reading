# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "ResNetBlockBase",
    "BasicBlock",
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "make_stage",
    "build_resnet_backbone",
]


class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        # 第一个1x1卷积层，用于降维
        self.conv1 = Conv2d(
            in_channels,          # 输入通道数
            bottleneck_channels,  # 降维后的通道数
            kernel_size=1,        # 1x1卷积核
            stride=stride_1x1,    # 可配置的步长
            bias=False,           # 不使用偏置项
            norm=get_norm(norm, bottleneck_channels),  # 添加归一化层
        )

        # 3x3卷积层，主要特征提取
        self.conv2 = Conv2d(
            bottleneck_channels,  # 输入通道数（与conv1输出相同）
            bottleneck_channels,  # 输出通道数保持不变
            kernel_size=3,        # 3x3卷积核
            stride=stride_3x3,    # 可配置的步长
            padding=1 * dilation, # 考虑膨胀率的填充
            bias=False,           # 不使用偏置项
            groups=num_groups,    # 分组卷积
            dilation=dilation,    # 膨胀卷积率
            norm=get_norm(norm, bottleneck_channels),  # 添加归一化层
        )

        # 第二个1x1卷积层，用于升维
        self.conv3 = Conv2d(
            bottleneck_channels,  # 输入通道数
            out_channels,         # 升维到输出通道数
            kernel_size=1,        # 1x1卷积核
            bias=False,           # 不使用偏置项
            norm=get_norm(norm, out_channels),  # 添加归一化层
        )

        # 对所有卷积层进行MSRA初始化
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut可能为None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."
        # 将每个残差分支中最后的归一化层初始化为零，这样在开始时，残差分支从零开始，
        # 每个残差块的行为就像一个恒等映射。详见论文第5.1节。
        # 对于BN层，可学习的缩放系数γ初始化为1，除了每个残差块的最后一个BN层，其γ初始化为0。

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.
        # 注：这种初始化方式在从头训练GN模型时会降低性能。
        # 当需要使用此代码训练主干网络时，可以将其作为一个选项。

    def forward(self, x):
        # 前向传播过程
        out = self.conv1(x)        # 第一个1x1卷积，降维
        out = F.relu_(out)         # ReLU激活（原地操作）

        out = self.conv2(out)      # 3x3卷积，特征提取
        out = F.relu_(out)         # ReLU激活（原地操作）

        out = self.conv3(out)      # 第二个1x1卷积，升维

        # 处理快捷连接
        if self.shortcut is not None:
            shortcut = self.shortcut(x)  # 通过1x1卷积调整维度
        else:
            shortcut = x               # 直接使用输入

        out += shortcut              # 残差连接
        out = F.relu_(out)          # 最终的ReLU激活（原地操作）
        return out


class DeformBottleneckBlock(CNNBlockBase):
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    类似于BottleneckBlock，但在3x3卷积层使用了可变形卷积。可变形卷积允许卷积核根据输入
    内容动态调整采样位置，增强了模型对物体形变的建模能力。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,  # 是否使用可调制的可变形卷积
        deform_num_groups=1,     # 可变形卷积的分组数
    ):
        # 调用父类初始化
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated  # 记录是否使用可调制的可变形卷积

        # 当输入输出通道数不同时，需要使用1x1卷积进行通道调整
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,      # 输入通道数
                out_channels,     # 输出通道数
                kernel_size=1,    # 1x1卷积核
                stride=stride,    # 使用与主分支相同的步长
                bias=False,       # 不使用偏置项
                norm=get_norm(norm, out_channels),  # 添加归一化层
            )
        else:
            self.shortcut = None  # 通道数相同时不需要快捷连接

        # 确定1x1卷积和3x3卷积的步长
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        # 第一个1x1卷积层，用于降维
        self.conv1 = Conv2d(
            in_channels,          # 输入通道数
            bottleneck_channels,  # 降维后的通道数
            kernel_size=1,        # 1x1卷积核
            stride=stride_1x1,    # 可配置的步长
            bias=False,           # 不使用偏置项
            norm=get_norm(norm, bottleneck_channels),  # 添加归一化层
        )

        # 根据是否使用可调制的可变形卷积选择不同的卷积操作
        if deform_modulated:
            deform_conv_op = ModulatedDeformConv  # 可调制的可变形卷积
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            # 偏移通道数：3（x偏移、y偏移、调制因子）* kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv  # 标准可变形卷积
            # 偏移通道数：2（x偏移、y偏移）* kernel_size * kernel_size
            offset_channels = 18

        # 用于预测偏移量的卷积层
        self.conv2_offset = Conv2d(
            bottleneck_channels,                  # 输入通道数
            offset_channels * deform_num_groups,  # 输出通道数（偏移量和调制因子）
            kernel_size=3,                       # 3x3卷积核
            stride=stride_3x3,                   # 与主卷积相同的步长
            padding=1 * dilation,                # 考虑膨胀率的填充
            dilation=dilation,                   # 膨胀率
        )
        # 可变形卷积层
        self.conv2 = deform_conv_op(
            bottleneck_channels,  # 输入通道数
            bottleneck_channels,  # 输出通道数
            kernel_size=3,        # 3x3卷积核
            stride=stride_3x3,    # 步长
            padding=1 * dilation, # 填充
            bias=False,           # 不使用偏置项
            groups=num_groups,    # 特征分组数
            dilation=dilation,    # 膨胀率
            deformable_groups=deform_num_groups,  # 可变形卷积分组数
            norm=get_norm(norm, bottleneck_channels),  # 归一化层
        )

        # 第二个1x1卷积层，用于升维
        self.conv3 = Conv2d(
            bottleneck_channels,  # 输入通道数
            out_channels,         # 升维到输出通道数
            kernel_size=1,        # 1x1卷积核
            bias=False,           # 不使用偏置项
            norm=get_norm(norm, out_channels),  # 归一化层
        )

        # 对所有主要卷积层进行MSRA初始化
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut可能为None
                weight_init.c2_msra_fill(layer)

        # 将偏移量预测层的权重和偏置初始化为0
        # 这样在训练初期，可变形卷积的采样位置会接近于规则卷积
        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        # 前向传播过程
        out = self.conv1(x)        # 第一个1x1卷积，降维
        out = F.relu_(out)         # ReLU激活（原地操作）

        # 根据是否使用可调制的可变形卷积选择不同的处理流程
        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)  # 预测偏移量和调制因子
            # 将输出分成x偏移、y偏移和调制因子
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            # 组合x和y偏移
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()  # 将调制因子转换到0-1范围
            # 应用可调制的可变形卷积
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)  # 只预测偏移量
            out = self.conv2(out, offset)    # 应用标准可变形卷积
        out = F.relu_(out)         # ReLU激活（原地操作）

        out = self.conv3(out)      # 第二个1x1卷积，升维

        # 处理快捷连接
        if self.shortcut is not None:
            shortcut = self.shortcut(x)  # 通过1x1卷积调整维度
        else:
            shortcut = x               # 直接使用输入

        out += shortcut              # 残差连接
        out = F.relu_(out)          # 最终的ReLU激活（原地操作）
        return out


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    标准ResNet的stem层（第一个残差块之前的层），包含一个卷积层、ReLU激活函数和最大池化层。
    这个结构用于初始特征提取和降采样。
    """

    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
                第一个卷积层后的归一化层，支持的格式见layers.get_norm函数
            in_channels (int): 输入通道数，默认为3（RGB图像）
            out_channels (int): 输出通道数，默认为64
        """
        # 调用父类初始化，总步长为4（2x2，来自卷积和池化）
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        # 7x7卷积层
        self.conv1 = Conv2d(
            in_channels,      # 输入通道数
            out_channels,     # 输出通道数
            kernel_size=7,    # 7x7卷积核
            stride=2,         # 步长2用于降采样
            padding=3,        # 填充3保持特征图大小合适
            bias=False,       # 不使用偏置项
            norm=get_norm(norm, out_channels),  # 添加归一化层
        )
        # 使用MSRA方法初始化卷积层权重
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)     # 7x7卷积
        x = F.relu_(x)        # ReLU激活（原地操作）
        # 3x3最大池化，步长2，填充1
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ResNet(Backbone):
    """
    Implement :paper:`ResNet`.
    实现ResNet网络架构，继承自Backbone基类。ResNet是一个深度残差网络，通过跳跃连接解决深层网络的退化问题。
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        Args:
            stem (nn.Module): a stem module
            # stem模块，通常是一个7x7卷积层或者多个3x3卷积层的组合，用于初始特征提取
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            # 网络的主体部分，通常包含4个阶段，每个阶段包含多个残差块
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            # 分类任务的类别数，如果为None则不执行分类任务，否则创建一个线性层用于分类
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            # 指定需要输出的特征层名称列表，可以是"stem"、"linear"或"res2"等，如果为None则返回最后一层的输出
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
            # 指定从开始需要冻结的阶段数，用于微调时固定部分参数
        """
        super().__init__()
        self.stem = stem  # 初始化stem层，用于初始特征提取
        self.num_classes = num_classes  # 存储分类类别数

        current_stride = self.stem.stride  # 获取stem层的步长
        self._out_feature_strides = {"stem": current_stride}  # 记录stem层的总步长
        self._out_feature_channels = {"stem": self.stem.out_channels}  # 记录stem层的输出通道数

        self.stage_names, self.stages = [], []  # 初始化存储stage名称和stage模块的列表

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            # 避免保留未使用的层，因为它们会消耗额外的内存并可能导致allreduce失败
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
            )  # 根据需要输出的特征层确定需要保留的stage数量
            stages = stages[:num_stages]  # 只保留需要的stage
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)  # 确保每个stage至少包含一个block
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block  # 确保每个block都是CNNBlockBase的实例

            name = "res" + str(i + 2)  # 生成stage的名称，从res2开始
            stage = nn.Sequential(*blocks)  # 将block组合成一个Sequential模块

            self.add_module(name, stage)  # 将stage添加到模型中
            self.stage_names.append(name)  # 记录stage名称
            self.stages.append(stage)  # 记录stage模块

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )  # 计算并记录每个stage的总步长
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels  # 记录每个stage的输出通道数
        self.stage_names = tuple(self.stage_names)  # 将stage名称列表转换为元组，使其不可修改

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 添加自适应平均池化层，将特征图压缩为1x1
            self.linear = nn.Linear(curr_channels, num_classes)  # 添加全连接层用于分类

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            # 根据论文中的建议，使用均值为0，标准差为0.01的正态分布初始化全连接层的权重
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"  # 设置全连接层的名称为"linear"

        if out_features is None:
            out_features = [name]  # 如果未指定输出特征，则使用最后一层作为输出
        self._out_features = out_features  # 存储需要输出的特征层名称
        assert len(self._out_features)  # 确保至少有一个输出特征
        children = [x[0] for x in self.named_children()]  # 获取所有子模块的名称
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))  # 确保指定的输出特征名称存在
        self.freeze(freeze_at)  # 根据freeze_at参数冻结指定数量的阶段

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
            # 输入张量，形状为(批量大小,通道数,高度,宽度)。高度和宽度必须是size_divisibility的倍数

        Returns:
            dict[str->Tensor]: names and the corresponding features
            # 返回一个字典，键为特征层名称，值为对应的特征图张量
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"  # 确保输入是4维张量
        outputs = {}  # 初始化输出字典
        x = self.stem(x)  # 通过stem层进行初始特征提取
        if "stem" in self._out_features:
            outputs["stem"] = x  # 如果需要stem层的输出，则保存到输出字典中
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)  # 依次通过每个stage进行特征提取
            if name in self._out_features:
                outputs[name] = x  # 如果当前stage的输出在需要输出的特征列表中，则保存到输出字典
        if self.num_classes is not None:
            x = self.avgpool(x)  # 通过平均池化层将特征图压缩为1x1
            x = torch.flatten(x, 1)  # 将特征展平
            x = self.linear(x)  # 通过全连接层进行分类
            if "linear" in self._out_features:
                outputs["linear"] = x  # 如果需要分类层的输出，则保存到输出字典中
        return outputs  # 返回包含所有需要的特征图的字典

    def output_shape(self):
        # 返回一个字典，包含每个输出特征层的通道数和总步长信息
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )  # 使用ShapeSpec封装每个特征层的信息，包括通道数和总步长
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.
        # 冻结ResNet的前几个阶段，通常用于微调模型

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()  # 如果freeze_at>=1，则冻结stem层
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()  # 冻结指定阶段的所有block
        return self  # 返回模型自身

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.
        # 创建一个ResNet阶段，由相同类型的block组成

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            # block类型，必须是CNNBlockBase的子类，除非stride!=1，否则不能改变输入的空间分辨率
            num_blocks (int): number of blocks in this stage
            # 该阶段包含的block数量
            in_channels (int): input channels of the entire stage.
            # 整个阶段的输入通道数
            out_channels (int): output channels of **every block** in the stage.
            # 该阶段中每个block的输出通道数
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.
            # 传递给block_class构造函数的其他参数。如果参数名是"xx_per_block"，则该参数是一个列表，
            # 列表中的值会分别传递给每个block；否则，同一个参数值会传递给所有block

        Returns:
            list[CNNBlockBase]: a list of block module.
            # 返回一个包含所有block模块的列表

        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        # 通常，产生相同特征图空间大小的层被定义为一个"stage"（在FPN论文中）。
        # 在这种定义下，除第一个block外，其他block的stride应该都是1
        """
        blocks = []  # 初始化blocks列表
        for i in range(num_blocks):
            curr_kwargs = {}  # 当前block的参数字典
            for k, v in kwargs.items():
                if k.endswith("_per_block"):  # 处理以_per_block结尾的参数
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )  # 确保_per_block参数的长度与block数量相同
                    newk = k[: -len("_per_block")]  # 去掉_per_block后缀
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"  # 确保不同时存在带和不带_per_block后缀的参数
                    curr_kwargs[newk] = v[i]  # 取出当前block对应的参数值
                else:
                    curr_kwargs[k] = v  # 非_per_block参数直接使用相同值

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )  # 创建block并添加到列表中
            in_channels = out_channels  # 更新输入通道数为当前block的输出通道数
        return blocks  # 返回创建的blocks列表

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.
        # 根据预定义的深度（18、34、50、101、152之一）创建ResNet的stages列表。
        # 如果需要创建其他变体，请使用make_stage方法进行自定义。

        Args:
            depth (int): depth of ResNet
            # ResNet的深度
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            # CNN block类型，当深度>50时必须接受bottleneck_channels参数。
            # 默认根据深度选择BasicBlock或BottleneckBlock
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.
            # 传递给make_stage的其他参数，不应包含stride和channels，因为这些参数已根据深度预定义

        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
            # 返回所有stage的模块列表，参见ResNet.__init__的参数说明
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],    # ResNet-18的每个stage的block数量
            34: [3, 4, 6, 3],    # ResNet-34的每个stage的block数量
            50: [3, 4, 6, 3],    # ResNet-50的每个stage的block数量
            101: [3, 4, 23, 3],  # ResNet-101的每个stage的block数量
            152: [3, 8, 36, 3],  # ResNet-152的每个stage的block数量
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock  # 根据深度选择block类型
        if depth < 50:
            in_channels = [64, 64, 128, 256]      # ResNet-18/34的每个stage的输入通道数
            out_channels = [64, 128, 256, 512]    # ResNet-18/34的每个stage的输出通道数
        else:
            in_channels = [64, 256, 512, 1024]    # ResNet-50/101/152的每个stage的输入通道数
            out_channels = [256, 512, 1024, 2048] # ResNet-50/101/152的每个stage的输出通道数
        ret = []  # 初始化返回列表
        for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4  # 对于深度>=50的网络，设置bottleneck通道数为输出通道数的1/4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,    # block类型
                    num_blocks=n,               # block数量
                    stride_per_block=[s] + [1] * (n - 1),  # 第一个block使用指定步长，其余block步长为1
                    in_channels=i,              # 输入通道数
                    out_channels=o,             # 输出通道数
                    **kwargs,                   # 其他参数
                )
            )
        return ret  # 返回所有stage的列表


ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
为了向后兼容性而设置的别名，将CNNBlockBase重命名为ResNetBlockBase。
"""


def make_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    已弃用的别名函数，为了保持向后兼容性。直接调用ResNet.make_stage方法。
    """
    return ResNet.make_stage(*args, **kwargs)


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.
    根据配置创建一个ResNet实例。

    Returns:
        ResNet: a :class:`ResNet` instance.
        返回一个ResNet类的实例。
    """
    # need registration of new blocks/stems?
    # 是否需要注册新的blocks或stems？
    norm = cfg.MODEL.RESNETS.NORM  # 获取归一化层的类型
    stem = BasicStem(
        in_channels=input_shape.channels,  # 输入通道数，来自输入图像的通道数
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,  # stem层的输出通道数
        norm=norm,  # 使用指定的归一化层
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT         # 指定从哪一层开始冻结参数
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES      # 指定需要输出的特征层
    depth               = cfg.MODEL.RESNETS.DEPTH             # ResNet的深度（18/34/50/101/152）
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS        # 组卷积的组数
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP   # 每组的通道数
    bottleneck_channels = num_groups * width_per_group        # Bottleneck块中间层的通道数
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS # 输入通道数（等于stem的输出通道数）
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS # res2阶段的输出通道数
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1     # 是否在1x1卷积中使用步长
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION     # res5阶段的膨胀率
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE # 每个阶段是否使用可变形卷积
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED    # 是否使用调制型可变形卷积
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS   # 可变形卷积的组数
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)  # 确保res5的膨胀率只能是1或2

    num_blocks_per_stage = {  # 不同深度的ResNet在每个stage中的block数量
        18: [2, 2, 2, 2],    # ResNet-18的每个stage有2个block
        34: [3, 4, 6, 3],    # ResNet-34的每个stage分别有3,4,6,3个block
        50: [3, 4, 6, 3],    # ResNet-50的每个stage分别有3,4,6,3个block
        101: [3, 4, 23, 3],  # ResNet-101的每个stage分别有3,4,23,3个block
        152: [3, 8, 36, 3],  # ResNet-152的每个stage分别有3,8,36,3个block
    }[depth]

    if depth in [18, 34]:  # 对于ResNet-18和ResNet-34的特殊配置检查
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"  # res2的输出通道必须是64
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"  # 不支持可变形卷积
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"  # res5的膨胀率必须是1
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"  # 组卷积的组数必须是1

    stages = []  # 存储所有stage的列表

    for idx, stage_idx in enumerate(range(2, 6)):  # 构建res2到res5四个stage
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        # 根据R-FCN和可变形卷积论文的约定设置res5的膨胀率
        dilation = res5_dilation if stage_idx == 5 else 1  # 只在res5阶段使用指定的膨胀率
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2  # 确定第一个block的步长
        stage_kargs = {  # 构建stage的参数字典
            "num_blocks": num_blocks_per_stage[idx],  # 当前stage的block数量
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),  # 每个block的步长
            "in_channels": in_channels,  # 输入通道数
            "out_channels": out_channels,  # 输出通道数
            "norm": norm,  # 归一化层类型
        }
        # Use BasicBlock for R18 and R34.
        # ResNet-18和ResNet-34使用BasicBlock
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock  # 使用基础残差块
        else:  # ResNet-50及以上使用BottleneckBlock
            stage_kargs["bottleneck_channels"] = bottleneck_channels  # 瓶颈层通道数
            stage_kargs["stride_in_1x1"] = stride_in_1x1  # 是否在1x1卷积中使用步长
            stage_kargs["dilation"] = dilation  # 当前stage的膨胀率
            stage_kargs["num_groups"] = num_groups  # 组卷积的组数
            if deform_on_per_stage[idx]:  # 如果当前stage启用可变形卷积
                stage_kargs["block_class"] = DeformBottleneckBlock  # 使用可变形瓶颈块
                stage_kargs["deform_modulated"] = deform_modulated  # 是否使用调制型可变形卷积
                stage_kargs["deform_num_groups"] = deform_num_groups  # 可变形卷积的组数
            else:
                stage_kargs["block_class"] = BottleneckBlock  # 使用标准瓶颈块
        blocks = ResNet.make_stage(**stage_kargs)  # 使用配置参数构建当前stage
        in_channels = out_channels  # 更新下一个stage的输入通道数
        out_channels *= 2  # 下一个stage的输出通道数翻倍
        bottleneck_channels *= 2  # 下一个stage的瓶颈层通道数翻倍
        stages.append(blocks)  # 将当前stage添加到stages列表
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)  # 构建并返回ResNet实例
