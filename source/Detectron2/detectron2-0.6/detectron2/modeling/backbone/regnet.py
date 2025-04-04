# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Implementation of RegNet models from :paper:`dds` and :paper:`scaling`.

This code is adapted from https://github.com/facebookresearch/pycls with minimal modifications.
Some code duplication exists between RegNet and ResNets (e.g., ResStem) in order to simplify
model loading.
"""

# 导入必要的库
import numpy as np
from torch import nn

# 从detectron2导入基础组件
from detectron2.layers import CNNBlockBase, ShapeSpec, get_norm

from .backbone import Backbone

# 定义模块的公开接口
__all__ = [
    "AnyNet",      # 通用网络架构
    "RegNet",      # RegNet网络
    "ResStem",     # ResNet风格的stem层
    "SimpleStem",  # 简单的stem层
    "VanillaBlock", # 基础卷积块
    "ResBasicBlock", # 基础残差块
    "ResBottleneckBlock", # 瓶颈残差块
]


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""  # 用于构建二维卷积层的辅助函数
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."  # 确保卷积核大小为奇数，以避免填充问题
    s, p, g, b = stride, (k - 1) // 2, groups, bias  # 设置步长、填充、分组和偏置参数
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)  # 返回配置好的卷积层


def gap2d():
    """Helper for building a global average pooling layer."""  # 用于构建全局平均池化层的辅助函数
    return nn.AdaptiveAvgPool2d((1, 1))  # 返回自适应全局平均池化层，输出大小为1x1


def pool2d(k, *, stride=1):
    """Helper for building a pool2d layer."""  # 用于构建二维池化层的辅助函数
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."  # 确保池化核大小为奇数
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)  # 返回最大池化层


def init_weights(m):
    """Performs ResNet-style weight initialization."""  # 执行ResNet风格的权重初始化
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN  # 由于使用BN层，不需要偏置项
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # 计算扇出
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))  # 使用He初始化
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)  # BN层的gamma参数初始化为1
        m.bias.data.zero_()      # BN层的beta参数初始化为0
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)  # 线性层权重初始化
        m.bias.data.zero_()                        # 线性层偏置初始化为0


class ResStem(CNNBlockBase):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""  # ResNet的stem层：7x7卷积、BN、激活函数、最大池化

    def __init__(self, w_in, w_out, norm, activation_class):
        super().__init__(w_in, w_out, 4)  # 步长为4
        self.conv = conv2d(w_in, w_out, 7, stride=2)  # 7x7卷积，步长2
        self.bn = get_norm(norm, w_out)  # 批归一化层
        self.af = activation_class()  # 激活函数
        self.pool = pool2d(3, stride=2)  # 3x3最大池化，步长2

    def forward(self, x):
        for layer in self.children():  # 按顺序通过所有层
            x = layer(x)
        return x


class SimpleStem(CNNBlockBase):
    """Simple stem for ImageNet: 3x3, BN, AF."""  # 简单的stem层：3x3卷积、BN、激活函数

    def __init__(self, w_in, w_out, norm, activation_class):
        super().__init
        for i in range(d):
            block = block_class(w_in, w_out, stride, norm, activation_class, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(Backbone):
    """AnyNet model. See :paper:`dds`."""

    def __init__(
        self,
        *,
        stem_class,
        stem_width,
        block_class,
        depths,
        widths,
        group_widths,
        strides,
        bottleneck_ratios,
        se_ratio,
        activation_class,
        freeze_at=0,
        norm="BN",
        out_features=None,
    ):
        """
        Args:
            stem_class (callable): A callable taking 4 arguments (channels in, channels out,
                normalization, callable returning an activation function) that returns another
                callable implementing the stem module.
            stem_width (int): The number of output channels that the stem produces.
            block_class (callable): A callable taking 6 arguments (channels in, channels out,
                stride, normalization, callable returning an activation function, a dict of
                block-specific parameters) that returns another callable implementing the repeated
                block module.
            depths (list[int]): Number of blocks in each stage.
            widths (list[int]): For each stage, the number of output channels of each block.
            group_widths (list[int]): For each stage, the number of channels per group in group
                convolution, if the block uses group convolution.
            strides (list[int]): The stride that each network stage applies to its input.
            bottleneck_ratios (list[float]): For each stage, the ratio of the number of bottleneck
                channels to the number of block input channels (or, equivalently, output channels),
                if the block uses a bottleneck.
            se_ratio (float): The ratio of the number of channels used inside the squeeze-excitation
                (SE) module to it number of input channels, if SE the block uses SE.
            activation_class (callable): A callable taking no arguments that returns another
                callable implementing an activation function.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. RegNet's use "stem" and "s1", "s2", etc for the stages after
                the stem. If None, will return the output of the last layer.
        """
        super().__init__()
        self.stem = stem_class(3, stem_width, norm, activation_class)

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}
        self.stages_and_names = []
        prev_w = stem_width

        for i, (d, w, s, b, g) in enumerate(
            zip(depths, widths, strides, bottleneck_ratios, group_widths)
        ):
            params = {"bot_mul": b, "group_w": g, "se_r": se_ratio}
            stage = AnyStage(prev_w, w, s, d, block_class, norm, activation_class, params)
            name = "s{}".format(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in stage.children()])
            )
            self._out_feature_channels[name] = list(stage.children())[-1].out_channels
            prev_w = w

        self.apply(init_weights)

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {} does not include {}".format(
                ", ".join(children), out_feature
            )
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
                # 输入张量，形状为(批次大小,通道数,高度,宽度)。高度和宽度必须是self.size_divisibility的倍数

        Returns:
            dict[str->Tensor]: names and the corresponding features
                # 返回一个字典，键为特征层名称，值为对应的特征张量
        """
        assert x.dim() == 4, f"Model takes an input of shape (N, C, H, W). Got {x.shape} instead!"  # 确保输入是4维张量
        outputs = {}  # 初始化输出字典
        x = self.stem(x)  # 通过stem层处理输入
        if "stem" in self._out_features:  # 如果stem层在输出特征列表中
            outputs["stem"] = x  # 将stem层的输出添加到输出字典
        for stage, name in self.stages_and_names:  # 依次通过每个阶段
            x = stage(x)  # 执行当前阶段的前向传播
            if name in self._out_features:  # 如果当前阶段在输出特征列表中
                outputs[name] = x  # 将当前阶段的输出添加到输出字典
        return outputs  # 返回包含所有指定特征层输出的字典

    def output_shape(self):
        # 返回每个输出特征层的形状规格
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],  # 特征层的通道数
                stride=self._out_feature_strides[name]  # 特征层的总步长
            )
            for name in self._out_features  # 遍历所有输出特征层
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the model. Commonly used in fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this model itself
        """
        # 冻结模型的前几个阶段，常用于微调
        if freeze_at >= 1:
            self.stem.freeze()  # 冻结stem层
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:  # 根据freeze_at参数冻结对应的阶段
                for block in stage.children():
                    block.freeze()
        return self


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""  # 调整宽度、瓶颈比和分组数的兼容性
    assert len(ws) == len(bs) == len(gs)  # 确保三个列表长度相同
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))  # 确保所有值都为正
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]  # 计算瓶颈层通道数
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]  # 确保分组数不超过通道数
    ms = [np.lcm(g, b) if b > 1 else g for g, b in zip(gs, bs)]  # 计算最小公倍数
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]  # 调整通道数为最小公倍数的整数倍
    ws = [int(v / b) for v, b in zip(vs, bs)]  # 计算调整后的宽度
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))  # 确保通道数能被分组数整除
    return ws, bs, gs


def generate_regnet_parameters(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""  # 根据RegNet参数生成每个阶段的宽度和深度
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0  # 参数有效性检查
    # Generate continuous per-block ws  # 生成连续的每个块的宽度
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws  # 生成量化后的每个块的宽度
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))  # 计算量化指数
    ws_all = w_0 * np.power(w_m, ks)  # 计算所有宽度
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q  # 量化为q的整数倍
    # Generate per stage ws and ds (assumes ws_all are sorted)  # 生成每个阶段的宽度和深度
    ws, ds = np.unique(ws_all, return_counts=True)  # 统计不同宽度的数量
    # Compute number of actual stages and total possible stages  # 计算实际阶段数和可能的总阶段数
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return  # 将numpy数组转换为列表并返回
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


class RegNet(AnyNet):
    """RegNet model. See :paper:`dds`."""  # RegNet模型，参见论文dds

    def __init__(
        self,
        *,
        stem_class,  # stem层类型
        stem_width,  # stem层输出通道数
        block_class,  # 基本块类型
        depth,       # 网络总块数
        w_a,         # 块宽度增长因子
        w_0,         # 初始块宽度
        w_m,         # 宽度量化参数
        group_width, # 分组卷积的每组通道数
        stride=2,    # 每个网络阶段的步长
        bottleneck_ratio=1.0,  # 瓶颈比例
        se_ratio=0.0,          # SE模块的通道压缩比例
        activation_class=None,  # 激活函数类
        freeze_at=0,           # 冻结的阶段数
        norm="BN",             # 归一化层类型
        out_features=None,      # 输出特征层名称
    ):
        """
        Build a RegNet from the parameterization described in :paper:`dds` Section 3.3.

        Args:
            See :class:`AnyNet` for arguments that are not listed here.
            depth (int): Total number of blocks in the RegNet.
            w_a (float): Factor by which block width would increase prior to quantizing block widths
                by stage. See :paper:`dds` Section 3.3.
            w_0 (int): Initial block width. See :paper:`dds` Section 3.3.
            w_m (float): Parameter controlling block width quantization.
                See :paper:`dds` Section 3.3.
            group_width (int): Number of channels per group in group convolution, if the block uses
                group convolution.
            bottleneck_ratio (float): The ratio of the number of bottleneck channels to the number
                of block input channels (or, equivalently, output channels), if the block uses a
                bottleneck.
            stride (int): The stride that each network stage applies to its input.
        """
        # 根据RegNet参数生成每个阶段的宽度和深度
        ws, ds = generate_regnet_parameters(w_a, w_0, w_m, depth)[0:2]
        ss = [stride for _ in ws]  # 每个阶段的步长列表
        bs = [bottleneck_ratio for _ in ws]  # 每个阶段的瓶颈比例列表
        gs = [group_width for _ in ws]  # 每个阶段的分组宽度列表
        ws, bs, gs = adjust_block_compatibility(ws, bs, gs)  # 调整参数兼容性

        def default_activation_class():  # 默认使用ReLU激活函数
            return nn.ReLU(inplace=True)

        # 调用父类AnyNet的初始化方法
        super().__init__(
            stem_class=stem_class,
            stem_width=stem_width,
            block_class=block_class,
            depths=ds,
            widths=ws,
            strides=ss,
            group_widths=gs,
            bottleneck_ratios=bs,
            se_ratio=se_ratio,
            activation_class=default_activation_class
            if activation_class is None
            else activation_class,
            freeze_at=freeze_at,
            norm=norm,
            out_features=out_features,
        )
