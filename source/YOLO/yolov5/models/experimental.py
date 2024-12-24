# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Experimental modules."""

import math  # 导入数学库

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

from utils.downloads import attempt_download  # 从utils下载模块导入attempt_download函数


class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""
    # 加权求和2个或更多层的输出

    def __init__(self, n, weight=False):
        """Initializes a module to sum outputs of layers with number of inputs `n` and optional weighting, supporting 2+
        inputs.
        """
        # 初始化一个模块，用于对具有输入数量`n`的层的输出进行求和，并可选择加权，支持2个以上的输入
        super().__init__()  # 调用父类初始化方法
        self.weight = weight  # apply weights boolean
        # 是否应用权重的布尔值
        self.iter = range(n - 1)  # iter object
        # 创建迭代对象
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights
            # 如果需要权重，则创建可训练的权重参数

    def forward(self, x):
        """Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights."""
        # 通过可定制的加权和处理输入，支持`n`个输入，并可选择应用学习到的权重
        y = x[0]  # no weight
        # 初始化输出为第一个输入
        if self.weight:
            w = torch.sigmoid(self.w) * 2  # 应用sigmoid激活函数并乘以2
            for i in self.iter:
                y = y + x[i + 1] * w[i]  # 加权求和
        else:
            for i in self.iter:
                y = y + x[i + 1]  # 直接求和
        return y  # 返回求和结果


class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""
    # 混合深度卷积

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Initializes MixConv2d with mixed depth-wise convolutional layers, taking input and output channels (c1, c2),
        kernel sizes (k), stride (s), and channel distribution strategy (equal_ch).
        """
        # 初始化MixConv2d，使用混合深度卷积层，接受输入和输出通道（c1, c2）、卷积核大小（k）、步幅（s）和通道分配策略（equal_ch）
        super().__init__()  # 调用父类初始化方法
        n = len(k)  # number of convolutions
        # 卷积的数量
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            # 生成均匀分布的通道索引
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
            # 计算每组的中间通道数
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n  # 初始化b
            a = np.eye(n + 1, n, k=-1)  # 创建单位矩阵
            a -= np.roll(a, 1, axis=1)  # 计算差分
            a *= np.array(k) ** 2  # 根据卷积核大小调整
            a[0] = 1  # 设置第一行
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b
            # 通过最小二乘法求解每组的通道数

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
            # 创建卷积层列表
        )
        self.bn = nn.BatchNorm2d(c2)  # 添加批归一化层
        self.act = nn.SiLU()  # 添加SiLU激活函数

    def forward(self, x):
        """Performs forward pass by applying SiLU activation on batch-normalized concatenated convolutional layer
        outputs.
        """
        # 执行前向传播，对批归一化后的卷积层输出应用SiLU激活
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))
        # 将所有卷积层的输出拼接在一起，并进行激活


class Ensemble(nn.ModuleList):
    """Ensemble of models."""
    # 模型的集成

    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        # 初始化一个模型集成，用于聚合预测
        super().__init__()  # 调用父类初始化方法

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        # 执行前向传播，聚合来自模型集成的输出
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # 遍历每个模块，获取输出
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        # 将所有输出在通道维度上拼接
        return y, None  # inference, train output
        # 返回推理结果和训练输出


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    # 从权重加载并融合一个或多个YOLOv5模型，处理设备放置和模型调整
    from models.yolo import Detect, Model  # 从yolo模块导入Detect和Model类

    model = Ensemble()  # 创建模型集成
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        # 加载权重
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model
        # 获取模型并转换为FP32格式

        # Model compatibility updates
        # 模型兼容性更新
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])  # 设置默认步幅
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
            # 将类别名称转换为字典

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode
        # 将模型添加到集成中，设置为评估模式

    # Module updates
    # 模块更新
    for m in model.modules():
        t = type(m)  # 获取模块类型
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # 设置是否就地操作
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")  # 删除anchor_grid属性
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)  # 重新设置anchor_grid
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            # 处理向上采样的兼容性

    # Return model
    # 返回模型
    if len(model) == 1:
        return model[-1]  # 如果只有一个模型，直接返回

    # Return detection ensemble
    # 返回检测模型集成
    print(f"Ensemble created with {weights}\n")  # 打印创建的模型集成信息
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))  # 从第一个模型获取属性并设置到集成模型
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    # 设置最大步幅
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    # 确保所有模型的类别数量相同
    return model  # 返回模型集成

