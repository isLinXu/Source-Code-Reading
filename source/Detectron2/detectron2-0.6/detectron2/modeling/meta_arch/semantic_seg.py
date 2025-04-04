# Copyright (c) Facebook, Inc. and its affiliates.
# 导入必要的库
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union  # 导入类型注解相关的工具
import fvcore.nn.weight_init as weight_init  # 导入权重初始化工具
import torch
from torch import nn
from torch.nn import functional as F

# 导入detectron2相关的模块
from detectron2.config import configurable  # 导入配置相关的工具
from detectron2.layers import Conv2d, ShapeSpec, get_norm  # 导入基础网络层
from detectron2.structures import ImageList  # 导入图像列表数据结构
from detectron2.utils.registry import Registry  # 导入注册表工具

# 导入本地模块
from ..backbone import Backbone, build_backbone  # 导入主干网络相关的模块
from ..postprocessing import sem_seg_postprocess  # 导入语义分割后处理模块
from .build import META_ARCH_REGISTRY  # 导入元架构注册表

# 定义可以被外部导入的类和函数
__all__ = [
    "SemanticSegmentor",
    "SEM_SEG_HEADS_REGISTRY",
    "SemSegFPNHead",
    "build_sem_seg_head",
]


# 创建语义分割头部的注册表
SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
SEM_SEG_HEADS_REGISTRY.__doc__ = """
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
语义分割头部的注册表，用于从特征图生成语义分割预测。
"""


@META_ARCH_REGISTRY.register()  # 注册SemanticSegmentor类到元架构注册表
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    语义分割架构的主类。
    """

    @configurable  # 标记该方法为可配置的
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        参数说明：
            backbone: 骨干网络模块，必须遵循detectron2的骨干网络接口
            sem_seg_head: 从骨干网络特征预测语义分割的模块
            pixel_mean, pixel_std: 列表或元组，包含通道数量的元素，表示用于归一化输入图像的每个通道的均值和标准差
        """
        super().__init__()  # 调用父类的初始化方法
        self.backbone = backbone  # 设置骨干网络
        self.sem_seg_head = sem_seg_head  # 设置语义分割头部
        # 注册像素均值和标准差为模型的缓冲区
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):  # 从配置创建模型实例的类方法
        backbone = build_backbone(cfg)  # 构建骨干网络
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())  # 构建语义分割头部
        return {
            "backbone": backbone,  # 返回骨干网络
            "sem_seg_head": sem_seg_head,  # 返回语义分割头部
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,  # 返回像素均值
            "pixel_std": cfg.MODEL.PIXEL_STD,  # 返回像素标准差
        }

    @property
    def device(self):  # 获取模型所在设备的属性方法
        return self.pixel_mean.device  # 返回像素均值张量所在的设备

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.


        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        参数：
            batched_inputs: 一个列表，包含:class:`DatasetMapper`的批处理输出。
                列表中的每个元素包含一张图像的输入数据。

                目前，列表中的每个元素是一个字典，包含：

                   * "image": 张量，格式为(C, H, W)的图像数据
                   * "sem_seg": 语义分割的真实标注
                   * 原始字典中包含的其他信息，例如：
                     "height", "width" (int): 模型的输出分辨率（可能与输入分辨率不同），
                     用于推理阶段。

        返回值：
            list[dict]：
              每个字典是一张输入图像的输出结果。
              字典包含一个键"sem_seg"，其值是一个张量，表示
              头部预测的每个像素的分割结果。
              预测结果的形状为KxHxW，表示每个像素对应每个类别的logits值。
        """
        images = [x["image"].to(self.device) for x in batched_inputs]  # 将输入图像转移到指定设备
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]  # 对图像进行标准化处理
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)  # 将图像转换为ImageList格式

        features = self.backbone(images.tensor)  # 通过骨干网络提取特征

        if "sem_seg" in batched_inputs[0]:  # 如果输入数据包含语义分割标注
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]  # 将标注数据转移到指定设备
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor  # 将标注转换为张量格式
        else:
            targets = None  # 如果没有标注数据，则设为None
        results, losses = self.sem_seg_head(features, targets)  # 通过语义分割头部进行预测

        if self.training:  # 如果是训练模式
            return losses  # 返回损失

        processed_results = []  # 存储处理后的结果
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])  # 获取输出高度
            width = input_per_image.get("width", image_size[1])  # 获取输出宽度
            r = sem_seg_postprocess(result, image_size, height, width)  # 对预测结果进行后处理
            processed_results.append({"sem_seg": r})  # 将处理后的结果添加到列表
        return processed_results  # 返回处理后的结果


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    从配置文件构建语义分割头部。
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME  # 获取语义分割头部的名称
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)  # 从注册表中获取并构建对应的头部模块


@SEM_SEG_HEADS_REGISTRY.register()  # 注册SemSegFPNHead类到语义分割头部注册表
class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`.
    It takes a list of FPN features as input, and applies a sequence of
    3x3 convs and upsampling to scale all of them to the stride defined by
    ``common_stride``. Then these features are added and used to make final
    predictions by another 1x1 conv layer.
    在PanopticFPN论文中描述的语义分割头部。
    它接收一系列FPN特征作为输入，并应用一系列3x3卷积和上采样操作，将所有特征缩放到
    由common_stride定义的步长。然后将这些特征相加，并通过另一个1x1卷积层进行最终预测。
    """

    @configurable  # 标记该方法为可配置的
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        注意：这个接口是实验性的。

        参数：
            input_shape: 输入特征的形状（通道数和步长）
            num_classes: 需要预测的类别数量
            conv_dims: 中间卷积层的输出通道数
            common_stride: 所有特征将被上采样到的公共步长
            loss_weight: 损失权重
            norm: 所有卷积层的归一化方法
            ignore_value: 训练时需要忽略的类别ID
        """
        # 调用父类的初始化方法
        super().__init__()
        # 根据stride对输入特征形状进行排序
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # 确保输入特征不为空
        if not len(input_shape):
            raise ValueError("SemSegFPNHead(input_shape=) cannot be empty!")
        # 提取特征名称、步长和通道数
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        # 设置忽略值、公共步长和损失权重
        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight

        # 初始化尺度头部列表
        self.scale_heads = []
        # 为每个输入特征创建对应的尺度头部
        for in_feature, stride, channels in zip(
            self.in_features, feature_strides, feature_channels
        ):
            head_ops = []
            # 计算需要的上采样次数，确保最终特征图达到目标步长
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            # 构建每个尺度头部的操作序列
            for k in range(head_length):
                # 获取归一化模块
                norm_module = get_norm(norm, conv_dims)
                # 创建3x3卷积层
                conv = Conv2d(
                    channels if k == 0 else conv_dims,  # 第一层使用输入通道数，之后使用统一的通道数
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,  # 如果使用归一化，则不使用偏置
                    norm=norm_module,
                    activation=F.relu,
                )
                # 使用MSRA初始化卷积权重
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                # 如果当前步长大于目标步长，添加2倍上采样层
                if stride != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            # 将操作序列组合成一个Sequential模块
            self.scale_heads.append(nn.Sequential(*head_ops))
            # 将尺度头部添加为模块的子模块
            self.add_module(in_feature, self.scale_heads[-1])
        # 创建最终的预测器，使用1x1卷积输出类别预测
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # 从配置文件中构建模型参数
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
        }

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        返回值：
            在训练时，返回(None, 损失字典)
            在推理时，返回(CxHxW的logits预测值, 空字典)
        """
        # 通过特征提取层获取特征
        x = self.layers(features)
        if self.training:
            # 训练模式：计算损失
            return None, self.losses(x, targets)
        else:
            # 推理模式：对特征图进行上采样
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):
        # 融合不同尺度的特征
        for i, f in enumerate(self.in_features):
            if i == 0:
                # 第一个特征直接通过对应的尺度头部
                x = self.scale_heads[i](features[f])
            else:
                # 其他特征经过尺度头部处理后与之前的特征相加
                x = x + self.scale_heads[i](features[f])
        # 通过预测器生成最终的分割预测
        x = self.predictor(x)
        return x

    def losses(self, predictions, targets):
        # 将预测转换为float类型以避免数值问题
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        # 对预测结果进行上采样，使其与目标大小一致
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        # 计算交叉熵损失
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        # 返回加权后的损失字典
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
