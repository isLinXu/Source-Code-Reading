# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import division
from typing import Any, List, Tuple
import torch
from torch import device
from torch.nn import functional as F

from detectron2.layers.wrappers import shapes_to_tensor


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    图像列表类，将不同尺寸的图像填充为统一尺寸的tensor，并记录原始尺寸
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): 填充后的tensor，形状为(N, H, W)或(N, C_1, ..., C_K, H, W)
            image_sizes (list[tuple[int, int]]): 原始图像尺寸列表，每个元素为(h, w)
        """
        self.tensor = tensor  # 存储填充后的tensor
        self.image_sizes = image_sizes  # 存储原始图像尺寸

    def __len__(self) -> int:
        return len(self.image_sizes)  # 返回图像数量

    def __getitem__(self, idx) -> torch.Tensor:
        """
        按原始尺寸获取图像（自动去除填充部分）
        """
        size = self.image_sizes[idx]  # 获取指定索引的原始尺寸
        return self.tensor[idx, ..., : size[0], : size[1]]  # 切片去除填充部分

    @torch.jit.unused  # 禁用TorchScript编译
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        """转移设备"""
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> device:
        return self.tensor.device  # 返回所在设备

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        """
        从多个tensor创建ImageList，自动填充到统一尺寸
        Args:
            tensors: 输入图像列表，每个tensor形状为(Hi, Wi)或(C_1,...,C_K, Hi, Wi)
            size_divisibility: 尺寸对齐基数（如32），确保填充后的尺寸能被该数整除
            pad_value: 填充值
        """
        assert len(tensors) > 0  # 输入不能为空
        assert isinstance(tensors, (tuple, list))  # 检查输入类型
        for t in tensors:
            assert t.shape[:-2] == tensors[0].shape[:-2], "通道维度必须一致"  # 验证通道数一致

        # 收集原始尺寸信息
        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]  # 提取各图高宽
        image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]  # 转为tensor列表
        max_size = torch.stack(image_sizes_tensor).max(0).values  # 计算最大高宽

        # 处理尺寸对齐要求
        if size_divisibility > 1:
            stride = size_divisibility
            # 计算对齐后的尺寸：(当前尺寸 + 基数-1) // 基数 * 基数
            max_size = (max_size + (stride - 1)) // stride * stride

        # 处理TorchScript的追踪模式
        if torch.jit.is_scripting():
            max_size = max_size.to(dtype=torch.long).tolist()  # 转为整数列表
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor  # 追踪模式下保持tensor格式

        # 填充图像到统一尺寸
        if len(tensors) == 1:
            # 单图优化路径：直接填充并增加批次维度
            image_size = image_sizes[0]
            # 计算各边填充量：[左，右，上，下]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # 多图处理：创建填充模板并复制数据
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)  # 创建填充模板
            for img, pad_img in zip(tensors, batched_imgs):
                # 将原始图像复制到填充模板的对应位置
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)  # 返回连续内存的ImageList
