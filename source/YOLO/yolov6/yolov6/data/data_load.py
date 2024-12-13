#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import os
import torch.distributed as dist  # 导入PyTorch分布式训练模块
from torch.utils.data import dataloader, distributed  # 导入数据加载器和分布式训练相关组件

from .datasets import TrainValDataset  # 导入训练和验证数据集类
from yolov6.utils.events import LOGGER  # 导入日志记录器
from yolov6.utils.torch_utils import torch_distributed_zero_first  # 导入分布式训练工具函数


def create_dataloader(
    path,  # 数据集路径
    img_size,  # 图像尺寸
    batch_size,  # 批次大小
    stride,  # 模型步长
    hyp=None,  # 超参数字典
    augment=False,  # 是否使用数据增强
    check_images=False,  # 是否检查图像
    check_labels=False,  # 是否检查标签
    pad=0.0,  # 填充比例
    rect=False,  # 是否使用矩形训练
    rank=-1,  # 分布式训练的进程序号
    workers=8,  # 数据加载的工作进程数
    shuffle=False,  # 是否打乱数据
    data_dict=None,  # 数据配置字典
    task="Train",  # 任务类型（训练/验证）
    specific_shape=False,  # 是否使用指定形状
    height=1088,  # 指定高度
    width=1920,  # 指定宽度
    cache_ram=False  # 是否缓存到内存
    ):
    """Create general dataloader.  # 创建通用数据加载器

    Returns dataloader and dataset  # 返回数据加载器和数据集
    """
    if rect and shuffle:
        # 矩形训练模式下不能打乱数据，因为需要按照高宽比对图片进行分组
        LOGGER.warning(
            "WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False"
        )
        shuffle = False
    with torch_distributed_zero_first(rank):  # 确保在分布式训练中只有一个进程执行数据集初始化
        dataset = TrainValDataset(  # 创建训练验证数据集实例
            path,
            img_size,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            check_images=check_images,
            check_labels=check_labels,
            stride=int(stride),
            pad=pad,
            rank=rank,
            data_dict=data_dict,
            task=task,
            specific_shape = specific_shape,
            height=height,
            width=width,
            cache_ram=cache_ram
        )

    batch_size = min(batch_size, len(dataset))  # 确保batch_size不超过数据集大小
    workers = min(  # 计算实际的工作进程数
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),  # CPU核心数除以总进程数
            batch_size if batch_size > 1 else 0,  # batch_size大于1时才使用多进程
            workers,  # 用户指定的工作进程数
        ]
    )  # number of workers  # 工作进程数

    # in DDP mode, if GPU number is greater than 1, and set rect=True,
    # DistributedSampler will sample from start if the last samples cannot be assigned equally to each
    # GPU process, this might cause shape difference in one batch, such as (384,640,3) and (416,640,3)
    # will cause exception in collate function of torch.stack.
    # 在分布式训练模式下，如果使用多GPU且启用矩形训练，最后一个批次可能会出现形状不一致的问题
    drop_last = rect and dist.is_initialized() and dist.get_world_size() > 1  # 是否丢弃最后一个不完整的批次
    sampler = (  # 创建数据采样器
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    )
    return (  # 返回数据加载器和数据集
        TrainValDataLoader(  # 创建训练验证数据加载器
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,  # 只有在非分布式训练且指定shuffle时才打乱数据
            num_workers=workers,  # 工作进程数
            sampler=sampler,  # 数据采样器
            pin_memory=True,  # 将数据放入固定内存中，加快GPU读取速度
            collate_fn=TrainValDataset.collate_fn,  # 数据批处理函数
        ),
        dataset,  # 返回数据集对象
    )


class TrainValDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers  # 复用工作进程的数据加载器

    Uses same syntax as vanilla DataLoader  # 使用与原生DataLoader相同的语法
    """

    def __init__(self, *args, **kwargs):
        # 初始化父类DataLoader
        super().__init__(*args, **kwargs)
        # 使用object.__setattr__直接设置属性，避免触发__setattr__方法
        # 将原有的batch_sampler替换为_RepeatSampler的实例
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        # 获取父类的迭代器
        self.iterator = super().__iter__()

    def __len__(self):
        # 返回采样器的长度（数据集大小）
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        # 自定义迭代器，每次迭代都从父类迭代器中获取下一个批次
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever  # 永久重复的采样器

    Args:
        sampler (Sampler)  # 输入参数：基础采样器
    """

    def __init__(self, sampler):
        # 保存基础采样器实例
        self.sampler = sampler

    def __iter__(self):
        # 创建一个无限循环的迭代器
        # 每次迭代都重新获取基础采样器的迭代器
        # yield from语法将迭代器的所有值依次产出
        while True:
            yield from iter(self.sampler)
