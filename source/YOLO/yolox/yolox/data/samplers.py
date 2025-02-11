#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler

class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.mosaic = mosaic  # 设置是否启用马赛克增强

    def __iter__(self):
        for batch in super().__iter__():  # 从父类的迭代器中获取批次
            yield [(self.mosaic, idx) for idx in batch]  # 生成 (mosaic, index) 元组


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from  # 要采样的基础数据集的总数量
            shuffle (bool): whether to shuffle the indices or not  # 是否打乱索引
            seed (int): the initial seed of the shuffle. Must be the same  # 随机打乱的初始种子，所有工作进程必须相同
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size  # 设置数据集大小
        assert size > 0  # 确保数据集大小大于 0
        self._shuffle = shuffle  # 设置是否打乱索引
        self._seed = int(seed)  # 设置种子

        if dist.is_available() and dist.is_initialized():  # 如果分布式训练可用并已初始化
            self._rank = dist.get_rank()  # 获取当前进程的排名
            self._world_size = dist.get_world_size()  # 获取总进程数
        else:
            self._rank = rank  # 设置排名
            self._world_size = world_size  # 设置总进程数

    def __iter__(self):
        start = self._rank  # 获取当前进程的起始索引
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )  # 生成无限索引的切片

    def _infinite_indices(self):
        g = torch.Generator()  # 创建随机数生成器
        g.manual_seed(self._seed)  # 设置生成器的种子
        while True:  # 无限循环
            if self._shuffle:  # 如果需要打乱
                yield from torch.randperm(self._size, generator=g)  # 生成随机排列的索引
            else:
                yield from torch.arange(self._size)  # 生成顺序索引

    def __len__(self):
        return self._size // self._world_size  # 返回每个进程的样本数量