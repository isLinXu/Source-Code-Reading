# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    包装另一个采样器以生成一个小批量的索引。
    它确保批次中只包含来自同一组的元素。
    同时，它尽可能地保持生成的小批量的顺序与原始采样器的顺序接近。
    """

    def __init__(self, sampler, group_ids, batch_size):
        """
        Args:
            sampler (Sampler): Base sampler.
                基础采样器。
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
                如果采样器生成范围在[0, N)内的索引，
                `group_ids`必须是一个包含N个整数的列表，其中包含每个样本的组ID。
                组ID必须是在[0, num_groups)范围内的一组整数。
            batch_size (int): Size of mini-batch.
                小批量的大小。
        """
        # 验证sampler参数是否为Sampler类的实例
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler  # 保存基础采样器
        self.group_ids = np.asarray(group_ids)  # 将组ID列表转换为numpy数组
        assert self.group_ids.ndim == 1  # 确保组ID数组是一维的
        self.batch_size = batch_size  # 保存批量大小
        groups = np.unique(self.group_ids).tolist()  # 获取所有唯一的组ID

        # buffer the indices of each group until batch size is reached
        # 为每个组创建一个缓冲区，用于存储索引直到达到批量大小
        self.buffer_per_group = {k: [] for k in groups}

    def __iter__(self):
        # 遍历基础采样器生成的索引
        for idx in self.sampler:
            group_id = self.group_ids[idx]  # 获取当前索引对应的组ID
            group_buffer = self.buffer_per_group[group_id]  # 获取该组的缓冲区
            group_buffer.append(idx)  # 将索引添加到对应组的缓冲区
            if len(group_buffer) == self.batch_size:  # 当缓冲区达到批量大小时
                yield group_buffer[:]  # yield a copy of the list  # 生成该组的一个批次（返回列表的副本）
                del group_buffer[:]  # 清空缓冲区

    def __len__(self):
        # 由于不同组的样本数量可能不同，且可能有些组无法凑够一个完整的批次
        # 因此GroupedBatchSampler的长度（总批次数）无法准确定义
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")
