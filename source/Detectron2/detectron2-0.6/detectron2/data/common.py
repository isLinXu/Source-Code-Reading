# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "ToIterableDataset"]


def _shard_iterator_dataloader_worker(iterable):
    # 在多worker数据加载环境中分片数据迭代器
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # 单worker情况直接返回完整迭代器
        # do nothing
        yield from iterable
    else:
        # 多worker情况使用islice进行数据分片
        yield from itertools.islice(iterable, worker_info.id, None, worker_info.num_workers)


class _MapIterableDataset(data.IterableDataset):
    """
    Map a function over elements in an IterableDataset.

    Similar to pytorch's MapIterDataPipe, but support filtering when map_func
    returns None.

    This class is not public-facing. Will be called by `MapDataset`.
    可迭代数据集映射类（内部实现）：
    - 支持在map过程中过滤返回None的数据
    - 保持与原始数据集相同的长度
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset # 原始可迭代数据集
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work # 可序列化的映射函数包装器

    def __len__(self):
        return len(self._dataset) # 原始可迭代数据集

    def __iter__(self):
        # 应用映射函数并过滤None结果
        for x in map(self._map_func, self._dataset):
            if x is not None:
                yield x


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    通用数据集映射类核心实现：
    - 同时支持map-style和iterable数据集
    - 提供数据映射失败的重试机制
    - 自动处理不同类型数据集的差异
    """

    def __init__(self, dataset, map_func):
        """
        Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next.
        """
        self._dataset = dataset # 原始数据集（map或iterable类型）
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work # 可序列化映射函数

        self._rng = random.Random(42) # 固定随机种子用于确定性重试
        self._fallback_candidates = set(range(len(dataset))) # 备用索引池

    def __new__(cls, dataset, map_func):
        # 动态创建不同类型的实例
        is_iterable = isinstance(dataset, data.IterableDataset)
        if is_iterable:
            return _MapIterableDataset(dataset, map_func) # 返回可迭代版本
        else:
            return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)
        # 重试机制（最多3次）
        while True:
            data = self._map_func(self._dataset[cur_idx]) # 应用映射函数
            if data is not None:
                self._fallback_candidates.add(cur_idx)    # 成功则保留索引
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    列表数据集包装器核心功能：
    - 支持内存优化序列化
    - 可选择深拷贝保护原始数据
    - 高效的大型数据集存储
    """

    def __init__(self, lst: list, copy: bool = True, serialize: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self._lst = lst
        self._copy = copy  # 是否深拷贝数据
        self._serialize = serialize  # 是否序列化存储

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            # 序列化优化处理
            logger = logging.getLogger(__name__)
            logger.info(
                "Serializing {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst) # 计算偏移地址
                )
            )
            self._lst = [_serialize(x) for x in self._lst] # 拼接为连续内存
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        # # 反序列化处理
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx]) # 安全返回副本
        else:
            return self._lst[idx]                # 直接返回引用


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    迭代式数据集转换器：
    - 将map-style数据集转换为可迭代式
    - 支持分布式worker分片
    - 保持与原始采样器一致的顺序
    """

    def __init__(self, dataset: data.Dataset, sampler: Sampler, shard_sampler: bool = True):
        """
        Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler  # 采样器实例
        self.shard_sampler = shard_sampler # 是否自动分片

    def __iter__(self):
        # 分布式worker分片逻辑
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = _shard_iterator_dataloader_worker(self.sampler)
        for idx in sampler:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler) # 按采样顺序生成数据


class AspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    宽高比分组数据集：
    - 将图像分为宽>高和宽<高两组
    - 动态批量组合减少padding
    - 优化GPU内存利用率
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)] # 初始化两个分组桶
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1   # 计算分组ID
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size: # 桶满时生成批次
                yield bucket[:]
                del bucket[:]
