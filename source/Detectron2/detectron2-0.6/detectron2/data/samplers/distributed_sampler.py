# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler

from detectron2.utils import comm  # 导入通信模块，用于分布式训练中的进程间通信

logger = logging.getLogger(__name__)  # 创建日志记录器


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    在训练中，我们只关心训练数据的"无限流"。
    因此，这个采样器会产生一个无限的索引流，所有worker协同工作以正确地打乱索引并采样不同的索引。

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    每个worker中的采样器实际上产生 `indices[worker_id::num_workers]`，
    其中 `indices` 是一个无限的索引流，由以下部分组成：
    `shuffle(range(size)) + shuffle(range(size)) + ...`（如果shuffle为True）
    或 `range(size) + range(size) + ...`（如果shuffle为False）

    Note that this sampler does not shard based on pytorch DataLoader worker id.
    A sampler passed to pytorch DataLoader is used only with map-style dataset
    and will not be executed inside workers.
    But if this sampler is used in a way that it gets execute inside a dataloader
    worker, then extra work needs to be done to shard its outputs based on worker id.
    This is required so that workers don't produce identical data.
    :class:`ToIterableDataset` implements this logic.
    This note is true for all samplers in detectron2.
    注意，此采样器不基于pytorch DataLoader的worker id进行分片。
    传递给pytorch DataLoader的采样器仅用于map-style数据集，且不会在workers内部执行。
    但如果这个采样器在dataloader worker内部执行，则需要额外的工作来基于worker id对其输出进行分片。
    这是必要的，以确保workers不会产生相同的数据。
    :class:`ToIterableDataset` 实现了这个逻辑。
    这个说明适用于detectron2中的所有采样器。
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        参数：
            size (int)：要从中采样的底层数据集的总数据量
            shuffle (bool)：是否打乱索引
            seed (int)：打乱的初始种子。必须在所有workers之间保持相同。
                如果为None，将使用workers之间共享的随机种子（需要在所有workers之间同步）。
        """
        if not isinstance(size, int):  # 检查size参数类型是否为整数
            raise TypeError(f"TrainingSampler(size=) expects an int. Got type {type(size)}.")
        if size <= 0:  # 检查size参数是否为正数
            raise ValueError(f"TrainingSampler(size=) expects a positive int. Got {size}.")
        self._size = size  # 保存数据集大小
        self._shuffle = shuffle  # 保存是否需要打乱
        if seed is None:  # 如果没有指定种子，使用共享的随机种子
            seed = comm.shared_random_seed()
        self._seed = int(seed)  # 保存随机种子

        self._rank = comm.get_rank()  # 获取当前进程的rank
        self._world_size = comm.get_world_size()  # 获取总进程数

    def __iter__(self):
        start = self._rank  # 设置起始索引为当前进程的rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)  # 使用islice对无限索引流进行切片，实现数据分片

    def _infinite_indices(self):
        g = torch.Generator()  # 创建随机数生成器
        g.manual_seed(self._seed)  # 设置随机种子
        while True:  # 无限循环生成索引
            if self._shuffle:  # 如果需要打乱
                yield from torch.randperm(self._size, generator=g).tolist()  # 生成随机排列的索引
            else:  # 如果不需要打乱
                yield from torch.arange(self._size).tolist()  # 生成顺序索引


class RandomSubsetTrainingSampler(TrainingSampler):
    """
    Similar to TrainingSampler, but only sample a random subset of indices.
    This is useful when you want to estimate the accuracy vs data-number curves by
      training the model with different subset_ratio.
    与TrainingSampler类似，但只采样随机的索引子集。
    当你想通过使用不同的子集比例训练模型来估计准确率与数据量的关系曲线时，这很有用。
    """

    def __init__(
        self,
        size: int,
        subset_ratio: float,
        shuffle: bool = True,
        seed_shuffle: Optional[int] = None,
        seed_subset: Optional[int] = None,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            subset_ratio (float): the ratio of subset data to sample from the underlying dataset
            shuffle (bool): whether to shuffle the indices or not
            seed_shuffle (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            seed_subset (int): the seed to randomize the subset to be sampled.
                Must be the same across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        参数：
            size (int)：要从中采样的底层数据集的总数据量
            subset_ratio (float)：从底层数据集采样的子集数据比例
            shuffle (bool)：是否打乱索引
            seed_shuffle (int)：打乱的初始种子。必须在所有workers之间保持相同。
                如果为None，将使用workers之间共享的随机种子（需要在所有workers之间同步）。
            seed_subset (int)：用于随机化要采样的子集的种子。
                必须在所有workers之间保持相同。如果为None，将使用workers之间共享的随机种子（需要在所有workers之间同步）。
        """
        super().__init__(size=size, shuffle=shuffle, seed=seed_shuffle)  # 调用父类的初始化方法

        assert 0.0 < subset_ratio <= 1.0  # 确保子集比例在有效范围内
        self._size_subset = int(size * subset_ratio)  # 计算子集大小
        assert self._size_subset > 0  # 确保子集大小大于0
        if seed_subset is None:  # 如果没有指定子集随机种子
            seed_subset = comm.shared_random_seed()  # 使用共享的随机种子
        self._seed_subset = int(seed_subset)  # 保存子集随机种子

        # randomly generate the subset indexes to be sampled from
        # 随机生成要采样的子集索引
        g = torch.Generator()  # 创建随机数生成器
        g.manual_seed(self._seed_subset)  # 设置随机种子
        indexes_randperm = torch.randperm(self._size, generator=g)  # 生成随机排列
        self._indexes_subset = indexes_randperm[: self._size_subset]  # 选择前size_subset个索引作为子集

        logger.info("Using RandomSubsetTrainingSampler......")  # 记录使用RandomSubsetTrainingSampler的信息
        logger.info(f"Randomly sample {self._size_subset} data from the original {self._size} data")  # 记录采样信息

    def _infinite_indices(self):
        g = torch.Generator()  # 创建随机数生成器
        g.manual_seed(self._seed)  # self._seed equals seed_shuffle from __init__()  # 设置随机种子（等于__init__中的seed_shuffle）
        while True:  # 无限循环生成索引
            if self._shuffle:  # 如果需要打乱
                # generate a random permutation to shuffle self._indexes_subset
                # 生成随机排列来打乱self._indexes_subset
                randperm = torch.randperm(self._size_subset, generator=g)  # 生成子集大小的随机排列
                yield from self._indexes_subset[randperm].tolist()  # 根据随机排列返回子集索引
            else:  # 如果不需要打乱
                yield from self._indexes_subset.tolist()  # 直接返回子集索引


class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    与TrainingSampler类似，但根据"重复因子"，某些样本可能出现的次数比其他样本多。
    这适用于像LVIS这样的类别不平衡数据集的训练。
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        参数：
            repeat_factors (Tensor)：一个浮点向量，表示每个索引的重复因子。当全为1时，
                等价于``TrainingSampler(len(repeat_factors), ...)``。
            shuffle (bool)：是否打乱索引
            seed (int)：打乱的初始种子。必须在所有workers之间保持相同。
                如果为None，将使用workers之间共享的随机种子（需要在所有workers之间同步）。
        """
        self._shuffle = shuffle  # 保存是否需要打乱
        if seed is None:  # 如果没有指定种子
            seed = comm.shared_random_seed()  # 使用共享的随机种子
        self._seed = int(seed)  # 保存随机种子

        self._rank = comm.get_rank()  # 获取当前进程的rank
        self._world_size = comm.get_world_size()  # 获取总进程数

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        # 将重复因子分解为整数部分和小数部分
        self._int_part = torch.trunc(repeat_factors)  # 获取整数部分
        self._frac_part = repeat_factors - self._int_part  # 获取小数部分

    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.
        基于类别频率计算每个图像的（小数）重复因子。
        图像的重复因子是该图像中最稀有类别频率的函数。
        "类别c的频率"定义为训练集中（不重复）包含类别c的图像比例，取值范围为[0, 1]。
        参见LVIS论文(>= v2)附录B.2。

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.
            dataset_dicts (list[dict]): Detectron2格式的数据集标注。
            repeat_thresh (float): 数据重复的频率阈值。
                如果频率是`repeat_thresh`的一半，该图像将被重复两次。

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
                第i个元素是数据集中索引为i的图像的重复因子。
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        # 1. 对于每个类别c，计算包含它的图像比例：f(c)
        category_freq = defaultdict(int)  # 创建默认字典存储每个类别的出现次数
        for dataset_dict in dataset_dicts:  # For each image (without repeats)  # 遍历每个图像（不重复）
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}  # 获取图像中所有类别ID
            for cat_id in cat_ids:
                category_freq[cat_id] += 1  # 统计每个类别出现的图像数量
        num_images = len(dataset_dicts)  # 获取总图像数
        for k, v in category_freq.items():
            category_freq[k] = v / num_images  # 计算每个类别的频率（出现次数/总图像数）

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        # 2. 对于每个类别c，计算类别级别的重复因子：
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))  # 计算每个类别的重复因子
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        # 3. 对于每个图像I，计算图像级别的重复因子：
        #    r(I) = max_{c in I} r(c)
        rep_factors = []  # 存储每个图像的重复因子
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}  # 获取图像中所有类别ID
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)  # 取图像中所有类别的最大重复因子
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)  # 返回张量形式的重复因子列表

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.
        创建一个包含重复的数据集索引列表，用于一个训练周期。

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.
            generator (torch.Generator): 用于随机舍入的伪随机数生成器。

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
            torch.Tensor: 一个训练周期使用的数据集索引列表。每个索引根据其计算的重复因子重复。
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        # 由于重复因子是小数，我们使用随机舍入，以便在训练过程中期望达到目标重复因子
        rands = torch.rand(len(self._frac_part), generator=generator)  # 生成随机数用于随机舍入
        rep_factors = self._int_part + (rands < self._frac_part).float()  # 整数部分加上随机舍入的小数部分
        # Construct a list of indices in which we repeat images as specified
        # 构建索引列表，根据指定的重复因子重复图像
        indices = []  # 存储最终的索引列表
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))  # 根据重复因子重复索引
        return torch.tensor(indices, dtype=torch.int64)  # 返回张量形式的索引列表

    def __iter__(self):
        start = self._rank  # 设置起始索引为当前进程的rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)  # 根据进程数对无限索引流进行切片

    def _infinite_indices(self):
        g = torch.Generator()  # 创建随机数生成器
        g.manual_seed(self._seed)  # 设置随机种子
        while True:  # 无限循环生成索引
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            # 采样索引，重复次数由随机舍入决定；由于舍入的原因，每个"周期"的大小可能略有不同
            indices = self._get_epoch_indices(g)  # 获取一个周期的索引
            if self._shuffle:  # 如果需要打乱
                randperm = torch.randperm(len(indices), generator=g)  # 生成随机排列
                yield from indices[randperm].tolist()  # 返回打乱后的索引列表
            else:  # 如果不需要打乱
                yield from indices.tolist()  # 直接返回索引列表


class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    为所有worker生成用于推理的索引。
    推理需要在完全相同的样本集上运行，
    因此当样本总数不能被worker数整除时，
    该采样器在不同worker上产生不同数量的样本。
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            size (int): 要采样的底层数据集的总数据量
        """
        self._size = size  # 保存数据集大小
        assert size > 0  # 确保数据集大小大于0
        self._rank = comm.get_rank()  # 获取当前进程的rank
        self._world_size = comm.get_world_size()  # 获取总进程数
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)  # 获取当前进程的本地索引

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size  # 计算每个分片的基本大小
        left = total_size % world_size  # 计算剩余的样本数
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]  # 为每个进程分配分片大小

        begin = sum(shard_sizes[:rank])  # 计算当前进程分片的起始位置
        end = min(sum(shard_sizes[: rank + 1]), total_size)  # 计算当前进程分片的结束位置
        return range(begin, end)  # 返回当前进程的索引范围

    def __iter__(self):
        yield from self._local_indices  # 返回当前进程的本地索引

    def __len__(self):
        return len(self._local_indices)  # 返回当前进程的样本数量
