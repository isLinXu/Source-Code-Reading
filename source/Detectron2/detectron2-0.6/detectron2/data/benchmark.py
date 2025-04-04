# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from itertools import count
from typing import List, Tuple
import torch
import tqdm
from fvcore.common.timer import Timer

# 导入必要的模块
from detectron2.utils import comm  # 导入通信模块，用于分布式训练

from .build import build_batch_data_loader  # 导入批量数据加载器构建函数
from .common import DatasetFromList, MapDataset  # 导入数据集相关类
from .samplers import TrainingSampler  # 导入训练数据采样器

logger = logging.getLogger(__name__)  # 创建日志记录器


class _EmptyMapDataset(torch.utils.data.Dataset):
    """
    Map anything to emptiness.
    将任何数据映射为空值，用于测试数据加载的IPC开销。
    """

    def __init__(self, dataset):
        self.ds = dataset  # 保存原始数据集

    def __len__(self):
        return len(self.ds)  # 返回数据集长度

    def __getitem__(self, idx):
        _ = self.ds[idx]  # 访问原始数据集但忽略其返回值
        return [0]  # 始终返回[0]，模拟最小数据传输


def iter_benchmark(
    iterator, num_iter: int, warmup: int = 5, max_time_seconds: float = 60
) -> Tuple[float, List[float]]:
    """
    Benchmark an iterator/iterable for `num_iter` iterations with an extra
    `warmup` iterations of warmup.
    End early if `max_time_seconds` time is spent on iterations.
    对迭代器进行基准测试，执行num_iter次迭代，并额外进行warmup次预热迭代。
    如果迭代时间超过max_time_seconds秒则提前结束。

    Returns:
        float: average time (seconds) per iteration
               每次迭代的平均时间（秒）
        list[float]: time spent on each iteration. Sometimes useful for further analysis.
                     每次迭代所花费的时间列表，用于进一步分析
    """
    # 确保迭代次数和预热次数为整数
    num_iter, warmup = int(num_iter), int(warmup)

    iterator = iter(iterator)  # 获取迭代器对象
    for _ in range(warmup):  # 执行预热迭代
        next(iterator)
    timer = Timer()  # 创建计时器
    all_times = []  # 存储每次迭代的时间
    for curr_iter in tqdm.trange(num_iter):  # 使用tqdm显示进度条
        start = timer.seconds()  # 记录开始时间
        if start > max_time_seconds:  # 如果超过最大时间限制则提前结束
            num_iter = curr_iter
            break
        next(iterator)  # 执行一次迭代
        all_times.append(timer.seconds() - start)  # 记录本次迭代耗时
    avg = timer.seconds() / num_iter  # 计算平均耗时
    return avg, all_times


class DataLoaderBenchmark:
    """
    Some common benchmarks that help understand perf bottleneck of a standard dataloader
    made of dataset, mapper and sampler.
    一些常用的基准测试，帮助理解由数据集、映射器和采样器组成的标准数据加载器的性能瓶颈。
    """

    def __init__(
        self,
        dataset,  # 数据集
        *,
        mapper,  # 数据映射器
        sampler=None,  # 数据采样器
        total_batch_size,  # 总批量大小
        num_workers=0,  # 工作进程数
        max_time_seconds: int = 90,  # 每个基准测试的最大运行时间
    ):
        """
        Args:
            max_time_seconds (int): maximum time to spent for each benchmark
                                   每个基准测试的最大运行时间
            other args: same as in `build.py:build_detection_train_loader`
                       其他参数与build.py中的build_detection_train_loader相同
        """
        # 如果数据集是列表，转换为DatasetFromList对象
        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False, serialize=True)
        # 如果没有指定采样器，创建默认的训练采样器
        if sampler is None:
            sampler = TrainingSampler(len(dataset))

        self.dataset = dataset  # 保存数据集
        self.mapper = mapper  # 保存数据映射器
        self.sampler = sampler  # 保存数据采样器
        self.total_batch_size = total_batch_size  # 保存总批量大小
        self.num_workers = num_workers  # 保存工作进程数
        # 计算每个GPU的批量大小
        self.per_gpu_batch_size = self.total_batch_size // comm.get_world_size()

        self.max_time_seconds = max_time_seconds  # 保存最大运行时间

    def _benchmark(self, iterator, num_iter, warmup, msg=None):
        # 执行基准测试并获取平均时间和所有迭代时间
        avg, all_times = iter_benchmark(iterator, num_iter, warmup, self.max_time_seconds)
        # 如果提供了消息，则记录时间统计信息
        if msg is not None:
            self._log_time(msg, avg, all_times)
        return avg, all_times

    def _log_time(self, msg, avg, all_times, distributed=False):
        # 计算时间的1%、5%、95%和99%分位数
        percentiles = [np.percentile(all_times, k, interpolation="nearest") for k in [1, 5, 95, 99]]
        if not distributed:  # 非分布式模式
            # 记录平均速度和各个分位数的时间
            logger.info(
                f"{msg}: avg={1.0/avg:.1f} it/s, "  # 平均每秒迭代次数
                f"p1={percentiles[0]:.2g}s, p5={percentiles[1]:.2g}s, "  # 1%和5%分位数
                f"p95={percentiles[2]:.2g}s, p99={percentiles[3]:.2g}s."  # 95%和99%分位数
            )
            return
        # 分布式模式：收集所有GPU的统计信息
        avg_per_gpu = comm.all_gather(avg)  # 收集所有GPU的平均时间
        percentiles_per_gpu = comm.all_gather(percentiles)  # 收集所有GPU的分位数
        if comm.get_rank() > 0:  # 非主进程直接返回
            return
        # 主进程打印所有GPU的统计信息
        for idx, avg, percentiles in zip(count(), avg_per_gpu, percentiles_per_gpu):
            logger.info(
                f"GPU{idx} {msg}: avg={1.0/avg:.1f} it/s, "
                f"p1={percentiles[0]:.2g}s, p5={percentiles[1]:.2g}s, "
                f"p95={percentiles[2]:.2g}s, p99={percentiles[3]:.2g}s."
            )

    def benchmark_dataset(self, num_iter, warmup=5):
        """
        Benchmark the speed of taking raw samples from the dataset.
        测试从数据集中获取原始样本的速度。
        """

        def loader():
            while True:  # 无限循环生成器
                for k in self.sampler:  # 使用采样器遍历数据集
                    yield self.dataset[k]  # 返回数据集中的样本

        # 执行基准测试
        self._benchmark(loader(), num_iter, warmup, "Dataset Alone")

    def benchmark_mapper(self, num_iter, warmup=5):
        """
        Benchmark the speed of taking raw samples from the dataset and map
        them in a single process.
        测试在单进程中获取原始样本并进行数据映射的速度。
        """

        def loader():
            while True:
                for k in self.sampler:
                    yield self.mapper(self.dataset[k])

        self._benchmark(loader(), num_iter, warmup, "Single Process Mapper (sec/sample)")

    def benchmark_workers(self, num_iter, warmup=10):
        """
        Benchmark the dataloader by tuning num_workers to [0, 1, self.num_workers].
        通过调整工作进程数（0、1和self.num_workers）来测试数据加载器的性能。
        """
        candidates = [0, 1]  # 默认测试0和1个工作进程
        if self.num_workers not in candidates:
            candidates.append(self.num_workers)  # 如果设置了其他数量的工作进程，也加入测试

        dataset = MapDataset(self.dataset, self.mapper)  # 创建映射数据集
        for n in candidates:  # 对每个工作进程数进行测试
            loader = build_batch_data_loader(
                dataset,
                self.sampler,
                self.total_batch_size,
                num_workers=n,
            )
            self._benchmark(
                iter(loader),
                num_iter * max(n, 1),  # 根据工作进程数调整迭代次数
                warmup * max(n, 1),  # 根据工作进程数调整预热次数
                f"DataLoader ({n} workers, bs={self.per_gpu_batch_size})",
            )
            del loader  # 删除数据加载器释放资源

    def benchmark_IPC(self, num_iter, warmup=10):
        """
        Benchmark the dataloader where each worker outputs nothing. This
        eliminates the IPC overhead compared to the regular dataloader.
        测试数据加载器的进程间通信（IPC）开销，通过让每个工作进程不输出任何数据来实现。

        PyTorch multiprocessing's IPC only optimizes for torch tensors.
        Large numpy arrays or other data structure may incur large IPC overhead.
        PyTorch的多进程IPC只对torch张量进行了优化，大型numpy数组或其他数据结构可能会产生较大的IPC开销。
        """
        n = self.num_workers  # 获取工作进程数
        dataset = _EmptyMapDataset(MapDataset(self.dataset, self.mapper))  # 创建空数据集
        loader = build_batch_data_loader(
            dataset, self.sampler, self.total_batch_size, num_workers=n
        )
        self._benchmark(
            iter(loader),
            num_iter * max(n, 1),  # 根据工作进程数调整迭代次数
            warmup * max(n, 1),  # 根据工作进程数调整预热次数
            f"DataLoader ({n} workers, bs={self.per_gpu_batch_size}) w/o comm",
        )

    def benchmark_distributed(self, num_iter, warmup=10):
        """
        Benchmark the dataloader in each distributed worker, and log results of
        all workers. This helps understand the final performance as well as
        the variances among workers.
        在每个分布式工作进程中测试数据加载器，并记录所有工作进程的结果。
        这有助于理解最终性能以及工作进程之间的性能差异。

        It also prints startup time (first iter) of the dataloader.
        同时还会打印数据加载器的启动时间（第一次迭代）。
        """
        gpu = comm.get_world_size()  # 获取GPU数量
        dataset = MapDataset(self.dataset, self.mapper)  # 创建映射数据集
        n = self.num_workers  # 获取工作进程数
        loader = build_batch_data_loader(
            dataset, self.sampler, self.total_batch_size, num_workers=n
        )

        timer = Timer()  # 创建计时器
        loader = iter(loader)  # 获取迭代器
        next(loader)  # 执行第一次迭代
        startup_time = timer.seconds()  # 记录启动时间
        logger.info("Dataloader startup time: {:.2f} seconds".format(startup_time))

        comm.synchronize()  # 同步所有进程

        # 执行基准测试
        avg, all_times = self._benchmark(loader, num_iter * max(n, 1), warmup * max(n, 1))
        del loader  # 删除数据加载器释放资源
        self._log_time(
            f"DataLoader ({gpu} GPUs x {n} workers, total bs={self.total_batch_size})",
            avg,
            all_times,
            True,  # 启用分布式日志记录
        )
