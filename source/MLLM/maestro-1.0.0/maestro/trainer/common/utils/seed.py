# 导入必要的库
import random
from typing import Optional

import numpy as np
import torch


def ensure_reproducibility(
    seed: Optional[int] = None,
    disable_cudnn_benchmark: bool = True,
    avoid_non_deterministic_algorithms: bool = True,
) -> None:
    """
    Sets seeds and configuration options to improve experiment reproducibility.
    设置种子和配置选项以提高实验的可重复性。

    This function ensures that random number generation is controlled for
    Python's `random` module, NumPy, and PyTorch when a seed is provided.
    It also configures CUDA settings to reduce sources of non-determinism
    when using GPUs.
    该函数确保在提供种子时，控制Python的`random`模块、NumPy和PyTorch的随机数生成。
    它还配置CUDA设置，以减少使用GPU时的非确定性来源。

    Args:
        seed (Optional[int]):
            The random seed to use. If `None`, no seeding is applied, and
            the behavior remains stochastic.
            要使用的随机种子。如果为`None`，则不应用种子，行为保持随机。
        disable_cudnn_benchmark (bool):
            If `True`, disables cuDNN benchmarking. This can improve reproducibility
            by preventing cuDNN from selecting the fastest algorithm dynamically,
            which may introduce variability across runs.
            如果为`True`，则禁用cuDNN基准测试。这可以通过防止cuDNN动态选择最快的算法来提高可重复性，
            因为动态选择可能会在不同运行之间引入变异性。
        avoid_non_deterministic_algorithms (bool):
            If `True`, enforces deterministic algorithms in PyTorch by calling
            `torch.use_deterministic_algorithms(True)`. This helps ensure consistent
            results across runs but may impact performance by disabling certain
            optimizations.
            如果为`True`，则通过调用`torch.use_deterministic_algorithms(True)`在PyTorch中强制使用确定性算法。
            这有助于确保不同运行之间的一致性结果，但可能会通过禁用某些优化来影响性能。

    Returns:
        None
    """
    # 如果提供了种子，设置随机种子
    if seed is not None:
        random.seed(seed)  # 设置Python的random模块的种子
        torch.manual_seed(seed)  # 设置PyTorch的种子
        np.random.seed(seed)  # 设置NumPy的种子

        # 如果CUDA可用，设置CUDA的种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # 设置当前CUDA设备的种子
            torch.cuda.manual_seed_all(seed)  # 设置所有CUDA设备的种子

    # 如果避免非确定性算法，强制使用确定性算法
    if avoid_non_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # 如果禁用cuDNN基准测试，设置相关配置
    if disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试
        torch.backends.cudnn.deterministic = True  # 启用cuDNN的确定性模式