# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.

此文件包含多GPU通信的基本功能。
这在进行分布式训练时很有用。
"""

import functools  # 导入functools模块，提供高阶函数和操作可调用对象的工具
import logging  # 导入日志记录模块
import numpy as np  # 导入numpy库
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块

_LOCAL_PROCESS_GROUP = None  # 初始化本地进程组为None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".

一个torch进程组，只包括与当前进程在同一台机器上的进程。
这个变量在"engine/launch.py"中的`launch()`函数生成进程时设置。
"""


def get_world_size() -> int:
    if not dist.is_available():  # 如果分布式包不可用
        return 1  # 返回1，表示只有一个进程
    if not dist.is_initialized():  # 如果分布式环境未初始化
        return 1  # 返回1，表示只有一个进程
    return dist.get_world_size()  # 返回分布式环境中的总进程数


def get_rank() -> int:
    if not dist.is_available():  # 如果分布式包不可用
        return 0  # 返回0，表示当前是唯一的进程
    if not dist.is_initialized():  # 如果分布式环境未初始化
        return 0  # 返回0，表示当前是唯一的进程
    return dist.get_rank()  # 返回当前进程在全局中的排名


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
        
    返回：
        当前进程在本地（每台机器）进程组中的排名。
    """
    if not dist.is_available():  # 如果分布式包不可用
        return 0  # 返回0，表示当前是唯一的进程
    if not dist.is_initialized():  # 如果分布式环境未初始化
        return 0  # 返回0，表示当前是唯一的进程
    assert (
        _LOCAL_PROCESS_GROUP is not None
    ), "Local process group is not created! Please use launch() to spawn processes!"  # 确保本地进程组已创建
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)  # 返回当前进程在本地进程组中的排名


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
        
    返回：
        每台机器进程组的大小，
        即每台机器上的进程数。
    """
    if not dist.is_available():  # 如果分布式包不可用
        return 1  # 返回1，表示只有一个进程
    if not dist.is_initialized():  # 如果分布式环境未初始化
        return 1  # 返回1，表示只有一个进程
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)  # 返回本地进程组的大小


def is_main_process() -> bool:
    return get_rank() == 0  # 检查当前进程是否为主进程（排名为0）


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    
    辅助函数，在使用分布式训练时
    在所有进程之间进行同步（屏障）
    """
    if not dist.is_available():  # 如果分布式包不可用
        return  # 直接返回
    if not dist.is_initialized():  # 如果分布式环境未初始化
        return  # 直接返回
    world_size = dist.get_world_size()  # 获取总进程数
    if world_size == 1:  # 如果只有一个进程
        return  # 无需同步，直接返回
    if dist.get_backend() == dist.Backend.NCCL:  # 如果使用NCCL后端
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        # 这个参数是为了避免警告。
        # 它只对NCCL后端有效。
        dist.barrier(device_ids=[torch.cuda.current_device()])  # 使用当前设备ID创建屏障
    else:
        dist.barrier()  # 创建一个屏障，同步所有进程


@functools.lru_cache()  # 使用LRU缓存装饰器缓存函数结果
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    
    返回一个基于gloo后端的进程组，包含所有排名
    结果会被缓存。
    """
    if dist.get_backend() == "nccl":  # 如果当前使用的是nccl后端
        return dist.new_group(backend="gloo")  # 创建一个新的基于gloo的进程组
    else:
        return dist.group.WORLD  # 使用默认的世界进程组


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)  # 获取进程组的后端类型
    assert backend in ["gloo", "nccl"]  # 确保后端是gloo或nccl
    device = torch.device("cpu" if backend == "gloo" else "cuda")  # 根据后端选择设备

    buffer = pickle.dumps(data)  # 将数据序列化为二进制字符串
    if len(buffer) > 1024 ** 3:  # 如果缓冲区大于1GB
        logger = logging.getLogger(__name__)  # 获取记录器
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )  # 记录警告信息
    storage = torch.ByteStorage.from_buffer(buffer)  # 从缓冲区创建ByteStorage
    tensor = torch.ByteTensor(storage).to(device=device)  # 从存储创建ByteTensor并移至指定设备
    return tensor  # 返回张量


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
        
    返回：
        list[int]：每个排名上张量的大小
        Tensor：具有最大大小的填充张量
    """
    world_size = dist.get_world_size(group=group)  # 获取进程组的大小
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"  # 确保进程组有效
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)  # 获取本地张量的元素数量
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]  # 创建一个列表来存储所有进程的张量大小
    dist.all_gather(size_list, local_size, group=group)  # 收集所有进程的张量大小
    size_list = [int(size.item()) for size in size_list]  # 将张量转换为整数列表

    max_size = max(size_list)  # 获取最大的张量大小

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    # 我们对张量进行填充，因为torch的all_gather不支持
    # 收集不同形状的张量
    if local_size != max_size:  # 如果本地张量大小不是最大的
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)  # 创建填充张量
        tensor = torch.cat((tensor, padding), dim=0)  # 将原始张量和填充张量连接起来
    return size_list, tensor  # 返回大小列表和填充后的张量


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
        
    在任意可序列化数据（不一定是张量）上运行all_gather。

    参数：
        data：任何可序列化对象
        group：一个torch进程组。默认情况下，将使用包含
            所有在gloo后端上的排名的组。

    返回：
        list[data]：从每个排名收集的数据列表
    """
    if get_world_size() == 1:  # 如果只有一个进程
        return [data]  # 直接返回数据列表
    if group is None:  # 如果未指定进程组
        group = _get_global_gloo_group()  # 获取全局gloo进程组
    if dist.get_world_size(group) == 1:  # 如果进程组只有一个进程
        return [data]  # 直接返回数据列表

    tensor = _serialize_to_tensor(data, group)  # 将数据序列化为张量

    size_list, tensor = _pad_to_largest_tensor(tensor, group)  # 将张量填充到最大大小
    max_size = max(size_list)  # 获取最大大小

    # receiving Tensor from all ranks
    # 从所有排名接收张量
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]  # 创建张量列表来接收所有数据
    dist.all_gather(tensor_list, tensor, group=group)  # 从所有进程收集张量

    data_list = []  # 初始化数据列表
    for size, tensor in zip(size_list, tensor_list):  # 遍历每个大小和张量
        buffer = tensor.cpu().numpy().tobytes()[:size]  # 将张量转换为字节并截取有效部分
        data_list.append(pickle.loads(buffer))  # 反序列化数据并添加到列表

    return data_list  # 返回收集的数据列表


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
            
    在任意可序列化数据（不一定是张量）上运行gather。

    参数：
        data：任何可序列化对象
        dst (int)：目标排名
        group：一个torch进程组。默认情况下，将使用包含
            所有在gloo后端上的排名的组。

    返回：
        list[data]：在dst上，从每个排名收集的数据列表。
            否则，返回空列表。
    """
    if get_world_size() == 1:  # 如果只有一个进程
        return [data]  # 直接返回数据列表
    if group is None:  # 如果未指定进程组
        group = _get_global_gloo_group()  # 获取全局gloo进程组
    if dist.get_world_size(group=group) == 1:  # 如果进程组只有一个进程
        return [data]  # 直接返回数据列表
    rank = dist.get_rank(group=group)  # 获取当前进程在组中的排名

    tensor = _serialize_to_tensor(data, group)  # 将数据序列化为张量
    size_list, tensor = _pad_to_largest_tensor(tensor, group)  # 将张量填充到最大大小

    # receiving Tensor from all ranks
    # 从所有排名接收张量
    if rank == dst:  # 如果当前进程是目标进程
        max_size = max(size_list)  # 获取最大大小
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
        ]  # 创建张量列表来接收所有数据
        dist.gather(tensor, tensor_list, dst=dst, group=group)  # 收集所有进程的张量

        data_list = []  # 初始化数据列表
        for size, tensor in zip(size_list, tensor_list):  # 遍历每个大小和张量
            buffer = tensor.cpu().numpy().tobytes()[:size]  # 将张量转换为字节并截取有效部分
            data_list.append(pickle.loads(buffer))  # 反序列化数据并添加到列表
        return data_list  # 返回收集的数据列表
    else:
        dist.gather(tensor, [], dst=dst, group=group)  # 非目标进程只发送不接收
        return []  # 返回空列表


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    
    返回：
        int：一个在所有工作进程中相同的随机数。
        如果工作进程需要共享的随机数生成器，他们可以使用这个共享的种子
        来创建一个。

    所有工作进程必须调用此函数，否则将陷入死锁。
    """
    ints = np.random.randint(2 ** 31)  # 生成一个随机整数
    all_ints = all_gather(ints)  # 收集所有进程的随机整数
    return all_ints[0]  # 返回第一个进程的随机整数，确保所有进程使用相同的种子


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
        
    减少来自所有进程的字典中的值，使得排名为0的进程
    拥有减少后的结果。

    参数：
        input_dict (dict)：要减少的输入。所有值必须是标量CUDA张量。
        average (bool)：是进行平均还是求和

    返回：
        减少后的字典，具有与input_dict相同的键。
    """
    world_size = get_world_size()  # 获取总进程数
    if world_size < 2:  # 如果只有一个进程
        return input_dict  # 直接返回输入字典
    with torch.no_grad():  # 不跟踪梯度
        names = []  # 初始化名称列表
        values = []  # 初始化值列表
        # sort the keys so that they are consistent across processes
        # 对键进行排序，以确保它们在所有进程中保持一致
        for k in sorted(input_dict.keys()):  # 遍历排序后的键
            names.append(k)  # 添加键到名称列表
            values.append(input_dict[k])  # 添加值到值列表
        values = torch.stack(values, dim=0)  # 将值堆叠成一个张量
        dist.reduce(values, dst=0)  # 将值减少到排名为0的进程
        if dist.get_rank() == 0 and average:  # 如果是主进程且需要平均
            # only main process gets accumulated, so only divide by
            # world_size in this case
            # 只有主进程获取累积值，所以只在这种情况下
            # 除以world_size
            values /= world_size  # 计算平均值
        reduced_dict = {k: v for k, v in zip(names, values)}  # 创建减少后的字典
    return reduced_dict  # 返回减少后的字典
