# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from detectron2.utils import comm  # 导入通信模块

__all__ = ["DEFAULT_TIMEOUT", "launch"]  # 指定可以被外部导入的对象

DEFAULT_TIMEOUT = timedelta(minutes=30)  # 设置默认超时时间为30分钟


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建TCP套接字
    # Binding to port 0 will cause the OS to find an available port for us
    # 绑定端口0会让操作系统自动分配一个可用端口
    sock.bind(("", 0))
    port = sock.getsockname()[1]  # 获取分配的端口号
    sock.close()  # 关闭套接字
    # NOTE: there is still a chance the port could be taken by other processes.
    # 注意：仍然存在端口被其他进程占用的可能性
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.
    启动多GPU或分布式训练。
    此函数必须在所有参与训练的机器上调用。
    它将在每台机器上创建子进程（由``num_gpus_per_machine``定义）。

    Args:
        main_func: a function that will be called by `main_func(*args)`
                  将被调用的主函数，通过`main_func(*args)`方式调用
        num_gpus_per_machine (int): number of GPUs per machine
                                   每台机器上的GPU数量
        num_machines (int): the total number of machines
                           机器总数
        machine_rank (int): the rank of this machine
                           当前机器的序号
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
                       分布式作业的连接URL，包含协议
                       例如："tcp://127.0.0.1:8686"
                       可以设置为"auto"以在localhost上自动选择空闲端口
        timeout (timedelta): timeout of the distributed workers
                            分布式工作进程的超时时间
        args (tuple): arguments passed to main_func
                      传递给main_func的参数
    """
    world_size = num_machines * num_gpus_per_machine  # 计算总的进程数（世界大小）
    if world_size > 1:  # 如果是多GPU或多机训练
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":  # 如果URL设置为自动
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."  # 自动模式仅支持单机多卡
            port = _find_free_port()  # 获取空闲端口
            dist_url = f"tcp://127.0.0.1:{port}"  # 构建本地TCP URL
        if num_machines > 1 and dist_url.startswith("file://"):  # 多机训练时检查URL协议
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"  # 警告不要在多机训练中使用file://协议
            )

        mp.spawn(  # 启动多个分布式训练进程
            _distributed_worker,  # 分布式工作进程函数
            nprocs=num_gpus_per_machine,  # 进程数量等于GPU数量
            args=(  # 传递给工作进程的参数
                main_func,  # 主函数
                world_size,  # 总进程数
                num_gpus_per_machine,  # 每台机器的GPU数量
                machine_rank,  # 机器序号
                dist_url,  # 分布式URL
                args,  # 主函数参数
                timeout,  # 超时时间
            ),
            daemon=False,  # 设置为非守护进程
        )
    else:  # 单GPU训练时直接运行主函数
        main_func(*args)


def _distributed_worker(
    local_rank,  # 本地进程序号
    main_func,  # 主函数
    world_size,  # 总进程数
    num_gpus_per_machine,  # 每台机器的GPU数量
    machine_rank,  # 机器序号
    dist_url,  # 分布式URL
    args,  # 主函数参数
    timeout=DEFAULT_TIMEOUT,  # 超时时间
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."  # 确保CUDA可用
    global_rank = machine_rank * num_gpus_per_machine + local_rank  # 计算全局进程序号
    try:
        dist.init_process_group(  # 初始化分布式进程组
            backend="NCCL",  # 使用NCCL后端
            init_method=dist_url,  # 初始化方法（URL）
            world_size=world_size,  # 总进程数
            rank=global_rank,  # 全局进程序号
            timeout=timeout,  # 超时时间
        )
    except Exception as e:  # 捕获初始化异常
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))  # 记录错误URL
        raise e

    # Setup the local process group (which contains ranks within the same machine)
    # 设置本地进程组（包含同一机器内的所有进程）
    assert comm._LOCAL_PROCESS_GROUP is None  # 确保本地进程组未初始化
    num_machines = world_size // num_gpus_per_machine  # 计算机器数量
    for i in range(num_machines):  # 遍历所有机器
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))  # 计算当前机器上的进程序号列表
        pg = dist.new_group(ranks_on_i)  # 创建新的进程组
        if i == machine_rank:  # 如果是当前机器
            comm._LOCAL_PROCESS_GROUP = pg  # 设置本地进程组

    assert num_gpus_per_machine <= torch.cuda.device_count()  # 确保请求的GPU数量不超过可用数量
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU设备

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    # 在此处同步是必要的，以防止调用init_process_group后可能的超时
    # 参见：https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()  # 同步所有进程

    main_func(*args)  # 运行主函数
