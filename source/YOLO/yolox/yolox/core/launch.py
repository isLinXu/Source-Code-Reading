#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
from datetime import timedelta
from loguru import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import yolox.utils.dist as comm

__all__ = ["launch"]


DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port



def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`  # 主函数，将通过`main_func(*args)`调用
        num_machines (int): the total number of machines  # 总机器数量
        machine_rank (int): the rank of this machine (one per machine)  # 当前机器的排名（每台机器一个排名）
        dist_url (str): url to connect to for distributed training, including protocol  # 用于分布式训练的连接URL，包括协议
                       e.g. "tcp://127.0.0.1:8686".  # 示例URL
                       Can be set to auto to automatically select a free port on localhost  # 可以设置为auto以自动选择本地主机上的空闲端口
        args (tuple): arguments passed to main_func  # 传递给主函数的参数
    """
    world_size = num_machines * num_gpus_per_machine  # 计算总的进程数
    if world_size > 1:  # 如果进程数大于1，进行分布式训练
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes  # TODO: 在生成的进程中使用prctl

        if dist_url == "auto":  # 如果dist_url设置为auto
            assert (
                num_machines == 1
            ), "dist_url=auto cannot work with distributed training."  # 断言：如果有多个机器，不能使用auto
            port = _find_free_port()  # 查找一个空闲端口
            dist_url = f"tcp://127.0.0.1:{port}"  # 设置dist_url为找到的空闲端口

        start_method = "spawn"  # 设置进程启动方法为spawn
        cache = vars(args[1]).get("cache", False)  # 获取缓存参数

        # To use numpy memmap for caching image into RAM, we have to use fork method  # 为了使用numpy内存映射将图像缓存到RAM中，我们必须使用fork方法
        if cache:  # 如果启用了缓存
            assert sys.platform != "win32", (  # 断言：在Windows平台上不支持fork方法
                "As Windows platform doesn't support fork method, "
                "do not add --cache in your training command."  # 在训练命令中不要添加--cache
            )
            start_method = "fork"  # 设置进程启动方法为fork

        mp.start_processes(  # 启动多个进程
            _distributed_worker,
            nprocs=num_gpus_per_machine,  # 每台机器的GPU数量
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                backend,
                dist_url,
                args,
            ),
            daemon=False,  # 设置为非守护进程
            start_method=start_method,  # 使用指定的启动方法
        )
    else:
        main_func(*args)  # 如果只有一个进程，直接调用主函数



def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."  # 断言：CUDA可用，否则提示检查安装
    global_rank = machine_rank * num_gpus_per_machine + local_rank  # 计算全局排名
    logger.info("Rank {} initialization finished.".format(global_rank))  # 记录初始化完成的信息
    try:
        dist.init_process_group(  # 初始化进程组
            backend=backend,  # 设置后端
            init_method=dist_url,  # 设置初始化方法
            world_size=world_size,  # 设置总进程数
            rank=global_rank,  # 设置当前进程的排名
            timeout=timeout,  # 设置超时
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))  # 记录错误信息
        raise  # 抛出异常

    # Setup the local process group (which contains ranks within the same machine)  # 设置本地进程组（包含同一机器内的排名）
    assert comm._LOCAL_PROCESS_GROUP is None  # 断言本地进程组为空
    num_machines = world_size // num_gpus_per_machine  # 计算机器数量
    for i in range(num_machines):  # 遍历每台机器
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)  # 获取当前机器上的排名
        )
        pg = dist.new_group(ranks_on_i)  # 创建新的进程组
        if i == machine_rank:  # 如果是当前机器
            comm._LOCAL_PROCESS_GROUP = pg  # 设置本地进程组

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172  # 这里需要同步以防止在调用init_process_group后超时
    comm.synchronize()  # 同步所有进程

    assert num_gpus_per_machine <= torch.cuda.device_count()  # 断言每台机器的GPU数量不超过可用的GPU数量
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU

    main_func(*args)  # 调用主函数
