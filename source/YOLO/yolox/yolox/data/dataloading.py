#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random
import uuid

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate

from .samplers import YoloBatchSampler


def get_yolox_datadir():
    """
    get dataset dir of YOLOX. If environment variable named `YOLOX_DATADIR` is set,
    this function will return value of the environment variable. Otherwise, use data
    """
    yolox_datadir = os.getenv("YOLOX_DATADIR", None)  # 获取环境变量 YOLOX_DATADIR 的值
    if yolox_datadir is None:  # 如果环境变量未设置
        import yolox  # 导入 yolox 模块

        yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))  # 获取 yolox 的路径
        yolox_datadir = os.path.join(yolox_path, "datasets")  # 拼接数据集路径
    return yolox_datadir  # 返回数据集路径


class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.__initialized = False  # 初始化状态
        shuffle = False  # 是否打乱数据
        batch_sampler = None  # 批量采样器

        # 根据传入的参数设置 shuffle、sampler 和 batch_sampler
        if len(args) > 5:
            shuffle = args[2]  # 获取 shuffle 参数
            sampler = args[3]  # 获取 sampler 参数
            batch_sampler = args[4]  # 获取 batch_sampler 参数
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:  # 如果没有指定 batch_sampler
            if sampler is None:  # 如果没有指定 sampler
                if shuffle:  # 如果需要打乱数据
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)  # 随机采样
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)  # 顺序采样
            batch_sampler = YoloBatchSampler(  # 使用自定义的 YoloBatchSampler
                sampler,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler  # 设置批量采样器

        self.__initialized = True  # 设置初始化状态为 True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False  # 关闭马赛克功能


def list_collate(batch):
    """
    Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of
    items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))  # 将批次数据解压为列表

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):  # 如果是列表或元组
            items[i] = list(items[i])  # 转换为列表
        else:
            items[i] = default_collate(items[i])  # 使用默认的 collate 方法

    return items  # 返回合并后的列表


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32  # 生成一个随机种子
    random.seed(seed)  # 设置 Python 的随机种子
    torch.set_rng_state(torch.manual_seed(seed).get_state())  # 设置 PyTorch 的随机状态
    np.random.seed(seed)  # 设置 NumPy 的随机种子