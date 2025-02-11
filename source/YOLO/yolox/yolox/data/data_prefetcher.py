#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)  # 将数据加载器转换为迭代器
        self.stream = torch.cuda.Stream()  # 创建一个 CUDA 流
        self.input_cuda = self._input_cuda_for_image  # 设置输入图像的 CUDA 处理方法
        self.record_stream = DataPrefetcher._record_stream_for_image  # 设置记录流的方法
        self.preload()  # 预加载数据

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)  # 获取下一个输入和目标
        except StopIteration:  # 如果没有更多数据可加载
            self.next_input = None  # 设置下一个输入为 None
            self.next_target = None  # 设置下一个目标为 None
            return

        with torch.cuda.stream(self.stream):  # 在指定的 CUDA 流中执行
            self.input_cuda()  # 将输入数据转移到 CUDA
            self.next_target = self.next_target.cuda(non_blocking=True)  # 将目标数据转移到 CUDA，非阻塞方式

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)  # 等待当前流完成
        input = self.next_input  # 获取下一个输入
        target = self.next_target  # 获取下一个目标
        if input is not None:  # 如果输入不为 None
            self.record_stream(input)  # 记录输入流
        if target is not None:  # 如果目标不为 None
            target.record_stream(torch.cuda.current_stream())  # 记录目标流
        self.preload()  # 预加载下一个数据
        return input, target  # 返回输入和目标

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)  # 将下一个输入转移到 CUDA，非阻塞方式

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())  # 记录输入流
