import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，用于与操作系统交互
import threading  # 导入threading模块，用于多线程操作
import time  # 导入time模块，用于时间相关的操作
from functools import wraps  # 从functools模块导入wraps，用于装饰器

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.distributed.autograd as dist_autograd  # 导入分布式自动求导模块
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.optim as optim  # 导入优化器模块
from torch.distributed.optim import DistributedOptimizer  # 导入分布式优化器
from torch.distributed.rpc import RRef  # 导入远程引用

from torchvision.models.resnet import Bottleneck  # 从torchvision导入ResNet的Bottleneck模块


#########################################################
#           Define Model Parallel ResNet50              #
#########################################################

# In order to split the ResNet50 and place it on two different workers, we
# implement it in two model shards. The ResNetBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.
# 为了将ResNet50拆分并放置在两个不同的工作节点上，我们
# 将其实现为两个模型分片。ResNetBase类定义了两个分片共享的公共
# 属性和方法。ResNetShard1和ResNetShard2分别包含模型层的两个部分。

num_classes = 1000  # 类别数量


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # 1x1卷积
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)  # 返回1x1卷积层


class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=1000,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()  # 调用父类构造函数

        self._lock = threading.Lock()  # 创建线程锁
        self._block = block  # 保存块类型
        self._norm_layer = nn.BatchNorm2d  # 设置归一化层
        self.inplanes = inplanes  # 保存输入通道数
        self.dilation = 1  # 设置扩张率
        self.groups = groups  # 保存组数
        self.base_width = width_per_group  # 保存每组的宽度

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer  # 获取归一化层
        downsample = None  # 初始化下采样层
        previous_dilation = self.dilation  # 保存之前的扩张率
        if stride != 1 or self.inplanes != planes * self._block.expansion:  # 如果需要下采样
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),  # 创建1x1卷积进行下采样
                norm_layer(planes * self._block.expansion),  # 添加归一化层
            )

        layers = []  # 初始化层列表
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer))  # 添加第一个块
        self.inplanes = planes * self._block.expansion  # 更新输入通道数
        for _ in range(1, blocks):  # 遍历剩余块
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))  # 添加块

        return nn.Sequential(*layers)  # 返回由层组成的序列

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        # 为给定本地模块中的每个参数创建一个RRef，并返回RRef列表
        return [RRef(p) for p in self.parameters()]  # 返回参数的远程引用


class ResNetShard1(ResNetBase):
    """
    The first part of ResNet.
    """
    # ResNet的第一部分
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)  # 调用父类构造函数

        self.device = device  # 保存设备信息
        self.seq = nn.Sequential(  # 创建序列模型
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),  # 创建卷积层
            self._norm_layer(self.inplanes),  # 添加归一化层
            nn.ReLU(inplace=True),  # 添加ReLU激活层
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 添加最大池化层
            self._make_layer(64, 3),  # 添加第一个块
            self._make_layer(128, 4, stride=2)  # 添加第二个块
        ).to(self.device)  # 将模型移动到设备

        for m in self.modules():  # 遍历模型中的所有模块
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 初始化权重
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批归一化层
                nn.init.ones_(m.weight)  # 初始化权重为1
                nn.init.zeros_(m.bias)  # 初始化偏置为0

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)  # 获取远程引用的张量并移动到设备
        with self._lock:  # 使用锁确保线程安全
            out = self.seq(x)  # 通过序列模型进行前向传播
        return out.cpu()  # 将输出移动到CPU并返回


class ResNetShard2(ResNetBase):
    """
    The second part of ResNet.
    """
    # ResNet的第二部分
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 512, num_classes=num_classes, *args, **kwargs)  # 调用父类构造函数

        self.device = device  # 保存设备信息
        self.seq = nn.Sequential(  # 创建序列模型
            self._make_layer(256, 6, stride=2),  # 添加第一个块
            self._make_layer(512, 3, stride=2),  # 添加第二个块
            nn.AdaptiveAvgPool2d((1, 1)),  # 添加自适应平均池化层
        ).to(self.device)  # 将模型移动到设备

        self.fc = nn.Linear(512 * self._block.expansion, num_classes).to(self.device)  # 创建全连接层并移动到设备

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)  # 获取远程引用的张量并移动到设备
        with self._lock:  # 使用锁确保线程安全
            out = self.fc(torch.flatten(self.seq(x), 1))  # 通过序列模型和全连接层进行前向传播
        return out.cpu()  # 将输出移动到CPU并返回


class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """
    # 将两个部分组装为nn.Module并定义流水线逻辑
    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()  # 调用父类构造函数

        self.split_size = split_size  # 保存分割大小

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(  # 将ResNet50的第一部分放置在workers[0]
            workers[0],
            ResNetShard1,
            args=("cuda:0",) + args,
            kwargs=kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(  # 将ResNet50的第二部分放置在workers[1]
            workers[1],
            ResNetShard2,
            args=("cuda:1",) + args,
            kwargs=kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        # 将输入批次xs拆分为微批次，并将异步RPC的未来对象收集到列表中
        out_futures = []  # 初始化输出未来对象列表
        for x in iter(xs.split(self.split_size, dim=0)):  # 遍历拆分后的微批次
            x_rref = RRef(x)  # 创建微批次的远程引用
            y_rref = self.p1_rref.remote().forward(x_rref)  # 通过第一部分进行前向传播
            z_fut = self.p2_rref.rpc_async().forward(y_rref)  # 通过第二部分进行异步前向传播
            out_futures.append(z_fut)  # 将未来对象添加到列表中

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))  # 收集并连接所有输出张量

    def parameter_rrefs(self):
        remote_params = []  # 初始化远程参数列表
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())  # 获取第一部分的参数远程引用
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())  # 获取第二部分的参数远程引用
        return remote_params  # 返回所有远程参数引用


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 3  # 批次数
batch_size = 120  # 批处理大小
image_w = 128  # 图像宽度
image_h = 128  # 图像高度


def run_master(split_size):
    # put the two model parts on worker1 and worker2 respectively
    # 将两个模型部分分别放置在worker1和worker2上
    model = DistResNet50(split_size, ["worker1", "worker2"])  # 创建分布式ResNet50模型
    loss_fn = nn.MSELoss()  # 创建均方误差损失函数
    opt = DistributedOptimizer(  # 创建分布式优化器
        optim.SGD,
        model.parameter_rrefs(),  # 获取模型参数的远程引用
        lr=0.05,  # 设置学习率
    )

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)  # 创建随机的one-hot索引

    for i in range(num_batches):  # 遍历批次数
        print(f"Processing batch {i}")  # 打印处理批次的信息
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)  # 生成随机输入
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)  # 创建one-hot标签

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:  # 创建分布式自动求导上下文
            outputs = model(inputs)  # 前向传播
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])  # 反向传播
            opt.step(context_id)  # 更新优化器


def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '29500'  # 设置主节点端口

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)  # 设置RPC后端选项

    if rank == 0:  # 如果是主节点
        rpc.init_rpc(  # 初始化RPC
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)  # 运行主节点的训练
    else:  # 如果是工作节点
        rpc.init_rpc(  # 初始化工作节点的RPC
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass  # 工作节点不执行任何操作

    # block until all rpcs finish
    rpc.shutdown()  # 关闭RPC


if __name__ == '__main__':
    world_size = 3  # 设置世界规模
    for num_split in [1, 2, 4, 8]:  # 遍历分割数量
        tik = time.time()  # 记录开始时间
        mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)  # 启动多个进程
        tok = time.time()  # 记录结束时间
        print(f"number of splits = {num_split}, execution time = {tok - tik}")  # 打印分割数量和执行时间