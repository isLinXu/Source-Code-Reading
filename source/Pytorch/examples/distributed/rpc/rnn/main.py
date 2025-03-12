import os  # 导入os模块，用于与操作系统交互

import torch  # 导入PyTorch库
import torch.distributed.autograd as dist_autograd  # 导入分布式自动求导模块
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.optim as optim  # 导入优化器模块
from torch.distributed.optim import DistributedOptimizer  # 导入分布式优化器

import rnn  # 导入自定义的rnn模块


def _run_trainer():
    r"""
    The trainer creates a distributed RNNModel and a DistributedOptimizer. Then,
    it performs training using random input data.
    """  # 训练器创建一个分布式RNN模型和分布式优化器，然后使用随机输入数据进行训练
    batch = 5  # 批处理大小
    ntoken = 7  # 词汇表大小
    ninp = 2  # 输入特征维度

    nhid = 3  # 隐藏层维度
    nindices = 6  # 索引数量
    hidden = (  # 初始化隐藏状态
        torch.randn(nlayers, nindices, nhid),  # 随机初始化隐藏状态
        torch.randn(nlayers, nindices, nhid)  # 随机初始化隐藏状态
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)  # 创建RNN模型

    # setup distributed optimizer
    opt = DistributedOptimizer(  # 设置分布式优化器
        optim.SGD,  # 使用SGD优化器
        model.parameter_rrefs(),  # 获取模型参数的远程引用
        lr=0.05,  # 设置学习率
    )

    criterion = torch.nn.CrossEntropyLoss()  # 创建交叉熵损失函数

    def get_next_batch():  # 定义获取下一个批次的函数
        for _ in range(5):  # 生成5个批次
            data = torch.LongTensor(batch, nindices) % ntoken  # 生成随机输入数据
            target = torch.LongTensor(batch, ntoken) % nindices  # 生成随机目标标签
            yield data, target  # 返回数据和目标

    # train for 10 iterations
    for epoch in range(10):  # 训练10个回合
        # create distributed autograd context
        for data, target in get_next_batch():  # 遍历每个批次
            with dist_autograd.context() as context_id:  # 创建分布式自动求导上下文
                hidden[0].detach_()  # 分离第一个隐藏状态
                hidden[1].detach_()  # 分离第二个隐藏状态
                output, hidden = model(data, hidden)  # 前向传播
                loss = criterion(output, target)  # 计算损失
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])  # 反向传播
                # run distributed optimizer
                opt.step(context_id)  # 更新优化器
                # not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training epoch {}".format(epoch))  # 打印当前训练回合


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """  # 包装函数，初始化RPC，调用函数并关闭RPC
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '29500'  # 设置主节点端口
    if rank == 1:  # 如果是训练者
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)  # 初始化RPC
        _run_trainer()  # 运行训练器
    else:  # 如果是参数服务器
        rpc.init_rpc("ps", rank=rank, world_size=world_size)  # 初始化RPC
        # parameter server does nothing  # 参数服务器不执行任何操作
        pass  # 占位符

    # block until all rpcs finish
    rpc.shutdown()  # 关闭RPC


if __name__ == "__main__":
    world_size = 2  # 设置世界规模
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)  # 启动多个进程