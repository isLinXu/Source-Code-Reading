import random  # 导入random模块，用于生成随机数

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch的分布式模块
import torch.distributed.autograd as dist_autograd  # 导入分布式自动求导模块
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.optim as optim  # 导入优化器模块
from torch.distributed.nn import RemoteModule  # 从分布式模块导入远程模块
from torch.distributed.optim import DistributedOptimizer  # 导入分布式优化器
from torch.distributed.rpc import RRef  # 导入远程引用
from torch.distributed.rpc import TensorPipeRpcBackendOptions  # 导入TensorPipe RPC后端选项
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块

NUM_EMBEDDINGS = 100  # 嵌入数量
EMBEDDING_DIM = 16  # 嵌入维度


class HybridModel(torch.nn.Module):
    r"""
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """
    # 模型由稀疏部分和密集部分组成。
    # 1) 密集部分是一个nn.Linear模块，通过DistributedDataParallel在所有训练者中复制。
    # 2) 稀疏部分是一个远程模块，持有参数服务器上的nn.EmbeddingBag。
    # 该远程模型可以获取参数服务器上嵌入表的远程引用。

    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()  # 调用父类构造函数
        self.remote_emb_module = remote_emb_module  # 保存远程嵌入模块
        self.fc = DDP(torch.nn.Linear(16, 8).cuda(device), device_ids=[device])  # 创建分布式数据并行的全连接层
        self.device = device  # 保存设备信息

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)  # 从远程模块获取嵌入查找结果
        return self.fc(emb_lookup.cuda(self.device))  # 将结果移动到设备并通过全连接层返回


def _run_trainer(remote_emb_module, rank):
    r"""
    Each trainer runs a forward pass which involves an embedding lookup on the
    parameter server and running nn.Linear locally. During the backward pass,
    DDP is responsible for aggregating the gradients for the dense part
    (nn.Linear) and distributed autograd ensures gradients updates are
    propagated to the parameter server.
    """
    # 每个训练者运行一个前向传播，涉及在参数服务器上的嵌入查找和本地运行nn.Linear。
    # 在反向传播期间，DDP负责聚合密集部分（nn.Linear）的梯度，
    # 而分布式自动求导确保梯度更新传播到参数服务器。

    # Setup the model.
    model = HybridModel(remote_emb_module, rank)  # 设置模型

    # Retrieve all model parameters as rrefs for DistributedOptimizer.
    # 获取所有模型参数作为分布式优化器的远程引用

    # Retrieve parameters for embedding table.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()  # 获取嵌入表的参数

    # model.fc.parameters() only includes local parameters.
    # NOTE: Cannot call model.parameters() here,
    # because this will call remote_emb_module.parameters(),
    # which supports remote_parameters() but not parameters().
    for param in model.fc.parameters():  # 遍历全连接层的参数
        model_parameter_rrefs.append(RRef(param))  # 将本地参数的远程引用添加到列表中

    # Setup distributed optimizer
    opt = DistributedOptimizer(  # 设置分布式优化器
        optim.SGD,  # 使用SGD优化器
        model_parameter_rrefs,  # 传入模型参数的远程引用
        lr=0.05,  # 设置学习率
    )

    criterion = torch.nn.CrossEntropyLoss()  # 创建交叉熵损失函数

    def get_next_batch(rank):
        for _ in range(10):  # 生成10个批次
            num_indices = random.randint(20, 50)  # 随机生成索引数量
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)  # 生成随机索引

            # Generate offsets.
            offsets = []  # 初始化偏移量列表
            start = 0  # 初始化起始偏移量
            batch_size = 0  # 初始化批次大小
            while start < num_indices:  # 生成偏移量
                offsets.append(start)  # 添加当前偏移量
                start += random.randint(1, 10)  # 随机增加偏移量
                batch_size += 1  # 增加批次大小

            offsets_tensor = torch.LongTensor(offsets)  # 将偏移量转换为张量
            target = torch.LongTensor(batch_size).random_(8).cuda(rank)  # 生成随机目标标签并移动到设备
            yield indices, offsets_tensor, target  # 返回索引、偏移量和目标标签

    # Train for 100 epochs
    for epoch in range(100):  # 训练100个epoch
        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):  # 遍历每个批次
            with dist_autograd.context() as context_id:  # 创建分布式自动求导上下文
                output = model(indices, offsets)  # 前向传播
                loss = criterion(output, target)  # 计算损失

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])  # 反向传播

                # Tun distributed optimizer
                opt.step(context_id)  # 更新优化器

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training done for epoch {}".format(epoch))  # 打印当前epoch的训练完成信息


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    # 一个包装函数，初始化RPC，调用函数并关闭RPC。

    # We need to use different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()  # 设置RPC后端选项
    rpc_backend_options.init_method = "tcp://localhost:29501"  # 设置初始化方法

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    if rank == 2:  # 如果是主节点
        rpc.init_rpc(  # 初始化RPC
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(  # 创建远程嵌入模块
            "ps",
            torch.nn.EmbeddingBag,  # 使用EmbeddingBag
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),  # 传入参数
            kwargs={"mode": "sum"},  # 设置模式为sum
        )

        # Run the training loop on trainers.
        futs = []  # 初始化未来对象列表
        for trainer_rank in [0, 1]:  # 遍历训练者
            trainer_name = "trainer{}".format(trainer_rank)  # 获取训练者名称
            fut = rpc.rpc_async(  # 异步调用训练者的训练函数
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)  # 将未来对象添加到列表中

        # Wait for all training to finish.
        for fut in futs:  # 等待所有训练完成
            fut.wait()  # 等待未来对象完成
    elif rank <= 1:  # 如果是训练者
        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(  # 初始化进程组
            backend="gloo", rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)  # 获取训练者名称
        rpc.init_rpc(  # 初始化训练者的RPC
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Trainer just waits for RPCs from master.
        # 训练者被动等待主节点的RPC
    else:  # 如果是参数服务器
        rpc.init_rpc(  # 初始化参数服务器的RPC
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        # 参数服务器不执行任何操作
        pass

    # block until all rpcs finish
    rpc.shutdown()  # 关闭RPC


if __name__ == "__main__":
    # 2 trainers, 1 parameter server, 1 master.
    world_size = 4  # 设置世界规模
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)  # 启动多个进程