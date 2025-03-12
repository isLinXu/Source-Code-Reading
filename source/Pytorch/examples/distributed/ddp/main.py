import os  # 导入 os 库，用于与操作系统交互
import tempfile  # 导入 tempfile 库，用于创建临时文件
import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入分布式训练模块
import torch.multiprocessing as mp  # 导入多进程模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块

from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块


def setup(rank, world_size):
    """Setup the distributed training environment.
    设置分布式训练环境。
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
    """
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '12355'  # 设置主节点端口

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  # 初始化进程组


def cleanup():
    """Clean up the distributed training environment.
    清理分布式训练环境。
    """
    dist.destroy_process_group()  # 销毁进程组


class ToyModel(nn.Module):  # 定义一个简单的模型类
    def __init__(self):
        """Initialize the ToyModel.
        初始化 ToyModel。
        """
        super(ToyModel, self).__init__()  # 调用父类构造函数
        self.net1 = nn.Linear(10, 10)  # 定义第一层线性层
        self.relu = nn.ReLU()  # 定义 ReLU 激活函数
        self.net2 = nn.Linear(10, 5)  # 定义第二层线性层

    def forward(self, x):  # 前向传播
        """Forward pass of the ToyModel.
        ToyModel 的前向传播。
        
        Args:
            x: 输入张量
        Returns:
            输出张量
        """
        return self.net2(self.relu(self.net1(x)))  # 通过网络计算输出


def demo_basic(rank, world_size):
    """Demonstrate basic distributed training.
    演示基本的分布式训练。
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
    """
    print(f"Running basic DDP example on rank {rank}.")  # 打印当前进程的信息
    setup(rank, world_size)  # 设置分布式环境

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)  # 将模型移动到指定的 GPU
    ddp_model = DDP(model, device_ids=[rank])  # 使用 DDP 封装模型以支持分布式训练

    loss_fn = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # 定义 SGD 优化器

    optimizer.zero_grad()  # 清除梯度
    outputs = ddp_model(torch.randn(20, 10))  # 生成随机输入并通过模型计算输出
    labels = torch.randn(20, 5).to(rank)  # 生成随机标签并移动到 GPU
    loss_fn(outputs, labels).backward()  # 计算损失并反向传播
    optimizer.step()  # 更新优化器

    cleanup()  # 清理分布式环境


def run_demo(demo_fn, world_size):
    """Run the demo function using multiple processes.
    使用多个进程运行演示函数。
    
    Args:
        demo_fn: 演示函数
        world_size: 总进程数
    """
    mp.spawn(demo_fn,  # 使用多进程启动演示函数
             args=(world_size,),  # 传递参数
             nprocs=world_size,  # 进程数量
             join=True)  # 等待所有进程完成


def demo_checkpoint(rank, world_size):
    """Demonstrate distributed training with checkpointing.
    演示带有检查点的分布式训练。
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
    """
    print(f"Running DDP checkpoint example on rank {rank}.")  # 打印当前进程的信息
    setup(rank, world_size)  # 设置分布式环境

    model = ToyModel().to(rank)  # 将模型移动到指定的 GPU
    ddp_model = DDP(model, device_ids=[rank])  # 使用 DDP 封装模型以支持分布式训练

    loss_fn = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # 定义 SGD 优化器

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"  # 创建检查点路径
    if rank == 0:  # 只有 rank 0 进程保存模型
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)  # 保存模型参数

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()  # 确保进程 1 在进程 0 保存模型后加载模型
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # 配置设备映射
    ddp_model.load_state_dict(  # 加载模型参数
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()  # 清除梯度
    outputs = ddp_model(torch.randn(20, 10))  # 生成随机输入并通过模型计算输出
    labels = torch.randn(20, 5).to(rank)  # 生成随机标签并移动到 GPU
    loss_fn(outputs, labels).backward()  # 计算损失并反向传播
    optimizer.step()  # 更新优化器

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()  # 确保所有进程完成读取检查点

    if rank == 0:  # 只有 rank 0 进程删除检查点文件
        os.remove(CHECKPOINT_PATH)  # 删除检查点文件

    cleanup()  # 清理分布式环境


class ToyMpModel(nn.Module):  # 定义模型并行的简单模型类
    def __init__(self, dev0, dev1):  # 初始化模型
        """Initialize the ToyMpModel.
        初始化 ToyMpModel。
        
        Args:
            dev0: 第一个设备
            dev1: 第二个设备
        """
        super(ToyMpModel, self).__init__()  # 调用父类构造函数
        self.dev0 = dev0  # 保存第一个设备
        self.dev1 = dev1  # 保存第二个设备
        self.net1 = torch.nn.Linear(10, 10).to(dev0)  # 在 dev0 上定义第一层线性层
        self.relu = torch.nn.ReLU()  # 定义 ReLU 激活函数
        self.net2 = torch.nn.Linear(10, 5).to(dev1)  # 在 dev1 上定义第二层线性层

    def forward(self, x):  # 前向传播
        """Forward pass of the ToyMpModel.
        ToyMpModel 的前向传播。
        
        Args:
            x: 输入张量
        Returns:
            输出张量
        """
        x = x.to(self.dev0)  # 将输入移动到 dev0
        x = self.relu(self.net1(x))  # 通过第一层计算输出并应用 ReLU
        x = x.to(self.dev1)  # 将输出移动到 dev1
        return self.net2(x)  # 通过第二层计算输出


def demo_model_parallel(rank, world_size):
    """Demonstrate model parallelism with DDP.
    演示带有 DDP 的模型并行。
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
    """
    print(f"Running DDP with model parallel example on rank {rank}.")  # 打印当前进程的信息
    setup(rank, world_size)  # 设置分布式环境

    # setup mp_model and devices for this process
    dev0 = rank * 2  # 设置第一个设备 ID
    dev1 = rank * 2 + 1  # 设置第二个设备 ID
    mp_model = ToyMpModel(dev0, dev1)  # 实例化模型并行模型
    ddp_mp_model = DDP(mp_model)  # 使用 DDP 封装模型以支持分布式训练

    loss_fn = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)  # 定义 SGD 优化器

    optimizer.zero_grad()  # 清除梯度
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))  # 生成随机输入并通过模型计算输出
    labels = torch.randn(20, 5).to(dev1)  # 生成随机标签并移动到 dev1
    loss_fn(outputs, labels).backward()  # 计算损失并反向传播
    optimizer.step()  # 更新优化器

    cleanup()  # 清理分布式环境


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()  # 获取可用的 GPU 数量
    if n_gpus < 8:  # 如果 GPU 数量少于 8
        print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")  # 打印错误信息
    else:
        run_demo(demo_basic, 8)  # 运行基本的 DDP 示例
        run_demo(demo_checkpoint, 8)  # 运行检查点示例
        run_demo(demo_model_parallel, 4)  # 运行模型并行示例