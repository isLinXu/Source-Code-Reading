import argparse  # 导入 argparse 库，用于处理命令行参数
import os  # 导入 os 库，用于与操作系统交互
import sys  # 导入 sys 库，用于访问 Python 解释器的变量和函数
import tempfile  # 导入 tempfile 库，用于创建临时文件
from urllib.parse import urlparse  # 导入 urlparse，用于解析 URL

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入分布式训练模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块

from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块


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


def demo_basic(local_world_size, local_rank):
    """Demonstrate basic distributed training.
    演示基本的分布式训练。
    
    Args:
        local_world_size: 本地世界大小
        local_rank: 本地进程的排名
    """
    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size  # 每个进程可用的 GPU 数量
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))  # 当前进程使用的 GPU ID

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    )  # 打印当前进程的信息

    model = ToyModel().cuda(device_ids[0])  # 将模型移动到指定的 GPU
    ddp_model = DDP(model, device_ids)  # 使用 DDP 封装模型以支持分布式训练

    loss_fn = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # 定义 SGD 优化器

    optimizer.zero_grad()  # 清除梯度
    outputs = ddp_model(torch.randn(20, 10))  # 生成随机输入并通过模型计算输出
    labels = torch.randn(20, 5).to(device_ids[0])  # 生成随机标签并移动到 GPU
    loss_fn(outputs, labels).backward()  # 计算损失并反向传播
    optimizer.step()  # 更新优化器


def spmd_main(local_world_size, local_rank):
    """Main function for single process multi-device training.
    单进程多设备训练的主函数。
    
    Args:
        local_world_size: 本地世界大小
        local_rank: 本地进程的排名
    """
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]  # 从环境变量中获取初始化进程组所需的参数
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    
    if sys.platform == "win32":  # 如果在 Windows 平台上
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        if "INIT_METHOD" in os.environ.keys():  # 检查是否设置了 INIT_METHOD
            print(f"init_method is {os.environ['INIT_METHOD']}")  # 打印初始化方法
            url_obj = urlparse(os.environ["INIT_METHOD"])  # 解析 URL
            if url_obj.scheme.lower() != "file":  # 只支持 FileStore
                raise ValueError("Windows only supports FileStore")  # 抛出错误
            else:
                init_method = os.environ["INIT_METHOD"]  # 使用指定的初始化方法
        else:
            # It is a example application, For convenience, we create a file in temp dir.
            temp_dir = tempfile.gettempdir()  # 获取临时目录
            init_method = f"file:///{os.path.join(temp_dir, 'ddp_example')}"  # 创建文件路径
        dist.init_process_group(backend="gloo", init_method=init_method, rank=int(env_dict["RANK"]), world_size=int(env_dict["WORLD_SIZE"]))  # 初始化进程组
    else:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
        dist.init_process_group(backend="nccl")  # 初始化进程组

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )  # 打印进程组信息

    demo_basic(local_world_size, local_rank)  # 调用演示函数

    # Tear down the process group
    dist.destroy_process_group()  # 销毁进程组


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # This is passed in via launch.py
    parser.add_argument("--local_rank", type=int, default=0)  # 本地进程的排名
    # This needs to be explicitly passed in
    parser.add_argument("--local_world_size", type=int, default=1)  # 本地世界大小
    args = parser.parse_args()  # 解析命令行参数
    # The main entry point is called directly without using subprocess
    spmd_main(args.local_world_size, args.local_rank)  # 调用主函数