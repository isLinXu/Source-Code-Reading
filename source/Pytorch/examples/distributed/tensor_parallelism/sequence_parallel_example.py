import os  # 导入os模块，用于与操作系统交互
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

from torch.distributed._tensor import Shard  # 从PyTorch分布式模块导入Shard类

from torch.distributed.tensor.parallel import (  # 从PyTorch分布式模块导入张量并行相关的函数
    parallelize_module,  # 导入并行化模块的函数
    ColwiseParallel,  # 导入列并行化类
    RowwiseParallel,  # 导入行并行化类
)

from log_utils import rank_log, get_logger, verify_min_gpu_count  # 从log_utils模块导入日志相关的函数


# ---- GPU check ------------
_min_gpu_count = 2  # 设置最小GPU数量为2

if not verify_min_gpu_count(min_gpus=_min_gpu_count):  # 验证是否有足够的GPU
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")  # 输出错误信息
    sys.exit()  # 退出程序
# ---------------------------


from torch.distributed._tensor.device_mesh import init_device_mesh  # 从PyTorch分布式模块导入初始化设备网格的函数



"""
This is the script to test Sequence Parallel(SP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of sequence parallel, which was proposed in paper:

https://arxiv.org/pdf/2205.05198.pdf.

Like tensor parallel, we parallelize the first linear layer by column
and also parallelize the second linear layer by row. But the input in each rank
now is different so that we need one all-gather for input and one reduce-scatter
in the end of the second linear layer.
"""  # 说明该脚本用于在玩具模型上测试序列并行(SP)，展示从前向传播、反向传播到优化的完整工作流程


class ToyModel(nn.Module):  # 定义玩具模型类
    """MLP based model"""  # 基于多层感知机的模型

    def __init__(self):  # 初始化模型
        super().__init__()  # 调用父类构造函数
        self.in_proj = nn.Linear(10, 32)  # 创建输入线性层
        self.relu = nn.ReLU()  # 创建ReLU激活层
        self.out_proj = nn.Linear(32, 5)  # 创建输出线性层

    def forward(self, x):  # 定义前向传播函数
        return self.out_proj(self.relu(self.in_proj(x)))  # 通过输入层、激活层和输出层进行前向传播


"""
Main body of the demo of a basic version of sequence parallel by using
PyTorch native APIs.
"""  # 使用PyTorch原生API演示序列并行的基本版本
logger = get_logger()  # 获取日志记录器

# create a device mesh based on the given world_size.
device_mesh = init_device_mesh(  # 初始化设备网格
    device_type="cuda",  # 指定设备类型为CUDA
    mesh_shape=(int(os.environ["WORLD_SIZE"]),)  # 根据世界大小设置网格形状
)

_rank = device_mesh.get_rank()  # 获取当前进程的排名

print(f"Starting PyTorch Sequence Parallel example on rank {_rank}.")  # 输出当前进程的启动信息

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")  # 记录设备网格创建信息

# create model and move it to GPU.  Init_device_mesh has already assigned gpu ids...
model = ToyModel().to("cuda")  # 创建模型并移动到GPU

# Custom parallelization plan for the model
sp_model = parallelize_module(  # 对模型进行并行化
    module=model,
    device_mesh=device_mesh,
    parallelize_plan={  # 定义并行化计划
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),  # 对输入线性层进行列并行化
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),  # 对输出线性层进行行并行化
    },
)


# Create a optimizer for the parallelized module.
lr = 0.25  # 设置学习率
optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr, foreach=True)  # 创建AdamW优化器


# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 10  # 设置迭代次数
rank_log(_rank, logger, "Sequence Parallel training starting...")  # 记录训练开始信息

for i in range(num_iters):  # 遍历每次迭代
    # For SP, input can be different across all ranks.
    inp = torch.rand(20, 10, device="cuda")  # 生成随机输入张量
    output = sp_model(inp)  # 通过模型进行前向传播
    output.sum().backward()  # 计算梯度
    optimizer.step()  # 更新优化器
    rank_log(_rank, logger, f"Sequence Parallel iter {i} completed")  # 记录当前迭代完成信息

rank_log(_rank, logger, "Sequence Parallel training completed!")  # 记录训练成功完成的信息