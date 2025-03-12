import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数
import os  # 导入os模块，用于与操作系统交互
import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch的分布式模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块

from log_utils import rank_log, get_logger, verify_min_gpu_count  # 从log_utils模块导入日志相关的函数

# ---- GPU check ------------
_min_gpu_count = 4  # 设置最小GPU数量为4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):  # 验证是否有足够的GPU
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")  # 输出错误信息
    sys.exit()  # 退出程序
# ---------------------------

from llama2_model import Transformer, ModelArgs  # 从llama2_model模块导入Transformer类和ModelArgs类

from torch.distributed.device_mesh import init_device_mesh  # 从PyTorch分布式模块导入初始化设备网格的函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 从PyTorch分布式模块导入完全分片数据并行类
from torch.distributed._tensor import Shard, Replicate  # 从PyTorch分布式模块导入Shard和Replicate类
from torch.distributed.tensor.parallel import (  # 从PyTorch分布式模块导入张量并行相关的函数
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""  # 说明该脚本用于测试2D并行，结合张量/序列并行和完全分片数据并行

tp_size = 2  # 设置张量并行的大小
logger = get_logger()  # 获取日志记录器

# understand world topology
_rank = int(os.environ["RANK"])  # 获取当前进程的排名
_world_size = int(os.environ["WORLD_SIZE"])  # 获取世界大小

print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")  # 输出当前进程的启动信息
assert (
    _world_size % tp_size == 0
), f"World size {_world_size} needs to be divisible by TP size {tp_size}"  # 确保世界大小可以被张量并行大小整除

# create a sharding plan based on the given world_size.
dp_size = _world_size // tp_size  # 计算数据并行的大小

# Create a device mesh with 2 dimensions.
# First dim is the data parallel dimension
# Second dim is the tensor parallel dimension.
device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))  # 初始化设备网格

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")  # 记录设备网格创建信息
tp_mesh = device_mesh["tp"]  # 获取张量并行的设备网格
dp_mesh = device_mesh["dp"]  # 获取数据并行的设备网格

# For TP, input needs to be same across all TP ranks.
# while for SP, input can be different across all ranks.
# We will use dp_rank for setting the random seed
# to mimic the behavior of the dataloader.
dp_rank = dp_mesh.get_local_rank()  # 获取数据并行的本地排名

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)  # 创建模型参数配置

model = Transformer.from_model_args(simple_llama2_config).to("cuda")  # 创建模型并移动到GPU

# init model weights
model.init_weights()  # 初始化模型权重

# parallelize the first embedding and the last linear out projection
model = parallelize_module(  # 对模型进行并行化
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(  # 对词嵌入层进行行并行化
            input_layouts=Replicate(),  # 输入布局为复制
            output_layouts=Shard(1),  # 输出布局为分片
        ),
        "norm": SequenceParallel(),  # 对归一化层进行序列并行化
        "output": ColwiseParallel(  # 对输出层进行列并行化
            input_layouts=Shard(1),  # 输入布局为分片
            output_layouts=Replicate()  # 输出布局为复制
        ),
    }
)

for layer_id, transformer_block in enumerate(model.layers):  # 遍历模型的每个变换块
    layer_tp_plan = {  # 定义每个变换块的并行化计划
        "attention_norm": SequenceParallel(),  # 对注意力归一化层进行序列并行化
        "attention": PrepareModuleInput(  # 准备注意力模块的输入
            input_layouts=(Shard(1), None),  # 输入布局为分片
            desired_input_layouts=(Replicate(), None),  # 期望输入布局为复制
        ),
        "attention.wq": ColwiseParallel(),  # 对注意力的查询权重进行列并行化
        "attention.wk": ColwiseParallel(),  # 对注意力的键权重进行列并行化
        "attention.wv": ColwiseParallel(),  # 对注意力的值权重进行列并行化
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),  # 对注意力的输出权重进行行并行化
        "ffn_norm": SequenceParallel(),  # 对前馈网络的归一化层进行序列并行化
        "feed_forward": PrepareModuleInput(  # 准备前馈模块的输入
            input_layouts=(Shard(1),),  # 输入布局为分片
            desired_input_layouts=(Replicate(),),  # 期望输入布局为复制
        ),
        "feed_forward.w1": ColwiseParallel(),  # 对前馈的第一层权重进行列并行化
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),  # 对前馈的第二层权重进行行并行化
        "feed_forward.w3": ColwiseParallel(),  # 对前馈的第三层权重进行列并行化
    }

    # Adjust attention module to use the local number of heads
    attn_layer = transformer_block.attention  # 获取当前变换块的注意力层
    attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()  # 调整注意力头的数量
    attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()  # 调整键值头的数量

    # Custom parallelization plan for the model
    parallelize_module(  # 对变换块进行并行化
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan  # 使用定义的并行化计划
    )

# Init FSDP using the dp device mesh
sharded_model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)  # 使用数据并行设备网格初始化完全分片数据并行模型

rank_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")  # 记录并行化后的模型信息

# Create an optimizer for the parallelized and sharded model.
lr = 3e-3  # 设置学习率
rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")  # 记录优化器创建信息
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)  # 创建AdamW优化器

# Training loop:
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
rank_log(_rank, logger, "\nStarting 2D training...")  # 记录训练开始信息
num_iterations = 10  # 设置迭代次数
batch_size = 2  # 设置批处理大小

for i in range(num_iterations):  # 遍历每次迭代
    # seeding with dp_rank to ensure identical inputs for TP groups
    torch.manual_seed(i + dp_rank)  # 设置随机种子以确保TP组的输入相同
    inp = torch.randint(32000, (8, 256), device="cuda")  # 生成随机输入张量

    output = sharded_model(inp)  # 通过模型进行前向传播
    output.sum().backward()  # 计算梯度
    optimizer.step()  # 更新优化器
    rank_log(_rank, logger, f"2D iter {i} complete")  # 记录当前迭代完成信息

rank_log(_rank, logger, "2D training successfully completed!")  # 记录训练成功完成的信息