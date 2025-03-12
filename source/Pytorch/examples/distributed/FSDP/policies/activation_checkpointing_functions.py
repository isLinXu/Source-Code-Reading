import torch  # 导入PyTorch库
import os  # 导入操作系统模块
import torch.distributed as dist  # 导入PyTorch的分布式模块
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # 从分布式检查点库中导入相关类
    checkpoint_wrapper,  # 检查点包装器
    CheckpointImpl,  # 检查点实现
    apply_activation_checkpointing,  # 应用激活检查点的函数
)

from transformers.models.t5.modeling_t5 import T5Block  # 从transformers库中导入T5模型块

from functools import partial  # 从functools模块导入partial函数，用于部分应用函数

# 创建一个非重入的检查点包装器
non_reentrant_wrapper = partial(  # 使用partial创建一个新的函数
    checkpoint_wrapper,  # 检查点包装器
    offload_to_cpu=False,  # 不将数据卸载到CPU
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,  # 设置检查点实现为非重入
)

# 检查子模块是否为T5Block的函数
check_fn = lambda submodule: isinstance(submodule, T5Block)  # 使用lambda表达式定义检查函数


def apply_fsdp_checkpointing(model):  # 定义应用FSDP检查点的函数
    """apply activation checkpointing to model  # 将激活检查点应用于模型
    returns None as model is updated directly  # 返回None，因为模型会直接更新
    """
    print(f"--> applying fdsp activation checkpointing...")  # 打印应用FSDP激活检查点的信息

    apply_activation_checkpointing(  # 应用激活检查点的函数
        model,  # 传入模型
        checkpoint_wrapper_fn=non_reentrant_wrapper,  # 使用非重入的检查点包装器
        check_fn=check_fn  # 使用检查函数
    )