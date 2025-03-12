# holds various wrapping policies for fsdp  # 保存各种FSDP的包装策略

import torch.distributed as dist  # 导入PyTorch的分布式模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch  # 导入PyTorch库

from transformers.models.t5.modeling_t5 import T5Block  # 从transformers库中导入T5模型块

from torch.distributed.fsdp.fully_sharded_data_parallel import (  # 从PyTorch的全分片数据并行库中导入相关类
    FullyShardedDataParallel as FSDP,  # 完全分片的数据并行
    CPUOffload,  # CPU卸载
    BackwardPrefetch,  # 后向预取
    MixedPrecision,  # 混合精度
)
from torch.distributed.fsdp.wrap import (  # 从FSDP的包装模块中导入相关函数
    transformer_auto_wrap_policy,  # 转换器自动包装策略
    size_based_auto_wrap_policy,  # 基于大小的自动包装策略
    enable_wrap,  # 启用包装
    wrap,  # 包装函数
)

import functools  # 导入functools模块
from typing import Type  # 从typing模块导入Type类型


def get_size_policy(min_params=1e8):  # 定义获取大小策略的函数，参数为最小参数数量
    num_wrap_policy = functools.partial(  # 创建一个部分应用的函数
        size_based_auto_wrap_policy, min_num_params=min_params  # 使用基于大小的自动包装策略
    )
    return num_wrap_policy  # 返回包装策略


def get_t5_wrapper():  # 定义获取T5包装器的函数
    """we register our main layer class and use the fsdp transformer wrapping policy  # 注册主要层类并使用FSDP转换器包装策略
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers  # 确保嵌入层位于根FSDP单元中以便共享访问，并且FSDP单元映射到转换器层
    """
    # ====   use new transformer wrapper  # 使用新的转换器包装器

    t5_auto_wrap_policy = functools.partial(  # 创建一个部分应用的函数
        transformer_auto_wrap_policy,  # 使用转换器自动包装策略
        transformer_layer_cls={  # 注册的转换器层类
            T5Block,  # T5模型块
        },
    )

    return t5_auto_wrap_policy  # 返回T5自动包装策略