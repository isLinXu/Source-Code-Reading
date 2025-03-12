import torch  # 导入PyTorch库

from torch.distributed.fsdp import (  # 从PyTorch的分布式库中导入相关类
    # FullyShardedDataParallel as FSDP,  # 完全分片的数据并行（注释掉的代码）
    # CPUOffload,  # CPU卸载（注释掉的代码）
    MixedPrecision,  # 导入混合精度类
    # BackwardPrefetch,  # 后向预取（注释掉的代码）
    # ShardingStrategy,  # 分片策略（注释掉的代码）
)

# requires grad scaler in main loop  # 在主循环中需要梯度缩放器
fpSixteen = MixedPrecision(  # 创建fp16混合精度策略
    param_dtype=torch.float16,  # 参数数据类型为float16
    # Gradient communication precision.  # 梯度通信精度
    reduce_dtype=torch.float16,  # 归约数据类型为float16
    # Buffer precision.  # 缓冲区精度
    buffer_dtype=torch.float16,  # 缓冲区数据类型为float16
)

bfSixteen = MixedPrecision(  # 创建bfloat16混合精度策略
    param_dtype=torch.bfloat16,  # 参数数据类型为bfloat16
    # Gradient communication precision.  # 梯度通信精度
    reduce_dtype=torch.bfloat16,  # 归约数据类型为bfloat16
    # Buffer precision.  # 缓冲区精度
    buffer_dtype=torch.bfloat16,  # 缓冲区数据类型为bfloat16
)

bfSixteen_working = MixedPrecision(  # 创建工作中的bfloat16混合精度策略
    param_dtype=torch.float32,  # 参数数据类型为float32
    reduce_dtype=torch.bfloat16,  # 归约数据类型为bfloat16
    buffer_dtype=torch.bfloat16,  # 缓冲区数据类型为bfloat16
)

fp32_policy = MixedPrecision(  # 创建fp32混合精度策略
    param_dtype=torch.float32,  # 参数数据类型为float32
    reduce_dtype=torch.float32,  # 归约数据类型为float32
    buffer_dtype=torch.float32,  # 缓冲区数据类型为float32
)