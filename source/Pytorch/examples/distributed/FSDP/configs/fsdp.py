from dataclasses import dataclass, field  # 从dataclasses模块导入dataclass和field
from typing import ClassVar  # 从typing模块导入ClassVar类型
from torch.distributed.fsdp import ShardingStrategy  # 从PyTorch的分布式库中导入分片策略
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # 从PyTorch的全分片数据并行库中导入状态字典类型

@dataclass  # 使用dataclass装饰器定义数据类
class fsdp_config:  # 定义fsdp_config类
    mixed_precision: bool = True  # 是否使用混合精度，默认为True
    use_fp16: bool = False  # 是否使用FP16，默认为False
    seed: int = 42  # 随机种子，默认为42
    fsdp_activation_checkpointing: bool = False  # 是否启用FSDP激活检查点，默认为False
    limit_all_gathers: bool = True  # 是否限制所有收集，默认为True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  # 设置分片策略为FULL_SHARD（完全分片），可选HYBRID_SHARD或SHARD_GRAD_OP
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT  # 设置检查点类型为FULL_STATE_DICT（完整状态字典），可选SHARDED_STATE_DICT以避免内存不足（OOM）
    save_optimizer: bool = False  # 是否保存优化器，默认为False