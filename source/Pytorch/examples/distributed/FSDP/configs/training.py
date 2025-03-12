from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import ClassVar  # 从typing模块导入ClassVar类型

@dataclass  # 使用dataclass装饰器定义数据类
class train_config:  # 定义train_config类
    model_name: str = "t5-base"  # 模型名称，默认为"t5-base"
    run_validation: bool = True  # 是否运行验证，默认为True
    batch_size_training: int = 4  # 训练批量大小，默认为4
    num_workers_dataloader: int = 2  # 数据加载器的工作线程数，默认为2
    lr: float = 0.002  # 学习率，默认为0.002
    weight_decay: float = 0.0  # 权重衰减，默认为0.0
    gamma: float = 0.85  # 学习率调度的衰减因子，默认为0.85
    use_fp16: bool = False  # 是否使用FP16，默认为False
    mixed_precision: bool = True  # 是否使用混合精度，默认为True
    save_model: bool = False  # 是否保存模型，默认为False