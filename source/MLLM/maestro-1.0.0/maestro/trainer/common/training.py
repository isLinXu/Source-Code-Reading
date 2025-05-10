from typing import TypeVar

import lightning
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

TProcessor = TypeVar("TProcessor", bound=ProcessorMixin)
TModel = TypeVar("TModel", bound=PreTrainedModel)


class MaestroTrainer(lightning.LightningModule):
    """Maestro训练器类，继承自PyTorch Lightning的LightningModule。"""
    
    def __init__(self, processor: TProcessor, model: TModel, train_loader: DataLoader, valid_loader: DataLoader):
        """初始化MaestroTrainer。
        
        Args:
            processor (TProcessor): 数据处理器实例
            model (TModel): 模型实例
            train_loader (DataLoader): 训练数据加载器
            valid_loader (DataLoader): 验证数据加载器
        """
        super().__init__()  # 调用父类构造函数
        self.processor = processor  # 存储数据处理器
        self.model = model  # 存储模型
        self.train_loader = train_loader  # 存储训练数据加载器
        self.valid_loader = valid_loader  # 存储验证数据加载器

    def train_dataloader(self):
        """返回训练数据加载器。
        
        Returns:
            DataLoader: 训练数据加载器
        """
        return self.train_loader  # 返回训练数据加载器

    def val_dataloader(self):
        """返回验证数据加载器。
        
        Returns:
            DataLoader: 验证数据加载器
        """
        return self.valid_loader  # 返回验证数据加载器
