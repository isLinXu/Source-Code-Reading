import torch
import torch.nn as nn

from . import register_connector
from .base import Connector


    
# 定义MoFMLP类，继承自nn.Module
class MoFMLP(nn.Module):
    def __init__(self, config):
        """
        初始化MoFMLP模型。

        Args:
            config: 配置对象，包含模型的参数设置。
        """
        super().__init__()
        # 定义Clip网络的模块列表
        modules_clip = [nn.Linear(config.vision_hidden_size, config.hidden_size),   # 线性层
                    nn.GELU(),                                                      # GELU激活函数
                    nn.Linear(config.hidden_size, config.hidden_size)               # 线性层
                    ]
        # 定义DiNov2网络的模块列表
        modules_dinov2 = [nn.Linear(config.vision_hidden_size, config.hidden_size), # 线性层
                    nn.GELU(),                                                      # GELU激活函数
                    nn.Linear(config.hidden_size, config.hidden_size)               # 线性层
                    ]
        # 使用nn.Sequential将模块列表组合成网络
        self.clip = nn.Sequential(*modules_clip)
        self.dinov2 = nn.Sequential(*modules_dinov2)



    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量，包含两个特征图。

        Returns:
            merged_features: 合并后的特征图。
        """
        # 分别通过Clip和DiNov2网络
        image_features_clip = self.clip(x[0])
        image_features_dinov2 = self.dinov2(x[1])
        # 获取批量大小、总长度和维度
        bs = image_features_clip.size(0)
        total_len = image_features_clip.size(1)+image_features_dinov2.size(1)
        dim = image_features_clip.size(-1)
        # 创建一个空的合并特征图张量，并将设备设置为输入张量的设备和数据类型
        merged_features = torch.empty(bs, total_len, dim).to(device=x[0].device, dtype=x[0].dtype)
        # 将Clip和DiNov2的特征图按顺序合并到merged_features中
        merged_features[:,0::2] = image_features_clip
        merged_features[:,1::2] = image_features_dinov2

        return merged_features
    
    

# 注册MoFMLP连接器
@register_connector('mof_mlp')    
class MoFMLPConnector(Connector):
    def __init__(self, config):
        """
        初始化MoFMLP连接器。

        Args:
            config: 配置对象，包含连接器的参数设置。
        """
        super().__init__()
        # 创建MoFMLP模型实例
        self._connector = MoFMLP(config)
