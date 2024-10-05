import re

import torch.nn as nn

from . import register_connector
from .base import Connector

# 定义激活函数的类型映射
ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}


# 注册一个名为'mlp'的连接器
@register_connector('mlp')    
class MLPConnector(Connector):
    def __init__(self, config):
        """
        MLP连接器的初始化方法

        :param config: 配置对象，包含连接器的配置信息
        """
        super().__init__()
        # 使用正则表达式匹配mlp的深度和激活函数类型
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.connector_type)
        act_type = config.connector_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))

        # 构建MLP的模块列表
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())    # 添加激活函数
            modules.append(nn.Linear(config.hidden_size, config.hidden_size)) # 添加线性层

        # 将模块列表转换为nn.Sequential模型
        self._connector = nn.Sequential(*modules)


#     @property
#     def config(self):
#         """
#         获取连接器的配置信息
#         :return: 包含连接器配置的字典
#         """
#         return {"connector_type": 'mlp',
#                 "in_hidden_size": self.in_hidden_size, 
#                 "out_hidden_size": self.out_hidden_size
#                }

    
