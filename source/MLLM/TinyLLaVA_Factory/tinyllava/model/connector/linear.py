import torch.nn as nn
# 从当前包中导入register_connector装饰器和Connector基类
from . import register_connector
from .base import Connector



    
# 使用register_connector装饰器注册一个名为'linear'的连接器
@register_connector('linear')    
class LinearConnector(Connector):
    """
    线性连接器类，继承自Connector基类。

    该类用于创建一个线性层，将输入特征映射到隐藏层。
    """
    def __init__(self, config):
        """
        构造函数，初始化线性连接器。

        :param config: 配置对象，包含vision_hidden_size和hidden_size属性
        """
        super().__init__() # 调用父类的构造函数
        # 创建一个线性层，输入大小为config.vision_hidden_size，输出大小为config.hidden_size
        self._connector =  nn.Linear(config.vision_hidden_size, config.hidden_size)

        
    # @property
    # def config(self):
    #     return {"connector_type": 'linear',
    #             "in_hidden_size": self.in_hidden_size, 
    #             "out_hidden_size": self.out_hidden_size
    #            }
