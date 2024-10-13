import torch.nn as nn

from . import register_connector
from .base import Connector


# 注册一个名为 'identity' 的连接器
@register_connector('identity')    
class IdentityConnector(Connector):
    def __init__(self, config=None):
        """
        初始化 IdentityConnector 类。

        参数:
            config (optional): 配置参数，本类中未使用。
        """
        super().__init__()                  # 调用父类的构造函数
        self._connector = nn.Identity()     # 创建一个恒等映射层，即输入什么就输出什么

        
    
