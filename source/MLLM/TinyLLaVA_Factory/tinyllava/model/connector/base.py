import os

import torch
import torch.nn as nn

# 定义一个名为Connector的神经网络模块，继承自nn.Module
class Connector(nn.Module):
    # 初始化方法，定义了一个_connector属性，初始为None
    def __init__(self, config=None):
        super().__init__()
        self._connector = None

    # 加载模型的方法，接受关键字参数
    def load_model(self, **kwargs):
        # 获取预训练连接器的路径，如果没有提供则默认为None
        pretrained_connector_path = kwargs.get('pretrained_connector_path', None)
        # 如果提供了预训练路径
        if pretrained_connector_path is not None:
            # 构造预训练模型文件的完整路径
            pretrained_connector_path = os.path.join(pretrained_connector_path, 'pytorch_model.bin')
            # 加载预训练模型的权重
            connector_weights = torch.load(pretrained_connector_path, map_location='cpu')

            # 定义一个内部函数get_w，用于从加载的权重中提取与_connector相关的部分
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # 将提取的权重加载到_connector中
            self._connector.load_state_dict(get_w(connector_weights, '_connector'))
            # 打印加载信息
            print(f'Loading connector from {pretrained_connector_path}...')

        # 将_connector的所有参数设置为不需要梯度计算，即不参与后续的训练
        for p in self._connector.parameters():
            p.requires_grad = False

    # 前向传播方法，将输入x传递给_connector并返回结果
    def forward(self, x):
        return self._connector(x)
        

  
