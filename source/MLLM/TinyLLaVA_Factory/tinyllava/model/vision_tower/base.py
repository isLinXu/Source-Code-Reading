import os

import torch
import torch.nn as nn

from transformers import PreTrainedModel
# from tinyllava.utils.data_utils import get_value_from_kwargs

# 从kwargs中获取指定名称的值，如果不存在则返回None
def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None    # 初始化视觉塔模型
        self._image_processor = None # 初始化图像处理器
        self.config = cfg            # 保存配置信息

    # 加载视觉塔模型
    def load_model(self, vision_tower_name, **kwargs):
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False) # 设置模型参数不需要梯度计算

    # 实际加载模型的方法
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_vision_tower_path')
        if isinstance(self._vision_tower, PreTrainedModel): # hf model # 如果是transformers的预训练模型
            if pretrained_vision_tower_path is not None:
                vision_tower_name = pretrained_vision_tower_path
            self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)      
        else: # nn.Module # 如果是普通的nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')

                # 从加载的权重中提取对应部分的权重
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)

        print("Loading vision tower from ", vision_tower_name)

    # 前向传播方法
    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        # 根据选择的特征选择策略处理特征
        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features

    # vision_tower属性的getter方法
    @property
    def vision_tower(self):
        return self._vision_tower

    # vision_tower属性的setter方法
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower
        
    
