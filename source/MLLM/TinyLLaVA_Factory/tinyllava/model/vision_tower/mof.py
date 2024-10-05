import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, Dinov2Model, AutoConfig

from . import register_vision_tower
from .base import VisionTower




# MoF类继承自nn.Module，用于结合CLIPVisionModel和Dinov2Model的特征
class MoF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化CLIPVisionModel
        self.clip = CLIPVisionModel(cfg)

        # 使用AutoConfig从预训练模型加载配置，并初始化Dinov2Model
        cfg_dinov2 = AutoConfig.from_pretrained(cfg.model_name_or_path2)
        self.dinov2 = Dinov2Model(cfg_dinov2)


#     def enable_input_require_grads(self):
#         def make_inputs_require_grad(module, input, output):
#             output.requires_grads()

#         if hasattr(self.clip, 'enable_input_require_grads'):
#             self.clip.enable_input_require_grads()
#         else:
#             self.clip.get_input_embeddings(make_inputs_require_grad)

#         if hasattr(self.dinov2, 'enable_input_require_grads'):
#             self.dinov2.enable_input_require_grads()
#         else:
#             self.dinov2.get_input_embeddings(make_inputs_require_grad)

    # 定义forward方法，用于前向传播
    def forward(self, x, **kwargs):
        # 获取CLIP模型的特征
        image_features_clip = self.clip(x, output_hidden_states=True)
        image_features_clip = image_features_clip.hidden_states[kwargs.get('vision_feature_layer', -2)]
        # 获取Dinov2模型的特征
        image_features_dinov2 = self.dinov2(x, output_hidden_states=True)
        image_features_dinov2 = image_features_dinov2.hidden_states[kwargs.get('vision_feature_layer', -2)]
        # 根据策略选择特征
        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features_clip = image_features_clip[:, 1:]
            image_features_dinov2 = image_features_dinov2[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features_clip = image_features_clip
            image_features_dinov2 = image_features_dinov2
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        # 返回组合的特征
        image_features = image_features_clip, image_features_dinov2

        return image_features





# MoFVisionTower类继承自VisionTower，用于加载模型并进行前向传播
@register_vision_tower('mof')      
class MoFVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 初始化MoF模型
        self._vision_tower = MoF(cfg)
        # 初始化CLIPImageProcessor
        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)

    # 定义_load_model方法，用于加载预训练模型
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)
        if pretrained_vision_tower_path is None:
            # 分别加载CLIPVisionModel和Dinov2Model的预训练权重
            model_name_or_path_dinov2 = kwargs.pop('model_name_or_path2')
            self._vision_tower.clip = self._vision_tower.clip.from_pretrained(vision_tower_name, **kwargs)
            self._vision_tower.dinov2 = self._vision_tower.dinov2.from_pretrained(model_name_or_path_dinov2, **kwargs)
            print("Loading vision tower1 from ", vision_tower_name)
            print("Loading vision tower2 from ", model_name_or_path_dinov2)
        else: # nn.Module
            # 加载整个MoF模型的预训练权重
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)
            print("Loading vision tower from ", pretrained_vision_tower_path)

    # 定义forward方法，用于前向传播
    def forward(self, x, **kwargs):
        device = x.data.device
        self.to(device)
        return self._vision_tower(x, **kwargs)


