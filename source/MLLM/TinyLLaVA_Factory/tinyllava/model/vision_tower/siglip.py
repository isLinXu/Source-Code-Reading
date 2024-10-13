from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor

from . import register_vision_tower
from .base import VisionTower

# 注册一个名为'siglip'的视觉塔
@register_vision_tower('siglip')      
class SIGLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        """
        初始化SIGLIPVisionTower类

        Args:
            cfg (SiglipVisionConfig): Siglip视觉模型的配置对象
        """
        super().__init__(cfg)
        self._vision_tower = SiglipVisionModel(cfg) # 调用父类的初始化方法
        self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path) # 创建Siglip视觉模型实例

        
#    def forward(self, x, **kwargs):
#        image_features = self._vision_tower(x, output_hidden_states=True)
#        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]


#        return image_features
