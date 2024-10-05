from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from . import register_vision_tower
from .base import VisionTower

# 注册一个名为 'clip' 的视觉塔
@register_vision_tower('clip')      
class CLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        """
            初始化 CLIPVisionTower 类。

            :param cfg: 配置对象，包含模型名称或路径等信息。
        """
        super().__init__(cfg) # 调用父类的初始化方法
        self._vision_tower = CLIPVisionModel(cfg) # 创建 CLIP 视觉模型实例
        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path) # 加载预训练的图像处理器
  

