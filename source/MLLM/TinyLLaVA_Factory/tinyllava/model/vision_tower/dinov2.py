from transformers import Dinov2Model, AutoImageProcessor

from . import register_vision_tower # 导入注册视觉塔的函数
from .base import VisionTower       # 导入VisionTower基类

# 使用@register_vision_tower装饰器注册DINOv2视觉塔
@register_vision_tower('dinov2')      
class DINOv2VisionTower(VisionTower):
    def __init__(self, cfg):
        """
        DINOv2视觉塔的初始化方法。

        :param cfg: 配置对象，包含模型名称或路径等配置信息。
        """
        super().__init__(cfg)                 # 调用父类的初始化方法
        self._vision_tower = Dinov2Model(cfg) # 初始化DINOv2模型
        self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path) # 初始化图像处理器

