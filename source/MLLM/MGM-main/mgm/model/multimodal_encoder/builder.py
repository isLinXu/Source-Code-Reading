# 导入操作系统模块和不同视觉编码器类
import os
from .clip_encoder import CLIPVisionTower
from .eva_encoder import EVAVisionTower
from .openclip_encoder import OpenCLIPVisionTower

# 主视觉编码器构建函数
def build_vision_tower(vision_tower_cfg, **kwargs):
    # 从配置中获取视觉模型路径，优先取mm_vision_tower，其次取vision_tower
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # 获取图像处理器路径，默认使用../processor/clip-patch14-224
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "../processor/clip-patch14-224"))
    
    # 检查视觉模型路径是否存在
    if not os.path.exists(vision_tower):
        raise ValueError(f'Not find vision tower: {vision_tower}')

    # 根据模型名称选择对应的视觉编码器
    if "openai" in vision_tower.lower() or "ShareGPT4V" in vision_tower:
        # 使用OpenAI的CLIP视觉编码器
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        # 使用EVA视觉编码器（来自LAVIS库）
        return EVAVisionTower(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    else:
        # 无法识别的视觉模型抛出异常
        raise ValueError(f'Unknown vision tower: {vision_tower}')

# 辅助视觉编码器构建函数（用于多模态中的第二个视觉编码器）
def build_vision_tower_aux(vision_tower_cfg, **kwargs):
    # 获取辅助视觉模型路径配置
    vision_tower_aux = getattr(vision_tower_cfg, 'mm_vision_tower_aux', getattr(vision_tower_cfg, 'vision_tower_aux', None))
    
    # 检查辅助视觉模型路径是否存在
    if not os.path.exists(vision_tower_aux):
        raise ValueError(f'Not find vision tower: {vision_tower_aux}')

    # 根据模型名称选择辅助编码器类型
    if "openclip" in vision_tower_aux.lower():
        # 使用OpenCLIP编码器
        return OpenCLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    elif "openai" in vision_tower_aux.lower():
        # 使用OpenAI的CLIP编码器
        return CLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    else:
        # 无法识别的辅助视觉模型抛出异常
        raise ValueError(f'Unknown vision tower: {vision_tower_aux}')