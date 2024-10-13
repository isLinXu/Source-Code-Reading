import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    根据配置构建视觉塔模型。

    该函数根据提供的配置选择并构建相应的视觉塔模型。它支持使用字符串指定的模型，
    包括绝对路径、预定义模型（如openai、laion）或特定的视觉塔模型。如果配置中包含
    's2'选项，并且为True，则构建CLIPVisionTowerS2模型；否则，构建CLIPVisionTower模型。

    参数:
        vision_tower_cfg: 配置对象，包含视觉塔的相关配置。
        **kwargs: 其他关键字参数，传递给视觉塔模型。

    返回:
        CLIPVisionTower或CLIPVisionTowerS2模型实例。

    抛出:
        ValueError: 如果视觉塔配置无法识别，则抛出此异常。
    """
    # 获取视觉塔配置
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    # 检查视觉塔配置是否为绝对路径存在或者以openai、laion开头
    is_absolute_path_exists = os.path.exists(vision_tower)

    # 检查是否使用S2版本的视觉塔
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    # 判断视觉塔类型并构建相应的模型
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 如果视觉塔配置无法识别，抛出异常
    raise ValueError(f'Unknown vision tower: {vision_tower}')
