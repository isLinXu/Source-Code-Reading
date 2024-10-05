import os

from ...utils import import_modules

# 定义一个字典，用于存储视觉塔工厂的注册信息
VISION_TOWER_FACTORY = {}

def VisionTowerFactory(vision_tower_name):
    """
    根据传入的视觉塔名称，返回对应的视觉塔模型实例。

    :param vision_tower_name: 视觉塔的名称，可能包含冒号分隔的额外信息
    :return: 对应的视觉塔模型实例
    :raises AssertionError: 如果传入的视觉塔名称未注册，则抛出此异常
    """
    # 只取名称部分，忽略冒号后的内容
    vision_tower_name = vision_tower_name.split(':')[0]
    model = None
    # 遍历注册表，查找匹配的视觉塔名称
    for name in VISION_TOWER_FACTORY.keys():
        if name.lower() in vision_tower_name.lower():
            model = VISION_TOWER_FACTORY[name]
    # 如果没有找到匹配的视觉塔，则抛出异常
    assert model, f"{vision_tower_name} is not registered"
    return model


def register_vision_tower(name):
    """
    注册一个新的视觉塔模型类。

    :param name: 视觉塔模型的名称
    :return: 一个装饰器，用于将视觉塔模型类注册到工厂中
    """
    def register_vision_tower_cls(cls):
        # 如果名称已存在，则返回已有的模型类
        if name in VISION_TOWER_FACTORY:
            return VISION_TOWER_FACTORY[name]
        # 否则，将新的模型类注册到工厂中
        VISION_TOWER_FACTORY[name] = cls
        return cls
    return register_vision_tower_cls


# automatically import any Python files in the models/ directory
# 自动导入models目录下的所有Python文件
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.vision_tower")
