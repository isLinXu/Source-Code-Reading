import os

from ..utils import import_modules

# 定义一个全局字典，用于存储训练配置的工厂类
RECIPE_FACTORY = {}

def TrainingRecipeFactory(training_recipe):
    """
    根据传入的训练配置名称，返回对应的工厂类实例。

    :param training_recipe: 训练配置的名称
    :return: 对应的工厂类实例
    :raises: AssertionError 如果传入的训练配置名称未注册
    """
    recipe = None
    # 遍历全局字典中的键，查找匹配的训练食谱名称
    for name in RECIPE_FACTORY.keys():
        if name.lower() == training_recipe.lower():
            recipe = RECIPE_FACTORY[name]
    # 如果没有找到匹配的训练配置，则抛出断言错误
    assert recipe, f"{training_recipe} is not registered"
    return recipe


def register_training_recipe(name):
    """
    注册一个新的训练配置工厂类。

    :param name: 训练配置的名称
    :return: 一个装饰器，用于将类注册为训练食谱工厂类
    """
    def register_training_recipe_cls(cls):
        # 如果该名称已存在于全局字典中，则返回已存在的工厂类
        if name in RECIPE_FACTORY:
            return RECIPE_FACTORY[name]
        # 否则，将新的工厂类添加到全局字典中，并返回该类
        RECIPE_FACTORY[name] = cls
        return cls
    return register_training_recipe_cls

# 获取当前文件所在的目录路径
models_dir = os.path.dirname(__file__)
# 导入指定目录下的所有模块
import_modules(models_dir, "tinyllava.training_recipe")
