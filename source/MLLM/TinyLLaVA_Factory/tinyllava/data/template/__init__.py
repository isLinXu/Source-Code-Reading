import os
from typing import Dict

from .base import *
from ...utils import import_modules

# 模板工厂字典，用于存储不同版本的模板实例
TEMPlATE_FACTORY: Dict[str, Template] = {}

def TemplateFactory(version):
    """
    根据版本号获取对应的模板实例。

    参数:
        version (str): 模板的版本号。

    返回:
        Template: 对应版本的模板实例。

    抛出:
        AssertionError: 如果请求的版本号没有对应的模板实现。
    """
    template = TEMPlATE_FACTORY.get(version, None)
    assert template, f"{version} is not implmentation"
    return template


def register_template(name):
    """
    注册模板类的装饰器。

    参数:
        name (str): 模板的名称。

    返回:
        function: 装饰器函数，用于注册模板类。
    """
    def register_template_cls(cls):
        if name in TEMPlATE_FACTORY:
            return TEMPlATE_FACTORY[name]

        TEMPlATE_FACTORY[name] = cls
        return cls

    return register_template_cls


# automatically import any Python files in the models/ directory
# 自动导入models目录下的所有Python文件
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.data.template")
