import os

from ...utils import import_modules

# 定义一个字典用于存储连接器类的工厂
CONNECTOR_FACTORY = {}

def ConnectorFactory(connector_name):
    """
    根据连接器名称返回对应的连接器类实例。

    :param connector_name: 连接器的名称
    :return: 对应的连接器类实例
    :raises AssertionError: 如果找不到对应的连接器，则抛出异常
    """
    model = None
    # 遍历工厂字典中的所有键
    for name in CONNECTOR_FACTORY.keys():
        # 如果传入的连接器名称包含在字典键中，则赋值给model
        if name.lower() in connector_name.lower():
            model = CONNECTOR_FACTORY[name]
    # 如果model仍为None，说明没有找到对应的连接器，抛出异常
    assert model, f"{connector_name} is not registered"
    return model


def register_connector(name):
    """
    注册一个新的连接器类到工厂字典中。

    :param name: 连接器的名称
    :return: 返回一个装饰器，用于注册连接器类
    """
    def register_connector_cls(cls):
        # 如果名称已存在于工厂字典中，则返回已存在的类
        if name in CONNECTOR_FACTORY:
            return CONNECTOR_FACTORY[name]
        # 否则，将新的连接器类注册到工厂字典中
        CONNECTOR_FACTORY[name] = cls
        return cls
    return register_connector_cls


# automatically import any Python files in the models/ directory
# 自动导入models目录下的所有Python文件
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.connector")
