from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import  Dict, Union, List

# SLOT 是一个类型别名，可以是字符串、字符串列表或字符串字典
SLOT = Union[str, List[str], Dict[str, str]]

# Formatter 是一个抽象基类，定义了一个抽象方法 apply
@dataclass
class Formatter(ABC):
    # slot 是 Formatter 类的一个属性，默认值为空字符串
    slot: SLOT = ""

    # apply 方法是一个抽象方法，需要子类实现
    # **kwargs 允许方法接收任意数量的关键字参数
    # 方法返回值类型为 SLOT
    @abstractmethod
    def apply(self, **kwargs) -> SLOT: ...



@dataclass
class EmptyFormatter(Formatter):
    """
    EmptyFormatter类继承自Formatter类，它的apply方法直接返回slot属性的值。
    """
    def apply(self, **kwargs) -> SLOT:
        # 直接返回slot属性的值
        return self.slot

# StringFormatter是Formatter的另一个子类，它也重写了apply方法
@dataclass
class StringFormatter(Formatter):
    """
    StringFormatter类继承自Formatter类，并使用dataclass装饰器。
    它的apply方法用于格式化字符串，将slot中的占位符替换为kwargs中提供的值。
    """
    def apply(self, **kwargs) -> SLOT:
        # 初始化msg为空字符串
        msg = ""
        # 遍历kwargs中的键值对
        for name, value in kwargs.items():
            # 如果value为None，则返回slot的前半部分加上冒号
            if value is None:
                msg = self.slot.split(':')[0] + ":"
                return msg
            # 如果value不是字符串类型，则抛出RuntimeError异常
            if not isinstance(value, str):
                raise RuntimeError("Expected a string, got {}".format(value))
            # 将slot中的占位符替换为value
            msg = self.slot.replace("{{" + name + "}}", value, 1)
        # 返回格式化后的字符串
        return msg
