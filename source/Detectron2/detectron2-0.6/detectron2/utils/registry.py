# Copyright (c) Facebook, Inc. and its affiliates.  # 版权声明：属于Facebook, Inc.及其附属公司

from typing import Any  # 导入Any类型注解
import pydoc  # 导入pydoc模块，用于访问Python文档
from fvcore.common.registry import Registry  # for backward compatibility.  # 从fvcore.common.registry导入Registry类，用于向后兼容

"""
``Registry`` and `locate` provide ways to map a string (typically found
in config files) to callable objects.
"""  # `Registry`和`locate`提供了将字符串（通常在配置文件中找到）映射到可调用对象的方法。

__all__ = ["Registry", "locate"]  # 定义模块的公开API


def _convert_target_to_string(t: Any) -> str:  # 定义一个函数，将对象转换为字符串表示
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """  # `locate()`的逆操作。
        # 参数：
        #    t: 任何具有`__module__`和`__qualname__`属性的对象
    
    module, qualname = t.__module__, t.__qualname__  # 获取对象的模块名和限定名

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.  # 压缩到该对象的路径，例如`module.submodule._impl.class`
    # 可能变成`module.submodule.class`，如果后者也解析为同一个对象。
    # 这简化了字符串，并且受类实现移动的影响较小。
    
    module_parts = module.split(".")  # 将模块名按点分割成部分
    for k in range(1, len(module_parts)):  # 遍历模块路径的不同层级
        prefix = ".".join(module_parts[:k])  # 获取模块路径的前缀
        candidate = f"{prefix}.{qualname}"  # 构造候选字符串路径
        try:  # 尝试定位候选路径
            if locate(candidate) is t:  # 如果候选路径定位到的对象与原对象相同
                return candidate  # 返回更简洁的路径
        except ImportError:  # 如果导入失败
            pass  # 继续尝试下一个候选路径
    return f"{module}.{qualname}"  # 如果没有找到更简洁的路径，返回完整路径


def locate(name: str) -> Any:  # 定义一个函数，通过字符串路径定位并返回对象
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """  # 使用输入字符串`{x.__module__}.{x.__qualname__}`（如"module.submodule.class_name"）
        # 定位并返回对象`x`。
        # 如果找不到对象，则引发异常。
    
    obj = pydoc.locate(name)  # 使用pydoc.locate尝试定位对象

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.  # 某些情况（例如torch.optim.sgd.SGD）
    # 不能被pydoc.locate正确处理。尝试使用hydra的私有函数。
    
    if obj is None:  # 如果pydoc.locate无法定位对象
        try:  # 尝试导入hydra的定位函数
            # from hydra.utils import get_method - will print many errors  # 从hydra.utils导入get_method - 会打印很多错误
            from hydra.utils import _locate  # 从hydra.utils导入_locate函数
        except ImportError as e:  # 如果导入失败
            raise ImportError(f"Cannot dynamically locate object {name}!") from e  # 抛出导入错误，表明无法动态定位对象
        else:  # 如果导入成功
            obj = _locate(name)  # it raises if fails  # 使用_locate尝试定位对象，如果失败会抛出异常

    return obj  # 返回定位到的对象
