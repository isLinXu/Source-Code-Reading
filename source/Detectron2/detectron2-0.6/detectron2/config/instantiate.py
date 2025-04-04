# Copyright (c) Facebook, Inc. and its affiliates.
import dataclasses  # 导入数据类模块
import logging  # 导入日志模块
from collections import abc  # 导入抽象基类容器
from typing import Any  # 导入类型提示

from detectron2.utils.registry import _convert_target_to_string, locate  # 从注册表导入工具函数

__all__ = ["dump_dataclass", "instantiate"]  # 定义模块公开接口


def dump_dataclass(obj: Any):
    """
    Dump a dataclass recursively into a dict that can be later instantiated.
    将数据类递归转储为可后续实例化的字典。

    Args:
        obj: a dataclass object
        # obj: 数据类对象

    Returns:
        dict
    """
    assert dataclasses.is_dataclass(obj) and not isinstance(
        obj, type
    ), "dump_dataclass() requires an instance of a dataclass."  # 验证输入是数据类实例
    ret = {"_target_": _convert_target_to_string(type(obj))}  # 构建包含目标类信息的字典
    for f in dataclasses.fields(obj):  # 遍历数据类字段
        v = getattr(obj, f.name)  # 获取字段值
        if dataclasses.is_dataclass(v):  # 如果值是数据类实例
            v = dump_dataclass(v)  # 递归处理嵌套数据类
        if isinstance(v, (list, tuple)):  # 如果值是列表或元组
            v = [dump_dataclass(x) if dataclasses.is_dataclass(x) else x for x in v]  # 递归处理列表中的元素
        ret[f.name] = v  # 将处理后的值存入结果字典
    return ret


def instantiate(cfg):
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.
    根据"_target_"和参数递归实例化字典中定义的对象。

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments
        # cfg: 包含"_target_"键的类字典对象，定义调用对象和其他参数

    Returns:
        object instantiated by cfg
    """
    from omegaconf import ListConfig  # 导入OmegaConf的列表配置类型

    if isinstance(cfg, ListConfig):  # 处理列表配置类型
        lst = [instantiate(x) for x in cfg]  # 递归实例化列表元素
        return ListConfig(lst, flags={"allow_objects": True})  # 返回新的ListConfig对象
    if isinstance(cfg, list):  # 处理普通列表
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        # 专门处理列表，因为许多类将对象列表作为参数，例如ResNet、DatasetMapper
        return [instantiate(x) for x in cfg]  # 递归实例化列表元素

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:  # 处理包含_target_的映射类型
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        # 概念上等效于hydra.utils.instantiate(cfg)并设置_convert_=all，但速度更快
        cfg = {k: instantiate(v) for k, v in cfg.items()}  # 递归实例化所有值
        cls = cfg.pop("_target_")  # 弹出目标类信息
        cls = instantiate(cls)  # 实例化目标类（支持嵌套目标）

        if isinstance(cls, str):  # 如果目标是字符串形式
            cls_name = cls
            cls = locate(cls_name)  # 通过字符串定位类对象
            assert cls is not None, cls_name  # 验证类存在
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__  # 获取完整类名
            except Exception:
                # target could be anything, so the above could fail
                # 目标可能是任何类型，因此上述操作可能失败
                cls_name = str(cls)  # 使用字符串表示
        assert callable(cls), f"_target_ {cls} does not define a callable object"  # 验证可调用性
        try:
            return cls(**cfg)  # 使用配置参数实例化类
        except TypeError:
            logger = logging.getLogger(__name__)  # 获取日志记录器
            logger.error(f"Error when instantiating {cls_name}!")  # 记录实例化错误
            raise
    return cfg  # 对于无法处理的情况直接返回原值
