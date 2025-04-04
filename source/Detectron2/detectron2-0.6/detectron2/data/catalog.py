# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import types
from collections import UserDict
from typing import List

from detectron2.utils.logger import log_first_n  # 导入日志记录工具函数

__all__ = ["DatasetCatalog", "MetadataCatalog", "Metadata"]  # 定义模块的公开接口


class _DatasetCatalog(UserDict):
    """
    A global dictionary that stores information about the datasets and how to obtain them.
    一个全局字典，用于存储数据集的信息以及如何获取这些数据集。

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.
    它包含从字符串（用于标识数据集的名称，例如"coco_2014_train"）
    到解析数据集并返回样本的函数的映射，返回格式为`list[dict]`。

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.
    如果要与data/build.py和data/detection_transform.py中的数据加载功能一起使用，
    返回的字典应该符合Detectron2数据集格式（详见DATASETS.md）。

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    设计这个目录的目的是为了通过在配置中使用字符串来方便地选择不同的数据集。
    """

    def register(self, name, func):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
                       用于标识数据集的名称，例如"coco_2014_train"。
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
                一个不接受参数并返回字典列表的可调用对象。
                多次调用时必须返回相同的结果。
        """
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"  # 确保func是可调用的
        assert name not in self, "Dataset '{}' is already registered!".format(name)  # 确保数据集名称未被注册
        self[name] = func  # 将数据集名称和对应的处理函数存储在字典中

    def get(self, name):
        """
        Call the registered function and return its results.
        调用已注册的函数并返回其结果。

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
                       用于标识数据集的名称，例如"coco_2014_train"。

        Returns:
            list[dict]: dataset annotations.
                       数据集注释信息。
        """
        try:
            f = self[name]  # 获取注册的数据集处理函数
        except KeyError as e:
            raise KeyError(  # 如果数据集未注册，抛出异常并列出所有可用的数据集
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name, ", ".join(list(self.keys()))
                )
            ) from e
        return f()  # 调用函数并返回数据集

    def list(self) -> List[str]:
        """
        List all registered datasets.
        列出所有已注册的数据集。

        Returns:
            list[str]: 返回已注册数据集名称的列表
        """
        return list(self.keys())  # 返回所有已注册数据集的名称列表

    def remove(self, name):
        """
        Alias of ``pop``.
        pop方法的别名，用于移除指定的数据集。
        """
        self.pop(name)  # 从字典中移除指定的数据集

    def __str__(self):
        return "DatasetCatalog(registered datasets: {})".format(", ".join(self.keys()))  # 返回数据集目录的字符串表示

    __repr__ = __str__  # 设置repr方法与str方法相同


DatasetCatalog = _DatasetCatalog()  # 创建全局的数据集目录实例
DatasetCatalog.__doc__ = (
    _DatasetCatalog.__doc__  # 继承原始文档字符串
    + """
    .. automethod:: detectron2.data.catalog.DatasetCatalog.register
    .. automethod:: detectron2.data.catalog.DatasetCatalog.get
"""  # 添加自动生成的方法文档
)


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    一个支持简单属性设置和获取的类。
    It is intended for storing metadata of a dataset and make it accessible globally.
    用于存储数据集的元数据并使其可以全局访问。

    Examples:
    示例：
    ::
        # somewhere when you load the data:
        # 在加载数据时的某个地方：
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        # 在打印统计信息或可视化时的某个地方：
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger getattr again
    # 数据集的名称
    # 默认设置为N/A，这样在错误信息中的`self.name`不会再次触发getattr
    name: str = "N/A"

    _RENAMED = {  # 重命名的属性映射字典
        "class_names": "thing_classes",  # 类名映射到物体类别
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",  # 数据集ID到连续ID的映射
        "stuff_class_names": "stuff_classes",  # 背景类别名称
    }

    def __getattr__(self, key):
        if key in self._RENAMED:  # 如果属性名在重命名映射中
            log_first_n(  # 记录警告日志，提示属性已被重命名
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,  # 最多记录10次
            )
            return getattr(self, self._RENAMED[key])  # 返回重命名后的属性值

        # "name" exists in every metadata
        # "name"存在于每个元数据中
        if len(self.__dict__) > 1:  # 如果元数据不为空（除了name属性外还有其他属性）
            raise AttributeError(  # 抛出属性错误，并列出所有可用的属性
                "Attribute '{}' does not exist in the metadata of dataset '{}'. Available "
                "keys are {}.".format(key, self.name, str(self.__dict__.keys()))
            )
        else:  # 如果元数据为空（只有name属性）
            raise AttributeError(  # 抛出属性错误，提示元数据为空
                f"Attribute '{key}' does not exist in the metadata of dataset '{self.name}': "
                "metadata is empty."
            )

    def __setattr__(self, key, val):
        if key in self._RENAMED:  # 如果要设置的属性名在重命名映射中
            log_first_n(  # 记录警告日志，提示属性已被重命名
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,  # 最多记录10次
            )
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val) # 使用重命名后的属性名设置值

    def as_dict(self):
        """
        Returns all the metadata as a dict.
        将所有元数据以字典形式返回。
        Note that modifications to the returned dict will not reflect on the Metadata object.
        注意：对返回的字典进行修改不会影响原始的Metadata对象。
        """
        return copy.copy(self.__dict__)  # 返回元数据字典的副本

    def set(self, **kwargs):
        """
        Set multiple metadata with kwargs.
        使用关键字参数设置多个元数据。
        """
        for k, v in kwargs.items():  # 遍历关键字参数
            setattr(self, k, v)  # 设置每个元数据属性
        return self  # 返回self以支持链式调用

    def get(self, key, default=None):
        """
        Access an attribute and return its value if exists.
        访问属性并返回其值（如果存在）。
        Otherwise return default.
        否则返回默认值。
        """
        try:
            return getattr(self, key)  # 尝试获取属性值
        except AttributeError:  # 如果属性不存在
            return default  # 返回默认值


class _MetadataCatalog(UserDict):
    """
    MetadataCatalog is a global dictionary that provides access to
    :class:`Metadata` of a given dataset.
    MetadataCatalog是一个全局字典，用于提供对给定数据集的:class:`Metadata`的访问。

    The metadata associated with a certain name is a singleton: once created, the
    与特定名称关联的元数据是单例的：一旦创建，
    metadata will stay alive and will be returned by future calls to ``get(name)``.
    元数据将一直保持活跃状态，并在后续调用``get(name)``时返回。

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the execution
    of the program, e.g.: the class names in COCO.
    它类似于全局变量，所以不要滥用它。
    它用于存储在程序执行过程中保持不变且共享的知识，例如：COCO数据集中的类别名称。
    """

    def get(self, name):
        """
        Args:
            name (str): name of a dataset (e.g. coco_2014_train).
                       数据集的名称（例如：coco_2014_train）。

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
            or create an empty one if none is available.
            与此名称关联的:class:`Metadata`实例，如果不存在则创建一个空实例。
        """
        assert len(name)  # 确保数据集名称非空
        r = super().get(name, None)  # 尝试从父类字典中获取元数据
        if r is None:  # 如果元数据不存在
            r = self[name] = Metadata(name=name)  # 创建新的元数据实例并存储
        return r  # 返回元数据实例

    def list(self):
        """
        List all registered metadata.
        列出所有已注册的元数据。

        Returns:
            list[str]: keys (names of datasets) of all registered metadata
                      所有已注册元数据的键（数据集名称）
        """
        return list(self.keys())  # 返回所有已注册数据集名称的列表

    def remove(self, name):
        """
        Alias of ``pop``.
        ``pop``方法的别名，用于移除指定的元数据。
        """
        self.pop(name)  # 从字典中移除指定名称的元数据

    def __str__(self):
        return "MetadataCatalog(registered metadata: {})".format(", ".join(self.keys()))  # 返回元数据目录的字符串表示，包含所有已注册的数据集名称

    __repr__ = __str__  # 设置repr方法与str方法相同


MetadataCatalog = _MetadataCatalog()
MetadataCatalog.__doc__ = (
    _MetadataCatalog.__doc__
    + """
    .. automethod:: detectron2.data.catalog.MetadataCatalog.get
"""
)
