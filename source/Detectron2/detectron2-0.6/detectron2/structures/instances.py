# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
from typing import Any, Dict, List, Tuple, Union
import torch


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.
    
    该类表示图像中的一系列实例。它将实例的属性（例如，边界框、掩码、标签、分数）存储为"字段"。
    所有字段必须具有相同的 ``__len__``，即实例的数量。

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.
    
    该类的所有其他（非字段）属性被视为私有：它们必须以'_'开头，且用户不可修改。

    Some basic usage:
    
    一些基本用法：

    1. Set/get/check a field:
       
       设置/获取/检查字段：

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
       
       ``len(instances)`` 返回实例的数量
       
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``
       
       索引：``instances[indices]`` 将对所有字段应用索引操作，并返回一个新的 :class:`Instances`。
       通常，``indices`` 是一个整数索引向量，或一个长度为 ``num_instances`` 的二进制掩码

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
            
        参数：
            image_size (height, width)：图像的空间尺寸。
            kwargs：要添加到该 `Instances` 的字段。
        """
        self._image_size = image_size  # 存储图像尺寸（高度，宽度）
        self._fields: Dict[str, Any] = {}  # 初始化字段字典，用于存储实例属性
        for k, v in kwargs.items():
            self.set(k, v)  # 通过set方法设置传入的任何字段

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
            
        返回：
            元组：高度, 宽度
        """
        return self._image_size  # 返回图像尺寸

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)  # 对于以_开头的私有属性，直接使用父类的设置方法
        else:
            self.set(name, val)  # 对于非私有属性，使用set方法设置字段

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))  # 如果字段不存在，则抛出异常
        return self._fields[name]  # 返回请求的字段值

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        
        将名为`name`的字段设置为`value`。
        `value`的长度必须等于实例的数量，并且必须与此对象中的其他现有字段保持一致。
        """
        data_len = len(value)  # 获取要设置的数据长度
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))  # 确保新字段的长度与现有字段一致
        self._fields[name] = value  # 存储字段值

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
            
        返回：
            布尔值：名为`name`的字段是否存在。
        """
        return name in self._fields  # 检查字段是否存在

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        
        移除名为`name`的字段。
        """
        del self._fields[name]  # 删除指定字段

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        
        返回名为`name`的字段。
        """
        return self._fields[name]  # 获取指定字段

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        
        返回：
            字典：将名称（字符串）映射到字段数据的字典

        修改返回的字典将修改此实例。
        """
        return self._fields  # 返回所有字段字典

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
            
        返回：
            Instances：如果字段具有`to(device)`方法，则对所有字段调用该方法。
        """
        ret = Instances(self._image_size)  # 创建新的Instances实例
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)  # 对有to方法的字段调用to方法（例如张量移动到特定设备）
            ret.set(k, v)  # 设置新Instances实例的对应字段
        return ret  # 返回新实例

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
            
        参数：
            item：索引类对象，将用于索引所有字段。

        返回：
            如果`item`是字符串，则返回相应字段中的数据。
            否则，返回一个`Instances`，其中所有字段均由`item`索引。
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")  # 检查索引是否越界
            else:
                item = slice(item, None, len(self))  # 将整数索引转换为切片形式

        ret = Instances(self._image_size)  # 创建新的Instances实例
        for k, v in self._fields.items():
            ret.set(k, v[item])  # 对每个字段应用索引操作并设置到新实例中
        return ret  # 返回新实例

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            # 使用__len__是因为len()必须是整数，对跟踪不友好
            return v.__len__()  # 返回任一字段的长度作为实例长度
        raise NotImplementedError("Empty Instances does not support __len__!")  # 如果没有字段，则无法确定长度

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")  # Instances对象不支持迭代

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
            
        参数：
            instance_lists (list[Instances])：Instances实例的列表

        返回：
            Instances：合并后的实例
        """
        assert all(isinstance(i, Instances) for i in instance_lists)  # 确保所有元素都是Instances
        assert len(instance_lists) > 0  # 确保列表非空
        if len(instance_lists) == 1:
            return instance_lists[0]  # 如果只有一个元素，直接返回它

        image_size = instance_lists[0].image_size  # 获取第一个实例的图像尺寸
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size  # 确保所有实例的图像尺寸一致
        ret = Instances(image_size)  # 创建新的Instances实例
        for k in instance_lists[0]._fields.keys():  # 遍历第一个实例的所有字段
            values = [i.get(k) for i in instance_lists]  # 收集所有实例对应字段的值
            v0 = values[0]  # 获取第一个值，用于确定类型
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)  # 对张量使用torch.cat连接
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))  # 对列表使用chain连接
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)  # 对有cat方法的对象使用其cat方法
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))  # 不支持的类型
            ret.set(k, values)  # 设置合并后的字段值
        return ret  # 返回合并后的实例

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("  # 开始构建字符串表示
        s += "num_instances={}, ".format(len(self))  # 添加实例数量
        s += "image_height={}, ".format(self._image_size[0])  # 添加图像高度
        s += "image_width={}, ".format(self._image_size[1])  # 添加图像宽度
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))  # 添加所有字段
        return s  # 返回字符串表示

    __repr__ = __str__  # 让__repr__使用相同的实现
