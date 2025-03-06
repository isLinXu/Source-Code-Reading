# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import abc  # 从collections导入abc模块，用于抽象基类
from itertools import repeat  # 从itertools导入repeat，用于重复元素
from numbers import Number  # 从numbers导入Number，用于数字类型检查
from typing import List  # 从typing导入List，用于类型注解

import numpy as np  # 导入numpy库，通常用于数组和矩阵操作

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh  # 从ops模块导入各种坐标转换函数


def _ntuple(n):
    """From PyTorch internals.
    从PyTorch内部实现的函数。"""

    def parse(x):
        """Parse bounding boxes format between XYWH and LTWH.
        解析XYWH和LTWH之间的边界框格式。"""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))  # 如果x是可迭代的，直接返回；否则返回重复n次的元组

    return parse  # 返回解析函数


to_2tuple = _ntuple(2)  # 创建一个将输入转换为2元组的函数
to_4tuple = _ntuple(4)  # 创建一个将输入转换为4元组的函数

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]  # 定义支持的边界框格式

__all__ = ("Bboxes", "Instances")  # 公开的类名称元组


class Bboxes:
    """
    A class for handling bounding boxes.
    处理边界框的类。

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    该类支持多种边界框格式，如'xyxy'、'xywh'和'ltwh'。
    Bounding box data should be provided in numpy arrays.
    边界框数据应以numpy数组的形式提供。

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        bboxes (numpy.ndarray): 存储在2D numpy数组中的边界框。
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').
        format (str): 边界框的格式（'xyxy'、'xywh'或'ltwh'）。

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
        此类不处理边界框的归一化或反归一化。
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """Initializes the Bboxes class with bounding box data in a specified format.
        使用指定格式的边界框数据初始化Bboxes类。"""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"  # 确保格式有效
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes  # 如果是1维数组，增加一个维度
        assert bboxes.ndim == 2  # 确保是2维数组
        assert bboxes.shape[1] == 4  # 确保每个边界框有4个坐标
        self.bboxes = bboxes  # 存储边界框
        self.format = format  # 存储格式
        # self.normalized = normalized  # 如果有归一化标志

    def convert(self, format):
        """Converts bounding box format from one type to another.
        将边界框格式从一种类型转换为另一种类型。"""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"  # 确保格式有效
        if self.format == format:  # 如果当前格式与目标格式相同
            return  # 不做任何转换
        elif self.format == "xyxy":  # 当前格式为xyxy
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh  # 选择转换函数
        elif self.format == "xywh":  # 当前格式为xywh
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh  # 选择转换函数
        else:  # 当前格式为ltwh
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh  # 选择转换函数
        self.bboxes = func(self.bboxes)  # 转换边界框
        self.format = format  # 更新格式

    def areas(self):
        """Return box areas.
        返回边界框的面积。"""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"  # 如果格式为xyxy
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True

    def mul(self, scale):
        """
        Multiply bounding box coordinates by scale factor(s).
        将边界框坐标乘以缩放因子。

        Args:
            scale (int | tuple | list): Scale factor(s) for four coordinates.
                If int, the same scale is applied to all coordinates.
            scale (int | tuple | list): 四个坐标的缩放因子。如果是int，则对所有坐标应用相同的缩放。
        """
        if isinstance(scale, Number):  # 如果缩放因子是数字
            scale = to_4tuple(scale)  # 转换为4元组
        assert isinstance(scale, (tuple, list))  # 确保缩放因子是元组或列表
        assert len(scale) == 4  # 确保缩放因子有4个元素
        self.bboxes[:, 0] *= scale[0]  # 缩放左上角x坐标
        self.bboxes[:, 1] *= scale[1]  # 缩放左上角y坐标
        self.bboxes[:, 2] *= scale[2]  # 缩放右下角x坐标
        self.bboxes[:, 3] *= scale[3]  # 缩放右下角y坐标

    def add(self, offset):
        """
        Add offset to bounding box coordinates.
        为边界框坐标添加偏移量。

        Args:
            offset (int | tuple | list): Offset(s) for four coordinates.
                If int, the same offset is applied to all coordinates.
            offset (int | tuple | list): 四个坐标的偏移量。如果是int，则对所有坐标应用相同的偏移。
        """
        if isinstance(offset, Number):  # 如果偏移量是数字
            offset = to_4tuple(offset)  # 转换为4元组
        assert isinstance(offset, (tuple, list))  # 确保偏移量是元组或列表
        assert len(offset) == 4  # 确保偏移量有4个元素
        self.bboxes[:, 0] += offset[0]  # 添加左上角x坐标的偏移
        self.bboxes[:, 1] += offset[1]  # 添加左上角y坐标的偏移
        self.bboxes[:, 2] += offset[2]  # 添加右下角x坐标的偏移
        self.bboxes[:, 3] += offset[3]  # 添加右下角y坐标的偏移

    def __len__(self):
        """Return the number of boxes.
        返回边界框的数量。"""
        return len(self.bboxes)  # 返回边界框的数量

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.
        将Bboxes对象的列表连接成一个单一的Bboxes对象。

        Args:
            boxes_list (List[Bboxes]): A list of Bboxes objects to concatenate.
            boxes_list (List[Bboxes]): 要连接的Bboxes对象列表。
            axis (int, optional): The axis along which to concatenate the bounding boxes.
                                   Defaults to 0.
            axis (int, optional): 连接边界框的轴。默认为0。

        Returns:
            Bboxes: A new Bboxes object containing the concatenated bounding boxes.
            Bboxes: 一个新的Bboxes对象，包含连接后的边界框。

        Note:
            The input should be a list or tuple of Bboxes objects.
            输入应为Bboxes对象的列表或元组。
        """
        assert isinstance(boxes_list, (list, tuple))  # 确保输入是列表或元组
        if not boxes_list:  # 如果列表为空
            return cls(np.empty(0))  # 返回一个空的Bboxes对象
        assert all(isinstance(box, Bboxes) for box in boxes_list)  # 确保所有元素都是Bboxes对象

        if len(boxes_list) == 1:  # 如果列表中只有一个Bboxes对象
            return boxes_list[0]  # 直接返回该对象
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))  # 连接边界框并返回新的Bboxes对象

    def __getitem__(self, index) -> "Bboxes":
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.
        使用索引检索特定的边界框或一组边界框。

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.
            index (int, slice, or np.ndarray): 用于选择所需边界框的索引、切片或布尔数组。

        Returns:
            Bboxes: A new Bboxes object containing the selected bounding boxes.
            Bboxes: 一个新的Bboxes对象，包含所选的边界框。

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.
            AssertionError: 如果索引的边界框未形成二维矩阵。

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
            注意：使用布尔索引时，请确保提供的布尔数组长度与边界框数量相同。
        """
        if isinstance(index, int):  # 如果索引是整数
            return Bboxes(self.bboxes[index].reshape(1, -1))  # 返回单个边界框
        b = self.bboxes[index]  # 获取索引的边界框
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"  # 确保返回的是二维矩阵
        return Bboxes(b)  # 返回新的Bboxes对象


class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.
    存储图像中检测到的对象的边界框、分段和关键点的容器。

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        _bboxes (Bboxes): 处理边界框操作的内部对象。
        keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3]. Default is None.
        keypoints (ndarray): 关键点（x, y, 可见性），形状为[N, 17, 3]。默认为None。
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        normalized (bool): 指示边界框坐标是否已归一化的标志。
        segments (ndarray): Segments array with shape [N, 1000, 2] after resampling.
        segments (ndarray): 经过重采样后的段数组，形状为[N, 1000, 2]。

    Args:
        bboxes (ndarray): An array of bounding boxes with shape [N, 4].
        bboxes (ndarray): 形状为[N, 4]的边界框数组。
        segments (list | ndarray, optional): A list or array of object segments. Default is None.
        segments (list | ndarray, optional): 对象分段的列表或数组。默认为None。
        keypoints (ndarray, optional): An array of keypoints with shape [N, 17, 3]. Default is None.
        keypoints (ndarray, optional): 形状为[N, 17, 3]的关键点数组。默认为None。
        bbox_format (str, optional): The format of bounding boxes ('xywh' or 'xyxy'). Default is 'xywh'.
        bbox_format (str, optional): 边界框的格式（'xywh'或'xyxy'）。默认为'xywh'。
        normalized (bool, optional): Whether the bounding box coordinates are normalized. Default is True.
        normalized (bool, optional): 边界框坐标是否已归一化。默认为True。

    Examples:
        ```python
        # Create an Instances object
        instances = Instances(
            bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
            segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
            keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
        )
        ```

    Note:
        The bounding box format is either 'xywh' or 'xyxy', and is determined by the `bbox_format` argument.
        边界框格式为'xywh'或'xyxy'，由`bbox_format`参数确定。
        This class does not perform input validation, and it assumes the inputs are well-formed.
        此类不执行输入验证，假定输入格式正确。
    """

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format="xywh", normalized=True) -> None:
        """
        Initialize the object with bounding boxes, segments, and keypoints.
        使用边界框、分段和关键点初始化对象。

        Args:
            bboxes (np.ndarray): Bounding boxes, shape [N, 4].
            bboxes (np.ndarray): 边界框，形状为[N, 4]。
            segments (list | np.ndarray, optional): Segmentation masks. Defaults to None.
            segments (list | np.ndarray, optional): 分割掩码。默认为None。
            keypoints (np.ndarray, optional): Keypoints, shape [N, 17, 3] and format (x, y, visible). Defaults to None.
            keypoints (np.ndarray, optional): 关键点，形状为[N, 17, 3]，格式为(x, y, 可见性)。默认为None。
            bbox_format (str, optional): Format of bboxes. Defaults to "xywh".
            bbox_format (str, optional): 边界框的格式。默认为"xywh"。
            normalized (bool, optional): Whether the coordinates are normalized. Defaults to True.
            normalized (bool, optional): 坐标是否已归一化。默认为True。
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)  # 初始化边界框
        self.keypoints = keypoints  # 初始化关键点
        self.normalized = normalized  # 初始化归一化标志
        self.segments = segments  # 初始化分段

    def convert_bbox(self, format):
        """Convert bounding box format.
        转换边界框格式。"""
        self._bboxes.convert(format=format)  # 调用Bboxes类的convert方法进行格式转换

    @property
    def bbox_areas(self):
        """Calculate the area of bounding boxes.
        计算边界框的面积。"""
        return self._bboxes.areas()  # 返回边界框的面积

    def scale(self, scale_w, scale_h, bbox_only=False):
        """Similar to denormalize func but without normalized sign.
        类似于反归一化函数，但没有归一化标志。"""
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))  # 缩放边界框
        if bbox_only:  # 如果只缩放边界框
            return  # 直接返回
        self.segments[..., 0] *= scale_w  # 缩放分段的x坐标
        self.segments[..., 1] *= scale_h  # 缩放分段的y坐标
        if self.keypoints is not None:  # 如果存在关键点
            self.keypoints[..., 0] *= scale_w  # 缩放关键点的x坐标
            self.keypoints[..., 1] *= scale_h  # 缩放关键点的y坐标

    def denormalize(self, w, h):
        """Denormalizes boxes, segments, and keypoints from normalized coordinates.
        将边界框、分段和关键点从归一化坐标反归一化。"""
        if not self.normalized:  # 如果未归一化
            return  # 直接返回
        self._bboxes.mul(scale=(w, h, w, h))  # 反归一化边界框
        self.segments[..., 0] *= w  # 反归一化分段的x坐标
        self.segments[..., 1] *= h  # 反归一化分段的y坐标
        if self.keypoints is not None:  # 如果存在关键点
            self.keypoints[..., 0] *= w  # 反归一化关键点的x坐标
            self.keypoints[..., 1] *= h  # 反归一化关键点的y坐标
        self.normalized = False  # 设置归一化标志为False

    def normalize(self, w, h):
        """Normalize bounding boxes, segments, and keypoints to image dimensions.
        将边界框、分段和关键点归一化到图像尺寸。"""
        if self.normalized:  # 如果已归一化
            return  # 直接返回
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))  # 归一化边界框
        self.segments[..., 0] /= w  # 归一化分段的x坐标
        self.segments[..., 1] /= h  # 归一化分段的y坐标
        if self.keypoints is not None:  # 如果存在关键点
            self.keypoints[..., 0] /= w  # 归一化关键点的x坐标
            self.keypoints[..., 1] /= h  # 归一化关键点的y坐标
        self.normalized = True  # 设置归一化标志为True

    def add_padding(self, padw, padh):
        """Handle rect and mosaic situation.
        处理矩形和马赛克情况。"""
        assert not self.normalized, "you should add padding with absolute coordinates."  # 确保在绝对坐标下添加填充
        self._bboxes.add(offset=(padw, padh, padw, padh))  # 为边界框添加偏移
        self.segments[..., 0] += padw  # 为分段的x坐标添加偏移
        self.segments[..., 1] += padh  # 为分段的y坐标添加偏移
        if self.keypoints is not None:  # 如果存在关键点
            self.keypoints[..., 0] += padw  # 为关键点的x坐标添加偏移
            self.keypoints[..., 1] += padh  # 为关键点的y坐标添加偏移

    def __getitem__(self, index) -> "Instances":
        """
        Retrieve a specific instance or a set of instances using indexing.
        使用索引检索特定实例或一组实例。

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.
            index (int, slice, or np.ndarray): 用于选择所需实例的索引、切片或布尔数组。

        Returns:
            Instances: A new Instances object containing the selected bounding boxes,
                       segments, and keypoints if present.
            Instances: 一个新的Instances对象，包含所选的边界框、分段和关键点（如果存在）。

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
            注意：使用布尔索引时，请确保提供的布尔数组长度与实例数量相同。
        """
        segments = self.segments[index] if len(self.segments) else self.segments  # 获取索引的分段
        keypoints = self.keypoints[index] if self.keypoints is not None else None  # 获取索引的关键点
        bboxes = self.bboxes[index]  # 获取索引的边界框
        bbox_format = self._bboxes.format  # 获取边界框格式
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )  # 返回新的Instances对象

    def flipud(self, h):
        """Flips the coordinates of bounding boxes, segments, and keypoints vertically.
        垂直翻转边界框、分段和关键点的坐标。"""
        if self._bboxes.format == "xyxy":  # 如果边界框格式为xyxy
            y1 = self.bboxes[:, 1].copy()  # 复制左上角y坐标
            y2 = self.bboxes[:, 3].copy()  # 复制右下角y坐标
            self.bboxes[:, 1] = h - y2  # 更新左上角y坐标
            self.bboxes[:, 3] = h - y1  # 更新右下角y坐标
        else:  # 如果边界框格式为其他格式
            self.bboxes[:, 1] = h - self.bboxes[:, 1]  # 更新左上角和右下角y坐标
        self.segments[..., 1] = h - self.segments[..., 1]  # 更新分段的y坐标
        if self.keypoints is not None:  # 如果存在关键点
            self.keypoints[..., 1] = h - self.keypoints[..., 1]  # 更新关键点的y坐标

    def fliplr(self, w):
        """Reverses the order of the bounding boxes and segments horizontally.
        水平翻转边界框和分段的顺序。"""
        if self._bboxes.format == "xyxy":  # 如果边界框格式为xyxy
            x1 = self.bboxes[:, 0].copy()  # 复制左上角x坐标
            x2 = self.bboxes[:, 2].copy()  # 复制右下角x坐标
            self.bboxes[:, 0] = w - x2  # 更新左上角x坐标
            self.bboxes[:, 2] = w - x1  # 更新右下角x坐标
        else:  # 如果边界框格式为其他格式
            self.bboxes[:, 0] = w - self.bboxes[:, 0]  # 更新左上角和右下角x坐标
        self.segments[..., 0] = w - self.segments[..., 0]  # 更新分段的x坐标
        if self.keypoints is not None:  # 如果存在关键点
            self.keypoints[..., 0] = w - self.keypoints[..., 0]  # 更新关键点的x坐标

    def clip(self, w, h):
        """Clips bounding boxes, segments, and keypoints values to stay within image boundaries.
        将边界框、分段和关键点的值裁剪到图像边界内。"""
        ori_format = self._bboxes.format  # 保存原始格式
        self.convert_bbox(format="xyxy")  # 转换为xyxy格式
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)  # 裁剪x坐标
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)  # 裁剪y坐标
        if ori_format != "xyxy":  # 如果原始格式不是xyxy
            self.convert_bbox(format=ori_format)  # 转换回原始格式
        self.segments[..., 0] = self.segments[..., 0].clip(0, w)  # 裁剪分段的x坐标
        self.segments[..., 1] = self.segments[..., 1].clip(0, h)  # 裁剪分段的y坐标
        if self.keypoints is not None:  # 如果存在关键点
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)  # 裁剪关键点的x坐标
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)  # 裁剪关键点的y坐标

    def remove_zero_area_boxes(self):
        """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.
        移除零面积的边界框，即裁剪后某些边界框可能具有零宽度或高度。"""
        good = self.bbox_areas > 0  # 找到有效的边界框
        if not all(good):  # 如果存在无效的边界框
            self._bboxes = self._bboxes[good]  # 仅保留有效的边界框
            if len(self.segments):  # 如果存在分段
                self.segments = self.segments[good]  # 仅保留有效的分段
            if self.keypoints is not None:  # 如果存在关键点
                self.keypoints = self.keypoints[good]  # 仅保留有效的关键点
        return good  # 返回有效标志

    def update(self, bboxes, segments=None, keypoints=None):
        """Updates instance variables.
        更新实例变量。"""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)  # 更新边界框
        if segments is not None:  # 如果提供了分段
            self.segments = segments  # 更新分段
        if keypoints is not None:  # 如果提供了关键点
            self.keypoints = keypoints  # 更新关键点

    def __len__(self):
        """Return the length of the instance list.
        返回实例列表的长度。"""
        return len(self.bboxes)  # 返回边界框的数量

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        """
        Concatenates a list of Instances objects into a single Instances object.
        将Instances对象的列表连接成一个单一的Instances对象。

        Args:
            instances_list (List[Instances]): A list of Instances objects to concatenate.
            instances_list (List[Instances]): 要连接的Instances对象列表。
            axis (int, optional): The axis along which the arrays will be concatenated. Defaults to 0.
            axis (int, optional): 数组连接的轴。默认为0。

        Returns:
            Instances: A new Instances object containing the concatenated bounding boxes,
                       segments, and keypoints if present.
            Instances: 一个新的Instances对象，包含连接后的边界框、分段和关键点（如果存在）。

        Note:
            The `Instances` objects in the list should have the same properties, such as
            the format of the bounding boxes, whether keypoints are present, and if the
            coordinates are normalized.
            注意：列表中的`Instances`对象应具有相同的属性，例如边界框的格式、关键点是否存在以及坐标是否已归一化。
        """
        assert isinstance(instances_list, (list, tuple))  # 确保输入是列表或元组
        if not instances_list:  # 如果列表为空
            return cls(np.empty(0))  # 返回一个空的Instances对象
        assert all(isinstance(instance, Instances) for instance in instances_list)  # 确保所有元素都是Instances对象

        if len(instances_list) == 1:  # 如果列表中只有一个Instances对象
            return instances_list[0]  # 直接返回该对象

        use_keypoint = instances_list[0].keypoints is not None  # 检查是否使用关键点
        bbox_format = instances_list[0]._bboxes.format  # 获取边界框格式
        normalized = instances_list[0].normalized  # 获取归一化标志

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=0)  # 连接边界框
        seg_len = [b.segments.shape[1] for b in instances_list]  # 获取每个实例的分段长度
        if len(frozenset(seg_len)) > 1:  # 如果分段长度不同，则重采样
            max_len = max(seg_len)  # 获取最大长度
            cat_segments = np.concatenate(
                [
                    resample_segments(list(b.segments), max_len)  # 重采样分段
                    if len(b.segments)
                    else np.zeros((0, max_len, 2), dtype=np.float32)  # 重新生成空分段
                    for b in instances_list
                ],
                axis=axis,
            )
        else:
            cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)  # 连接分段
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None  # 连接关键点
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)  # 返回新的Instances对象

    @property
    def bboxes(self):
        """Return bounding boxes.
        返回边界框。"""
        return self._bboxes.bboxes  # 返回边界框