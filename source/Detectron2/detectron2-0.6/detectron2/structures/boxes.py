# Copyright (c) Facebook, Inc. and its affiliates.
import math
import numpy as np
from enum import IntEnum, unique
from typing import List, Tuple, Union
import torch
from torch import device

# 定义原始框类型，支持多种数据格式
_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    边界框表示方式的枚举类
    """

    XYXY_ABS = 0 # 绝对坐标的(x0,y0,x1,y1)格式
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1 # 绝对坐标的(x0,y0,width,height)格式
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2 # 相对坐标的(x0,y0,x1,y1)格式（暂不支持）
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3 # 相对坐标的(x0,y0,width,height)格式（暂不支持）
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4 # 绝对坐标的旋转框(x_center,y_center,width,height,angle)格式
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """



    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        边界框格式转换方法
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
            box: 输入边界框，可以是列表、元组、numpy数组或tensor
            from_mode: 原始格式
            to_mode: 目标格式
        """
        if from_mode == to_mode:
            return box  # 相同格式直接返回

        original_type = type(box)  # 保存原始数据类型
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))  # 判断是否为单个框
        
        # 转换为tensor处理
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]  # 单个框转换为二维tensor
        else:
            arr = box.clone() if isinstance(box, torch.Tensor) else torch.from_numpy(np.asarray(box)).clone()

        # 检查暂不支持的相对坐标模式
        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL, BoxMode.XYWH_REL], "Relative mode not yet supported!"

        # 处理旋转框转换（XYWHA_ABS -> XYXY_ABS）
        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()  # 提高计算精度

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]

            c = torch.abs(torch.cos(a * math.pi / 180.0))  # 角度转弧度计算cos
            s = torch.abs(torch.sin(a * math.pi / 180.0))  # 角度转弧度计算sin
            
            # 计算旋转后的外接矩形宽高
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w
            
            # 转换为XYXY格式
            arr[:, 0] -= new_w / 2.0  # 中心点x转左上角x
            arr[:, 1] -= new_h / 2.0  # 中心点y转左上角y
            arr[:, 2] = arr[:, 0] + new_w  # 计算右下角x
            arr[:, 3] = arr[:, 1] + new_h  # 计算右下角y
            arr = arr[:, :4].to(dtype=original_dtype)  # 保留前4列并恢复原始数据类型

        # 处理XYWH_ABS转XYWHA_ABS（添加0角度）
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            arr[:, 0] += arr[:, 2] / 2.0  # x0转中心x
            arr[:, 1] += arr[:, 3] / 2.0  # y0转中心y
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)  # 创建0角度列
            arr = torch.cat((arr, angles), axis=1)  # 拼接角度信息

        # 处理普通矩形框转换
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]  # width转x1
                arr[:, 3] += arr[:, 1]  # height转y1
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]  # x1转width
                arr[:, 3] -= arr[:, 1]  # y1转height
            else:
                raise NotImplementedError("不支持的转换类型")

        # 恢复原始数据格式
        if single_box:
            return original_type(arr.flatten().tolist())  # 单个框转回原始类型
        return arr.numpy() if is_numpy else arr


class Boxes:
    """
    边界框操作类，使用Nx4的torch.Tensor存储边界框
    Attributes:
        tensor (torch.Tensor): Nx4矩阵，每行表示(x1,y1,x2,y2)
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """深拷贝Boxes对象"""
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """计算每个框的面积"""
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """将框坐标限制在图像尺寸内"""
        assert torch.isfinite(self.tensor).all(), "包含非法值（无限或NaN）"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """检测非空框（宽高大于阈值）"""
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """索引访问"""
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "索引结果必须保持二维"
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]  # 框的数量

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """判断框是否在指定边界内（可设置边界阈值）"""
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """获取框中心坐标"""
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """缩放框坐标"""
        self.tensor[:, 0::2] *= scale_x  # 处理x坐标（0和2列）
        self.tensor[:, 1::2] *= scale_y  # 处理y坐标（1和3列）

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """拼接多个Boxes对象"""
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """迭代返回单个框"""
        yield from self.tensor


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    计算两组边界框的两两相交面积，返回NxM矩阵
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor  # 获取底层张量数据
    # 计算相交区域的宽高：[N,M,2]
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )
    width_height.clamp_(min=0)  # 将负值（无交集的情况）置为0
    intersection = width_height.prod(dim=2)  # 计算面积：宽*高
    return intersection


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    计算两组边界框的两两交并比，返回NxM矩阵
    """
    area1 = boxes1.area()  # [N] 计算第一组框的面积
    area2 = boxes2.area()  # [M] 计算第二组框的面积
    inter = pairwise_intersection(boxes1, boxes2)  # 获取交集面积

    # 处理空框情况，避免除以零
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),  # IoU公式：交集/(A+B-交集)
        torch.zeros(1, dtype=inter.dtype, device=inter.device),  # 无交集时设为0
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 area).
    计算交集面积占第二个框面积的比例，返回NxM矩阵
    """
    area2 = boxes2.area()  # [M] 仅计算第二组框的面积
    inter = pairwise_intersection(boxes1, boxes2)  # 获取交集面积

    ioa = torch.where(
        inter > 0, 
        inter / area2,  # IoA公式：交集/box2面积
        torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


def pairwise_point_box_distance(points: torch.Tensor, boxes: Boxes) -> torch.Tensor:
    """
    Pairwise distance between N points and M boxes. The distance between a
    point and a box is represented by the distance from the point to 4 edges
    of the box. Distances are all positive when the point is inside the box.

    Args:
        points: Nx2 coordinates. Each row is (x, y)
        boxes: M boxes

    Returns:
        Tensor: distances of size (N, M, 4). The 4 values are distances from
            the point to the left, top, right, bottom of the box.
    计算点与边界框四边的符号距离，正值表示在框内，负值表示在框外
    """
    x, y = points.unsqueeze(dim=2).unbind(dim=1)  # 分解坐标到(N,1)维度
    x0, y0, x1, y1 = boxes.tensor.unsqueeze(dim=0).unbind(dim=2)  # 分解框坐标到(1,M)维度
    return torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # 计算四边距离


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes that have the same number of boxes.
    Similar to :func:`pairwise_iou`, but computes only diagonal elements of the matrix.

    Args:
        boxes1 (Boxes): bounding boxes, sized [N,4].
        boxes2 (Boxes): same length as boxes1
    Returns:
        Tensor: iou, sized [N].
    计算一一对应的框对IoU，比普通IoU计算更高效
    """
    assert len(boxes1) == len(boxes2), "输入框数量必须一致"
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    
    # 计算相交区域的左上和右下坐标
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    
    wh = (rb - lt).clamp(min=0)  # 计算宽高并处理无交集情况
    inter = wh[:, 0] * wh[:, 1]  # 计算相交面积
    iou = inter / (area1 + area2 - inter)  # 计算IoU
    return iou
