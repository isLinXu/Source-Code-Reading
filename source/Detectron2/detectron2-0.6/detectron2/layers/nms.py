# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
                                 # 向后兼容性导入


def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    与torchvision.ops.boxes.batched_nms相同，但更安全。
    """
    assert boxes.shape[-1] == 4  # 确保边界框是4维的（x1, y1, x2, y2）
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    # TODO 可能需要更好的策略。
    # 在拥有完全CUDA的NMS操作后进行调查。
    if len(boxes) < 40000:
        # fp16 does not have enough range for batched NMS
        # fp16精度不足以支持批量NMS
        return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)  # 将boxes转换为float以确保精度

    # 当边界框数量过多时使用循环处理，避免内存问题
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)  # 创建结果掩码
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):  # 遍历所有唯一的类别索引
        mask = (idxs == id).nonzero().view(-1)  # 找出当前类别的所有边界框索引
        keep = nms(boxes[mask], scores[mask], iou_threshold)  # 对当前类别应用NMS
        result_mask[mask[keep]] = True  # 标记保留的框
    keep = result_mask.nonzero().view(-1)  # 获取所有保留的框的索引
    keep = keep[scores[keep].argsort(descending=True)]  # 按分数降序排序
    return keep


# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
# 注意：此函数(nms_rotated)将来可能会移至torchvision/ops/boxes.py
def nms_rotated(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).
    根据旋转框的交并比(IoU)对其执行非极大值抑制(NMS)。

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.
    旋转NMS迭代地移除与另一个（得分更高的）旋转框有IoU大于iou_threshold的低分旋转框。

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.
    注意，旋转框(5, 3, 4, 2, -90)覆盖的区域与旋转框(5, 3, 4, 2, 90)完全相同，
    它们的IoU将为1。但是，在某些任务（如OCR）中，它们可能代表完全不同的对象。

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.
    至于旋转NMS是否应该将它们视为远距离框（尽管它们的IOU为1），这取决于应用和/或真值标注。

    As an extreme example, consider a single character v and the square box around it.
    作为一个极端的例子，考虑一个字符v和围绕它的方形框。

    If the angle is 0 degree, the object (text) would be read as 'v';
    如果角度为0度，对象（文本）将被读作'v'；

    If the angle is 90 degrees, the object (text) would become '>';
    如果角度为90度，对象（文本）将变成'>'；

    If the angle is 180 degrees, the object (text) would become '^';
    如果角度为180度，对象（文本）将变成'^'；

    If the angle is 270/-90 degrees, the object (text) would become '<'
    如果角度为270/-90度，对象（文本）将变成'<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.
    所有这些情况彼此之间的IoU都为1，而仅使用IoU作为标准的旋转NMS将只保留其中得分最高的一个 -
    这在实际中在大多数情况下仍然有意义，因为通常只有一个方向是正确的。
    如果框仅用于对对象进行分类（而不是后续使用序列OCR识别模型进行转录），则这也不那么重要。

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)
    另一方面，当我们在训练期间使用IoU筛选接近真值的候选框时，如果我们知道真值是以严格正确的方向标记的
    （例如，倒置单词用-180度标注，尽管它们可以用0/90/-90度的框覆盖等），我们应该考虑角度因素。

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)
    原始数据集的标注方式也很重要。例如，如果数据集是一个不强制顶点排序/方向的4点多边形数据集，
    我们可以估计该多边形的最小旋转边界框，但我们无法100%确定正确的角度（如上所示，可能有4个不同的
    旋转框，彼此间的角度相差90度，覆盖完全相同的区域）。在这种情况下，我们必须仅使用IoU来确定框
    的接近度（就像许多检测基准（甚至是文本）所做的那样），除非我们可以做出其他假设
    （如宽度总是大于高度，或对象的旋转不超过90度CCW/CW等）。

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.
    总之，目前在旋转NMS中不考虑角度似乎是一个不错的选择，但我们应该意识到其含义。

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        # boxes (Tensor[N, 5]): 要执行NMS的旋转框。它们应为
        # (x中心, y中心, 宽度, 高度, 角度_度数)格式。
        scores (Tensor[N]): Scores for each one of the rotated boxes
        # scores (Tensor[N]): 每个旋转框的分数
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold
        # iou_threshold (float): 丢弃所有与IoU < iou_threshold的重叠旋转框

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    # 返回：
    #   keep (Tensor): int64张量，包含被旋转NMS保留的元素索引，按分数降序排序
    """
    return torch.ops.detectron2.nms_rotated(boxes, scores, iou_threshold)  # 调用CUDA实现的旋转NMS操作


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
# 注意：此函数(batched_nms_rotated)将来可能会移至torchvision/ops/boxes.py
def batched_nms_rotated(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    以批处理方式执行非极大值抑制。

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    每个索引值对应一个类别，NMS不会应用于不同类别的元素之间。

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        # boxes (Tensor[N, 5]):
        #   将执行NMS的框。它们应为(x中心, y中心, 宽度, 高度, 角度_度数)格式
        scores (Tensor[N]):
           scores for each one of the boxes
        # scores (Tensor[N]):
        #   每个框的分数
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        # idxs (Tensor[N]):
        #   每个框的类别索引。
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold
        # iou_threshold (float):
        #   丢弃所有IoU < iou_threshold的重叠框

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    # 返回：
    #   Tensor: int64张量，包含被NMS保留的元素索引，按分数降序排序
    """
    assert boxes.shape[-1] == 5  # 确保边界框是5维的（中心x, 中心y, 宽, 高, 角度）

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)  # 如果没有框，返回空张量
    boxes = boxes.float()  # fp16 does not have enough range for batched NMS
                           # fp16精度不足以支持批量NMS
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    # 策略：为了对每个类别独立执行NMS，
    # 我们对所有框添加偏移量。偏移量仅依赖于类别索引，
    # 并且足够大，使不同类别的框不会重叠

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    # 注意，torchvision/ops/boxes.py中的batched_nms只使用max_coordinate，
    # 无法正确处理负坐标。
    # 在这里，通过使用min_coordinate，我们可以确保正确处理负坐标。
    
    # 计算框的最大坐标（中心+尺寸/2的最大值）
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    # 计算框的最小坐标（中心-尺寸/2的最小值）
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    # 计算每个类别的偏移量
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
                                    # 避免修改boxes中的原始值
    boxes_for_nms[:, :2] += offsets[:, None]  # 将偏移量添加到框的中心坐标
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)  # 应用旋转NMS
    return keep  # 返回保留的框索引
