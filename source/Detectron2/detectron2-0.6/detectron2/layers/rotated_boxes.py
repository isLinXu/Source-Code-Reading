# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import absolute_import, division, print_function, unicode_literals
# 导入未来特性，确保代码在Python 2和3中兼容

from detectron2 import _C  # 导入C++/CUDA实现的函数


def pairwise_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    返回框之间的交并比(Jaccard指数)。

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.
    两组框都应该采用(x中心, y中心, 宽度, 高度, 角度)格式。

    Arguments:
        boxes1 (Tensor[N, 5])
        # boxes1 (Tensor[N, 5]): N个旋转框，每个框有5个参数
        boxes2 (Tensor[M, 5])
        # boxes2 (Tensor[M, 5]): M个旋转框，每个框有5个参数

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    # 返回：
    #    iou (Tensor[N, M]): NxM矩阵，包含boxes1和boxes2中
    #    每个元素之间的成对IoU值
    """
    return _C.box_iou_rotated(boxes1, boxes2)  # 调用C++/CUDA实现的旋转框IoU计算函数
