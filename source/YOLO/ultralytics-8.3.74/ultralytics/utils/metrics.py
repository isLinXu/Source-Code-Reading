# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)  # 定义OKS_SIGMA数组并进行归一化处理


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
        # box1是一个形状为(n, 4)的numpy数组，表示n个边界框
        box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
        # box2是一个形状为(m, 4)的numpy数组，表示m个边界框
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        # 如果为True，则计算标准IoU，否则返回inter_area/box2_area
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # 一个小值，用于避免除以零的情况，默认为1e-7

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
        # 返回一个形状为(n, m)的numpy数组，表示box2的交集面积
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T  # 解包box1的坐标
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T  # 解包box2的坐标

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)  # 计算交集面积

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # 计算box2的面积
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # 计算box1的面积
        area = area + box1_area[:, None] - inter_area  # 计算IoU的分母

    # Intersection over box2 area
    return inter_area / (area + eps)  # 返回交集面积与box2面积的比值


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        # box1是一个形状为(N, 4)的张量，表示N个边界框
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        # box2是一个形状为(M, 4)的张量，表示M个边界框
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # 一个小值，用于避免除以零的情况，默认为1e-7

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
        # 返回一个形状为(N, M)的张量，包含box1和box2中每个元素的成对IoU值
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)  # 获取box1和box2的坐标
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)  # 计算交集面积

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)  # 返回IoU值


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        # box1是一个张量，表示一个或多个边界框，最后一维为4
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        # box2是一个张量，表示一个或多个边界框，最后一维为4
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        # 如果为True，输入框为(x, y, w, h)格式；如果为False，输入框为(x1, y1, x2, y2)格式，默认为True
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        # 如果为True，计算广义IoU，默认为False
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        # 如果为True，计算距离IoU，默认为False
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        # 如果为True，计算完整IoU，默认为False
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # 一个小值，用于避免除以零的情况，默认为1e-7

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
        # 返回IoU、GIoU、DIoU或CIoU值，具体取决于指定的标志
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)  # 将边界框从xywh格式转换为xyxy格式
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2  # 计算宽高的一半
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_  # 计算边界框的左上角和右下角坐标
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_  # 计算边界框的左上角和右下角坐标
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)  # 解包box1的坐标
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)  # 解包box2的坐标
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps  # 计算box1的宽和高
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps  # 计算box2的宽和高

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)  # 计算交集面积

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps  # 计算并集面积

    # IoU
    iou = inter / union  # 计算IoU
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)  # aspect ratio
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))  # 计算alpha
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # 返回IoU


def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        # mask1是一个形状为(N, n)的张量，其中N是真实对象的数量，n是图像宽度和高度的乘积
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        # mask2是一个形状为(M, n)的张量，其中M是预测对象的数量，n是图像宽度和高度的乘积
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # 一个小值，用于避免除以零的情况，默认为1e-7

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
        # 返回一个形状为(N, M)的张量，表示掩码的IoU
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)  # 计算交集
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)  # 返回掩码的IoU


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        # kpt1是一个形状为(N, 17, 3)的张量，表示真实关键点
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        # kpt2是一个形状为(M, 17, 3)的张量，表示预测关键点
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        # area是一个形状为(N,)的张量，表示真实边界框的面积
        sigma (list): A list containing 17 values representing keypoint scales.
        # sigma是一个包含17个值的列表，表示关键点的缩放因子
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # 一个小值，用于避免除以零的情况，默认为1e-7

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
        # 返回一个形状为(N, M)的张量，表示关键点的相似度
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # 计算距离
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # 将sigma转换为张量
    kpt_mask = kpt1[..., 2] != 0  # (N, 17) 创建关键点掩码，指示关键点是否存在
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # 根据公式计算e
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)  # 返回关键点相似度


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.
        # boxes是一个形状为(N, 5)的张量，表示旋转边界框，格式为xywhr

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
        # 返回与原始旋转边界框对应的协方差矩阵
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)  # 计算高斯边界框
    a, b, c = gbbs.split(1, dim=-1)  # 分割gbbs
    cos = c.cos()  # 计算cos值
    sin = c.sin()  # 计算sin值
    cos2 = cos.pow(2)  # 计算cos的平方
    sin2 = sin.pow(2)  # 计算sin的平方
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin  # 返回协方差矩阵


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        # obb1是一个形状为(N, 5)的张量，表示真实的OBB，格式为xywhr
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        # obb2是一个形状为(N, 5)的张量，表示预测的OBB，格式为xywhr
        CIoU (bool, optional): If True, calculate CIoU. Defaults to False.
        # 如果为True，计算CIoU，默认为False
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.
        # 一个小值，用于避免除以零的情况，默认为1e-7

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).
        # 返回OBB相似度，形状为(N,)
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)  # 获取obb1的中心坐标
    x2, y2 = obb2[..., :2].split(1, dim=-1)  # 获取obb2的中心坐标
    a1, b1, c1 = _get_covariance_matrix(obb1)  # 计算obb1的协方差矩阵
    a2, b2, c2 = _get_covariance_matrix(obb2)  # 计算obb2的协方差矩阵

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25  # 计算t1
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5  # 计算t2
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5  # 计算t3
    bd = (t1 + t2 + t3).clamp(eps, 100.0)  # 计算bd
    hd = (1.0 - (-bd).exp() + eps).sqrt()  # 计算hd
    iou = 1 - hd  # 计算IoU
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)  # 获取obb1的宽高
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)  # 获取obb2的宽高
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)  # 计算宽高比
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))  # 计算alpha
        return iou - v * alpha  # CIoU
    return iou  # 返回IoU


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        # obb1是一个形状为(N, 5)的张量，表示真实的OBB，格式为xywhr
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        # obb2是一个形状为(M, 5)的张量，表示预测的OBB，格式为xywhr
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # 一个小值，用于避免除以零的情况，默认为1e-7

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
        # 返回一个形状为(N, M)的张量，表示OBB相似度
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1  # 将obb1转换为张量
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2  # 将obb2转换为张量

    x1, y1 = obb1[..., :2].split(1, dim=-1)  # 获取obb1的中心坐标
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))  # 获取obb2的中心坐标
    a1, b1, c1 = _get_covariance_matrix(obb1)  # 计算obb1的协方差矩阵
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))  # 计算obb2的协方差矩阵

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25  # 计算t1
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5  # 计算t2
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5  # 计算t3
    bd = (t1 + t2 + t3).clamp(eps, 100.0)  # 计算bd
    hd = (1.0 - (-bd).exp() + eps).sqrt()  # 计算hd
    return 1 - hd  # 返回prob IoU


def smooth_bce(eps=0.1):
    """
    Computes smoothed positive and negative Binary Cross-Entropy targets.

    This function calculates positive and negative label smoothing BCE targets based on a given epsilon value.
    For implementation details, refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.

    Args:
        eps (float, optional): The epsilon value for label smoothing. Defaults to 0.1.
        # 用于标签平滑的epsilon值，默认为0.1

    Returns:
        (tuple): A tuple containing the positive and negative label smoothing BCE targets.
        # 返回一个元组，包含平滑的正负标签BCE目标
    """
    return 1.0 - 0.5 * eps, 0.5 * eps  # 返回平滑的正负标签

class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.
    用于计算和更新目标检测和分类任务的混淆矩阵的类。

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        任务类型（字符串）：可以是“detect”或“classify”。
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        矩阵（np.ndarray）：混淆矩阵，维度取决于任务。
        nc (int): The number of classes.
        nc（整数）：类别的数量。
        conf (float): The confidence threshold for detections.
        conf（浮点数）：检测的置信度阈值。
        iou_thres (float): The Intersection over Union threshold.
        iou_thres（浮点数）：交并比阈值。
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """Initialize attributes for the YOLO model."""
        # 初始化YOLO模型的属性
        self.task = task  # 任务类型
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        # 根据任务类型初始化混淆矩阵
        self.nc = nc  # number of classes 类别数量
        self.conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
        # 如果传入的conf为None或0.001，则将conf设置为0.25
        self.iou_thres = iou_thres  # 交并比阈值

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            preds（数组[N，min(nc,5) ]）：预测的类别标签。
            targets (Array[N, 1]): Ground truth class labels.
            targets（数组[N，1]）：真实类别标签。
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        # 将预测和目标标签合并，并提取出预测的类别
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1
            # 更新混淆矩阵，增加对应类别的计数

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            detections（数组[N，6] | 数组[N，7]）：检测到的边界框及其相关信息。
                                      每一行应包含（x1，y1，x2，y2，conf，class）
                                      或在obb时包含额外的元素`angle`。
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_bboxes（数组[M，4] | 数组[N，5]）：真实边界框，格式为xyxy/xyxyr。
            gt_cls (Array[M]): The class labels.
            gt_cls（数组[M]）：类别标签。
        """
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            # 检查标签是否为空
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                # 过滤掉置信度低于阈值的检测
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
                    # 更新混淆矩阵，增加假阳性的计数
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
                # 更新混淆矩阵，增加背景的假阴性计数
            return

        detections = detections[detections[:, 4] > self.conf]
        # 过滤掉置信度低于阈值的检测
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        # 判断是否为带有额外`angle`维度的obb
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )
        # 计算交并比

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            # 获取匹配的检测
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
                # 更新混淆矩阵，增加正确匹配的计数
            else:
                self.matrix[self.nc, gc] += 1  # true background
                # 更新混淆矩阵，增加真实背景的计数

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # predicted background
                # 更新混淆矩阵，增加预测背景的计数

    def matrix(self):
        """Returns the confusion matrix."""
        # 返回混淆矩阵
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        # 返回真正例和假阳性
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # remove background class if task=detect
        # 如果任务是检测，返回去掉背景类的真正例和假阳性

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            normalize（布尔值）：是否对混淆矩阵进行归一化。
            save_dir (str): Directory where the plot will be saved.
            save_dir（字符串）：保存图表的目录。
            names (tuple): Names of classes, used as labels on the plot.
            names（元组）：类别名称，用作图表上的标签。
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
            on_plot（函数）：可选的回调函数，在图表渲染时传递图表路径和数据。
        """
        import seaborn  # scope for faster 'import ultralytics'
        # 导入seaborn库

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        # 归一化列
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        # 不进行注释（会显示为0.00）

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        # 创建图形和坐标轴
        nc, nn = self.nc, len(names)  # number of classes, names
        # 获取类别数量和名称数量
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        # 设置seaborn主题
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        # 检查是否应用名称到刻度标签
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        # 设置刻度标签
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            # 抑制空矩阵的警告
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        # 设置标题
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        # 设置保存文件名
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """Print the confusion matrix to the console."""
        # 将混淆矩阵打印到控制台
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))
            # 记录每一行的混淆矩阵信息


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    # 进行f分数的箱型滤波
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    # 计算滤波元素的数量（必须为奇数）
    p = np.ones(nf // 2)  # ones padding
    # 创建填充元素
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    # 对y进行填充
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
    # 返回平滑后的y


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names={}, on_plot=None):
    """Plots a precision-recall curve."""
    # 绘制精确度-召回率曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    # 创建图形和坐标轴
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        # 如果类别少于21，则显示每个类别的图例
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
            # 绘制每个类别的精确度-召回率曲线
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)
        # 绘制所有类别的精确度-召回率曲线

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    # 绘制所有类别的平均精确度-召回率曲线
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

@plt_settings()
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names={}, xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    # 绘制一个度量-置信度曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    # 创建图形和坐标轴，设置图形大小为9x6，并调整布局

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        # 如果类别数量少于21，则显示每个类别的图例
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
            # 绘制每个类别的置信度与度量的关系曲线
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)
        # 绘制所有类别的置信度与度量的关系曲线，颜色为灰色

    y = smooth(py.mean(0), 0.05)
    # 对每个类别的度量取平均并进行平滑处理
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    # 绘制所有类别的平滑曲线，显示最大值及其对应的置信度

    ax.set_xlabel(xlabel)  # 设置x轴标签
    ax.set_ylabel(ylabel)  # 设置y轴标签
    ax.set_xlim(0, 1)  # 设置x轴范围
    ax.set_ylim(0, 1)  # 设置y轴范围
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 设置图例位置
    ax.set_title(f"{ylabel}-Confidence Curve")  # 设置标题
    fig.savefig(save_dir, dpi=250)  # 保存图形，分辨率为250dpi
    plt.close(fig)  # 关闭图形
    if on_plot:
        on_plot(save_dir)  # 如果提供了回调函数，则调用它并传递保存路径


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # 计算给定召回率和精确率曲线的平均精度（AP）
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    # 在召回率曲线的开始和结束处添加哨兵值
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # 在精确率曲线的开始和结束处添加哨兵值

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # 计算精确率包络线

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    # 选择计算方法：'continuous'或'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        # 创建101个点的线性插值
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        # 计算曲线下的面积
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        # 找到召回率变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        # 计算曲线下的面积

    return ap, mpre, mrec  # 返回平均精度、精确率包络线和修改后的召回率曲线


def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
        ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
        unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
        p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
        r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
        f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
        x (np.ndarray): X-axis values for the curves. Shape: (1000,).
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """
    # 计算每个类别的平均精度，用于目标检测评估
    # Sort by objectness
    i = np.argsort(-conf)
    # 根据置信度对检测结果进行排序
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # 根据排序结果更新真正例、置信度和预测类别

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    # 找到唯一类别及其对应的数量
    nc = unique_classes.shape[0]  # number of classes, number of detections
    # 获取类别数量和检测数量

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []
    # 创建1000个点的线性空间用于绘制精确率-召回率曲线

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    # 初始化平均精度、精确率曲线和召回率曲线
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        # 找到当前类别的预测索引
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        # 获取当前类别的真实标签数量和预测数量
        if n_p == 0 or n_l == 0:
            continue  # 如果没有预测或标签，则跳过

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        # 计算假阳性的累积和
        tpc = tp[i].cumsum(0)
        # 计算真正例的累积和

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # 计算召回率曲线
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        # 使用插值方法计算召回率曲线

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        # 计算精确率曲线
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score
        # 使用插值方法计算精确率曲线

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            # 计算每个类别的平均精度
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5
                # 记录mAP@0.5时的精确率值

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    # 计算F1分数（精确率和召回率的调和平均数）
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    # 仅保留有数据的类别名称
    names = dict(enumerate(names))  # to dict
    # 将类别名称转换为字典
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        # 绘制精确率-召回率曲线
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        # 绘制F1曲线
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        # 绘制精确率曲线
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)
        # 绘制召回率曲线

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    # 找到最大F1分数的索引
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    # 获取最大F1分数对应的精确率和召回率
    tp = (r * nt).round()  # true positives
    # 计算真正例
    fp = (tp / (p + eps) - tp).round()  # false positives
    # 计算假阳性
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values
    # 返回所有计算结果

class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.
    用于计算YOLOv8模型的评估指标的类。

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        p（列表）：每个类别的精确率。形状：（nc，）。
        r (list): Recall for each class. Shape: (nc,).
        r（列表）：每个类别的召回率。形状：（nc，）。
        f1 (list): F1 score for each class. Shape: (nc,).
        f1（列表）：每个类别的F1分数。形状：（nc，）。
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        all_ap（列表）：所有类别和所有IoU阈值的AP分数。形状：（nc，10）。
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        ap_class_index（列表）：每个AP分数的类别索引。形状：（nc，）。
        nc (int): Number of classes.
        nc（整数）：类别数量。

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        # 初始化一个Metric实例，用于计算YOLOv8模型的评估指标
        self.p = []  # (nc, )
        # 每个类别的精确率列表
        self.r = []  # (nc, )
        # 每个类别的召回率列表
        self.f1 = []  # (nc, )
        # 每个类别的F1分数列表
        self.all_ap = []  # (nc, 10)
        # 所有类别和所有IoU阈值的AP分数列表
        self.ap_class_index = []  # (nc, )
        # 每个AP分数的类别索引列表
        self.nc = 0  # 类别数量

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []
        # 返回所有类别在IoU阈值为0.5时的平均精度（AP50），如果没有可用数据则返回空列表

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []
        # 返回所有类别在IoU阈值为0.5到0.95时的平均精度（AP），如果没有可用数据则返回空列表

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0
        # 返回所有类别的平均精确率，如果没有可用数据则返回0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0
        # 返回所有类别的平均召回率，如果没有可用数据则返回0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0
        # 返回所有类别在IoU阈值为0.5时的平均AP，如果没有可用数据则返回0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0
        # 返回所有类别在IoU阈值为0.75时的平均AP，如果没有可用数据则返回0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0
        # 返回所有类别在IoU阈值为0.5到0.95之间的平均AP，如果没有可用数据则返回0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]
        # 返回所有结果的平均值，包括平均精确率、平均召回率、mAP@0.5和mAP

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]
        # 返回第i个类别的精确率、召回率、AP50和AP值

    @property
    def maps(self):
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        # 初始化每个类别的mAP
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
            # 根据类别索引更新每个类别的mAP
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()
        # 根据加权组合计算模型的适应度

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the [results](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/utils/metrics.py:1020:4-1023:87) tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results
        # 更新评估指标，使用提供的结果元组中的值

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []
        # 返回一个空列表，表示没有特定的曲线

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            # 返回召回率与精确率的曲线
            [self.px, self.f1_curve, "Confidence", "F1"],
            # 返回置信度与F1分数的曲线
            [self.px, self.p_curve, "Confidence", "Precision"],
            # 返回置信度与精确率的曲线
            [self.px, self.r_curve, "Confidence", "Recall"],
            # 返回置信度与召回率的曲线
        ]
        # 返回召回率与精确率的曲线、置信度与F1分数的曲线、置信度与精确率的曲线、置信度与召回率的曲线

class DetMetrics(SimpleClass):
    """
    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP) of an
    object detection model.
    用于计算目标检测模型的检测指标，如精确率、召回率和平均精度（mAP）的实用类。

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        save_dir（Path）：输出图表保存的目录路径。默认为当前目录。
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        plot（布尔值）：指示是否为每个类别绘制精确率-召回率曲线的标志。默认为False。
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        on_plot（函数）：可选的回调函数，在图表渲染时传递图表路径和数据。默认为None。
        names (dict of str): A dict of strings that represents the names of the classes. Defaults to an empty tuple.
        names（字符串字典）：表示类别名称的字符串字典。默认为空元组。

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        save_dir（Path）：输出图表保存的目录路径。
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        plot（布尔值）：指示是否为每个类别绘制精确率-召回率曲线的标志。
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        on_plot（函数）：可选的回调函数，在图表渲染时传递图表路径和数据。
        names (dict of str): A dict of strings that represents the names of the classes.
        names（字符串字典）：表示类别名称的字符串字典。
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        box（Metric）：用于存储检测指标结果的Metric类实例。
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.
        speed（字典）：用于存储检测过程不同部分执行时间的字典。

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        process(tp, conf, pred_cls, target_cls)：使用最新一批预测更新指标结果。
        keys: Returns a list of keys for accessing the computed detection metrics.
        keys：返回用于访问计算的检测指标的键列表。
        mean_results: Returns a list of mean values for the computed detection metrics.
        mean_results：返回计算的检测指标的平均值列表。
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        class_result(i)：返回特定类别的计算检测指标的值列表。
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        maps：返回不同IoU阈值的平均精度（mAP）值的字典。
        fitness: Computes the fitness score based on the computed detection metrics.
        fitness：根据计算的检测指标计算适应度分数。
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        ap_class_index：返回按平均精度（AP）值排序的类别索引列表。
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
        results_dict：返回一个字典，将检测指标键映射到其计算值。
        curves: TODO
        curves_results: TODO
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names={}) -> None:
        """Initialize a DetMetrics instance with a save directory, plot flag, callback function, and class names."""
        # 使用保存目录、绘图标志、回调函数和类别名称初始化DetMetrics实例
        self.save_dir = save_dir  # 保存目录
        self.plot = plot  # 绘图标志
        self.on_plot = on_plot  # 回调函数
        self.names = names  # 类别名称
        self.box = Metric()  # 创建Metric实例用于存储检测指标结果
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # 存储不同阶段的执行时间
        self.task = "detect"  # 任务类型为检测

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        # 处理目标检测的预测结果并更新指标
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]  # 只获取结果的后面部分
        self.box.nc = len(self.names)  # 更新类别数量
        self.box.update(results)  # 更新指标结果

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        # 返回用于访问特定指标的键列表

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()
        # 计算检测对象的平均值并返回精确率、召回率、mAP50和mAP50-95

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)
        # 返回针对特定类别评估对象检测模型性能的结果

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps
        # 返回每个类别的平均精度（mAP）分数

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()
        # 返回框对象的适应度

    @property
    def ap_class_index(self):
        """Boxes and masks have the same ap_class_index."""
        return self.box.ap_class_index
        # 返回框和掩膜的相同平均精度索引

    @property
    def results_dict(self):
        """Returns results of object detection model for evaluation."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))
        # 返回用于评估的对象检测模型结果的字典

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
        ]
        # 返回用于访问特定指标曲线的曲线列表

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.seg.curves_results
        # 返回计算的性能指标和统计信息的字典


class SegmentMetrics(SimpleClass):
    """
    Calculates and aggregates detection and segmentation metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        save_dir（Path）：输出图表保存的目录路径。默认为当前目录。
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        plot（布尔值）：是否保存检测和分割图表。默认为False。
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        on_plot（函数）：可选的回调函数，在图表渲染时传递图表路径和数据。默认为None。
        names (list): List of class names. Default is an empty list.
        names（列表）：类别名称列表。默认为空列表。

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        save_dir（Path）：输出图表保存的目录路径。
        plot (bool): Whether to save the detection and segmentation plots.
        plot（布尔值）：是否保存检测和分割图表。
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        on_plot（函数）：可选的回调函数，在图表渲染时传递图表路径和数据。
        names (list): List of class names.
        names（列表）：类别名称列表。
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        box（Metric）：用于计算框检测指标的Metric类实例。
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        seg（Metric）：用于计算掩膜分割指标的Metric类实例。
        speed (dict): Dictionary to store the time taken in different phases of inference.
        speed（字典）：用于存储推理不同阶段所用时间的字典。

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        process(tp_m, tp_b, conf, pred_cls, target_cls)：处理给定预测集的指标。
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        mean_results()：返回所有类别的检测和分割指标的平均值。
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        class_result(i)：返回类别`i`的检测和分割指标。
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        maps：返回IoU阈值范围从0.50到0.95的平均精度（mAP）分数。
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        fitness：返回适应度分数，这是指标的加权组合。
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        ap_class_index：返回用于计算平均精度（AP）的类别索引列表。
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
        results_dict：返回包含所有检测和分割指标和适应度分数的字典。
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize a SegmentMetrics instance with a save directory, plot flag, callback function, and class names."""
        # 使用保存目录、绘图标志、回调函数和类别名称初始化SegmentMetrics实例
        self.save_dir = save_dir  # 保存目录
        self.plot = plot  # 绘图标志
        self.on_plot = on_plot  # 回调函数
        self.names = names  # 类别名称
        self.box = Metric()  # 创建Metric实例用于计算框检测指标
        self.seg = Metric()  # 创建Metric实例用于计算掩膜分割指标
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # 存储不同阶段的执行时间
        self.task = "segment"  # 任务类型为分割

    def process(self, tp, tp_m, conf, pred_cls, target_cls):
        """
        Processes the detection and segmentation metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp（列表）：真实正例框的列表。
            tp_m (list): List of True Positive masks.
            tp_m（列表）：真实正例掩膜的列表。
            conf (list): List of confidence scores.
            conf（列表）：置信度分数的列表。
            pred_cls (list): List of predicted classes.
            pred_cls（列表）：预测类别的列表。
            target_cls (list): List of target classes.
            target_cls（列表）：目标类别的列表。
        """
        results_mask = ap_per_class(
            tp_m,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]  # 只获取结果的后面部分
        self.seg.nc = len(self.names)  # 更新类别数量
        self.seg.update(results_mask)  # 更新掩膜指标结果
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]  # 只获取结果的后面部分
        self.box.nc = len(self.names)  # 更新类别数量
        self.box.update(results_box)  # 更新框指标结果

    @property
    def keys(self):
        """Returns a list of keys for accessing metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]
        # 返回用于访问指标的键列表

    def mean_results(self):
        """Return the mean metrics for bounding box and segmentation results."""
        return self.box.mean_results() + self.seg.mean_results()
        # 返回框和分割结果的平均指标

    def class_result(self, i):
        """Returns classification results for a specified class index."""
        return self.box.class_result(i) + self.seg.class_result(i)
        # 返回指定类别索引的分类结果

    @property
    def maps(self):
        """Returns mAP scores for object detection and semantic segmentation models."""
        return self.box.maps + self.seg.maps
        # 返回对象检测和语义分割模型的平均精度（mAP）分数

    @property
    def fitness(self):
        """Get the fitness score for both segmentation and bounding box models."""
        return self.seg.fitness() + self.box.fitness()
        # 返回分割和框模型的适应度分数

    @property
    def ap_class_index(self):
        """Boxes and masks have the same ap_class_index."""
        return self.box.ap_class_index
        # 返回框和掩膜的相同平均精度索引

    @property
    def results_dict(self):
        """Returns results of object detection model for evaluation."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))
        # 返回用于评估的对象检测模型结果的字典

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]
        # 返回用于访问特定指标曲线的曲线列表

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.seg.curves_results
        # 返回计算的性能指标和统计信息的字典

class PoseMetrics(SegmentMetrics):
    """
    Calculates and aggregates detection and pose metrics over a given set of classes.
    计算和汇总给定类别集的检测和姿态指标。

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        save_dir（Path）：输出图表保存的目录路径。默认为当前目录。
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        plot（布尔值）：是否保存检测和分割图表。默认为False。
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        on_plot（函数）：可选的回调函数，在图表渲染时传递图表路径和数据。默认为None。
        names (list): List of class names. Default is an empty list.
        names（列表）：类别名称列表。默认为空列表。

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        save_dir（Path）：输出图表保存的目录路径。
        plot (bool): Whether to save the detection and segmentation plots.
        plot（布尔值）：是否保存检测和分割图表。
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        on_plot（函数）：可选的回调函数，在图表渲染时传递图表路径和数据。
        names (list): List of class names.
        names（列表）：类别名称列表。
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        box（Metric）：用于计算框检测指标的Metric类实例。
        pose (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        pose（Metric）：用于计算掩膜分割指标的Metric类实例。
        speed (dict): Dictionary to store the time taken in different phases of inference.
        speed（字典）：用于存储推理不同阶段所用时间的字典。

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        process(tp_m, tp_b, conf, pred_cls, target_cls)：处理给定预测集的指标。
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        mean_results()：返回所有类别的检测和分割指标的平均值。
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        class_result(i)：返回类别`i`的检测和分割指标。
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        maps：返回IoU阈值范围从0.50到0.95的平均精度（mAP）分数。
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        fitness：返回适应度分数，这是指标的加权组合。
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        ap_class_index：返回用于计算平均精度（AP）的类别索引列表。
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
        results_dict：返回包含所有检测和分割指标和适应度分数的字典。
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize the PoseMetrics class with directory path, class names, and plotting options."""
        # 使用保存目录、绘图标志、回调函数和类别名称初始化PoseMetrics类
        super().__init__(save_dir, plot, names)  # 调用父类构造函数
        self.save_dir = save_dir  # 保存目录
        self.plot = plot  # 绘图标志
        self.on_plot = on_plot  # 回调函数
        self.names = names  # 类别名称
        self.box = Metric()  # 创建Metric实例用于计算框检测指标
        self.pose = Metric()  # 创建Metric实例用于计算姿态检测指标
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # 存储不同阶段的执行时间
        self.task = "pose"  # 任务类型为姿态检测

    def process(self, tp, tp_m, conf, pred_cls, target_cls):
        """
        Processes the detection and pose metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp（列表）：真实正例框的列表。
            tp_m (list): List of True Positive keypoints.
            tp_m（列表）：真实正例关键点的列表。
            conf (list): List of confidence scores.
            conf（列表）：置信度分数的列表。
            pred_cls (list): List of predicted classes.
            pred_cls（列表）：预测类别的列表。
            target_cls (list): List of target classes.
            target_cls（列表）：目标类别的列表。
        """
        results_pose = ap_per_class(
            tp_m,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]  # 只获取结果的后面部分
        self.pose.nc = len(self.names)  # 更新类别数量
        self.pose.update(results_pose)  # 更新姿态指标结果
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]  # 只获取结果的后面部分
        self.box.nc = len(self.names)  # 更新类别数量
        self.box.update(results_box)  # 更新框指标结果

    @property
    def keys(self):
        """Returns list of evaluation metric keys."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]
        # 返回用于访问指标的键列表

    def mean_results(self):
        """Return the mean results of box and pose."""
        return self.box.mean_results() + self.pose.mean_results()
        # 返回框和姿态的平均结果

    def class_result(self, i):
        """Return the class-wise detection results for a specific class index."""
        return self.box.class_result(i) + self.pose.class_result(i)
        # 返回指定类别索引的分类结果

    @property
    def maps(self):
        """Returns the mean average precision (mAP) per class for both box and pose detections."""
        return self.box.maps + self.pose.maps
        # 返回框和姿态检测的每个类别的平均精度（mAP）分数

    @property
    def fitness(self):
        """Computes classification metrics and speed using the `targets` and [pred](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/utils/metrics.py:356:4-370:62) inputs."""
        return self.pose.fitness() + self.box.fitness()
        # 返回姿态和框模型的适应度分数

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]
        # 返回用于访问特定指标曲线的曲线列表

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.pose.curves_results
        # 返回计算的性能指标和统计信息的字典

class ClassifyMetrics(SimpleClass):
    """
    Class for computing classification metrics including top-1 and top-5 accuracy.
    计算分类指标的类，包括 top-1 和 top-5 精度。

    Attributes:
        top1 (float): The top-1 accuracy.
        top1（浮点数）：top-1 精度。
        top5 (float): The top-5 accuracy.
        top5（浮点数）：top-5 精度。
        speed (Dict[str, float]): A dictionary containing the time taken for each step in the pipeline.
        speed（字典[str, 浮点数]）：包含每个步骤所用时间的字典。
        fitness (float): The fitness of the model, which is equal to top-5 accuracy.
        fitness（浮点数）：模型的适应度，等于 top-5 精度。
        results_dict (Dict[str, Union[float, str]]): A dictionary containing the classification metrics and fitness.
        results_dict（字典[str, 联合[浮点数, 字符串] ]）：包含分类指标和适应度的字典。
        keys (List[str]): A list of keys for the results_dict.
        keys（列表[str]）：results_dict 的键列表。
    
    Methods:
        process(targets, pred): Processes the targets and predictions to compute classification metrics.
        process(targets, pred)：处理目标和预测以计算分类指标。
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        # 初始化一个 ClassifyMetrics 实例
        self.top1 = 0  # top-1 精度初始化为 0
        self.top5 = 0  # top-5 精度初始化为 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # 存储各步骤的执行时间
        self.task = "classify"  # 任务类型为分类

    def process(self, targets, pred):
        """Target classes and predicted classes."""
        # 目标类别和预测类别
        pred, targets = torch.cat(pred), torch.cat(targets)  # 将预测和目标合并为一个张量
        correct = (targets[:, None] == pred).float()  # 计算预测是否正确
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) 精度
        self.top1, self.top5 = acc.mean(0).tolist()  # 计算 top-1 和 top-5 精度

    @property
    def fitness(self):
        """Returns mean of top-1 and top-5 accuracies as fitness score."""
        return (self.top1 + self.top5) / 2  # 返回 top-1 和 top-5 精度的平均值作为适应度分数

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))  # 返回包含模型性能指标和适应度分数的字典

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property."""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]  # 返回 results_dict 的键列表

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []  # 返回用于访问特定指标曲线的列表

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []  # 返回用于访问特定指标曲线的列表


class OBBMetrics(SimpleClass):
    """Metrics for evaluating oriented bounding box (OBB) detection, see https://arxiv.org/pdf/2106.06072.pdf."""
    # 用于评估定向边界框 (OBB) 检测的指标，见 https://arxiv.org/pdf/2106.06072.pdf。

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize an OBBMetrics instance with directory, plotting, callback, and class names."""
        # 使用目录、绘图、回调和类别名称初始化 OBBMetrics 实例
        self.save_dir = save_dir  # 保存目录
        self.plot = plot  # 绘图标志
        self.on_plot = on_plot  # 回调函数
        self.names = names  # 类别名称
        self.box = Metric()  # 创建 Metric 实例用于计算框检测指标
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # 存储不同阶段的执行时间

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        # 处理目标检测的预测结果并更新指标
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]  # 只获取结果的后面部分
        self.box.nc = len(self.names)  # 更新类别数量
        self.box.update(results)  # 更新框指标结果

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]  # 返回用于访问特定指标的键列表

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()  # 计算检测对象的平均值并返回精确率、召回率、mAP50 和 mAP50-95

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)  # 返回针对特定类别评估对象检测模型性能的结果

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps  # 返回每个类别的平均精度（mAP）分数

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()  # 返回框对象的适应度

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index  # 返回每个类别的平均精度索引

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))  # 返回计算的性能指标和统计信息的字典

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []  # 返回用于访问特定指标曲线的列表

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []  # 返回用于访问特定指标曲线的列表