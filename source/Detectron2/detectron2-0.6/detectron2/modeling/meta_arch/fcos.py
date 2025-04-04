# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import List, Optional, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms
from detectron2.structures import Boxes, ImageList, Instances, pairwise_point_box_distance
from detectron2.utils.events import get_event_storage

from ..anchor_generator import DefaultAnchorGenerator
from ..backbone import Backbone
from ..box_regression import Box2BoxTransformLinear, _dense_box_regression_loss
from .dense_detector import DenseDetector
from .retinanet import RetinaNetHead

__all__ = ["FCOS"]


logger = logging.getLogger(__name__)


class FCOS(DenseDetector):
    """
    Implement FCOS in :paper:`fcos`.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        head_in_features: Optional[List[str]] = None,
        box2box_transform=None,
        num_classes,
        center_sampling_radius: float = 1.5,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        test_score_thresh=0.2,
        test_topk_candidates=1000,
        test_nms_thresh=0.6,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
    ):
        """
        Args:
            center_sampling_radius: radius of the "center" of a groundtruth box,
                within which all anchor points are labeled positive.
            Other arguments mean the same as in :class:`RetinaNet`.
        """
        super().__init__(
            backbone, head, head_in_features, pixel_mean=pixel_mean, pixel_std=pixel_std
        )

        self.num_classes = num_classes

        # FCOS uses one anchor point per location.
        # We represent the anchor point by a box whose size equals the anchor stride.
        feature_shapes = backbone.output_shape()
        fpn_strides = [feature_shapes[k].stride for k in self.head_in_features]
        self.anchor_generator = DefaultAnchorGenerator(
            sizes=[[k] for k in fpn_strides], aspect_ratios=[1.0], strides=fpn_strides
        )

        # FCOS parameterizes box regression by a linear transform,
        # where predictions are normalized by anchor stride (equal to anchor size).
        if box2box_transform is None:
            box2box_transform = Box2BoxTransformLinear(normalize_by_size=True)
        self.box2box_transform = box2box_transform

        self.center_sampling_radius = float(center_sampling_radius)

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image

    def forward_training(self, images, features, predictions, gt_instances):
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits, pred_anchor_deltas, pred_centerness = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4, 1]
        )
        anchors = self.anchor_generator(features)
        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        return self.losses(
            anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centerness
        )

    @torch.no_grad()
    def match_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        """
        Match anchors with ground truth boxes.

        Args:
            anchors: #level boxes, from the highest resolution to lower resolution
            gt_instances: ground truth instances per image

        Returns:
            List[Tensor]:
                #image tensors, each is a vector of matched gt
                indices (or -1 for unmatched anchors) for all anchors.
        """
        num_anchors_per_level = [len(x) for x in anchors]
        anchors = Boxes.cat(anchors)  # Rx4
        anchor_centers = anchors.get_centers()  # Rx2
        anchor_sizes = anchors.tensor[:, 2] - anchors.tensor[:, 0]  # R

        lower_bound = anchor_sizes * 4
        lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound[-num_anchors_per_level[-1] :] = float("inf")

        matched_indices = []  # 存储每张图像的锚点匹配结果
        for gt_per_image in gt_instances:  # 遍历每张图像的真实标注
            gt_centers = gt_per_image.gt_boxes.get_centers()  # Nx2，获取真实框的中心坐标
            # FCOS with center sampling: anchor point must be close enough to gt center.
            # FCOS的中心采样策略：锚点必须足够接近真实框的中心
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]  # 计算锚点与真实框中心的距离是否在采样半径内
            pairwise_dist = pairwise_point_box_distance(anchor_centers, gt_per_image.gt_boxes)  # 计算锚点到真实框的距离

            # The original FCOS anchor matching rule: anchor point must be inside gt
            # 原始FCOS的锚点匹配规则：锚点必须在真实框内部
            pairwise_match &= pairwise_dist.min(dim=2).values > 0  # 确保锚点在真实框内部

            # Multilevel anchor matching in FCOS: each anchor is only responsible
            # for certain scale range.
            # FCOS的多层级锚点匹配：每个锚点只负责特定的尺度范围
            pairwise_dist = pairwise_dist.max(dim=2).values  # 获取最大距离
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (  # 应用尺度范围约束
                pairwise_dist < upper_bound[:, None]
            )

            # Match the GT box with minimum area, if there are multiple GT matches
            # 如果一个锚点匹配到多个真实框，选择面积最小的真实框
            gt_areas = gt_per_image.gt_boxes.area()  # N，计算真实框面积
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])  # 将面积信息整合到匹配矩阵中
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match，为每个锚点选择最佳匹配
            matched_idx[min_values < 1e-5] = -1  # Unmatched anchors are assigned -1，将未匹配的锚点标记为-1

            matched_indices.append(matched_idx)  # 添加当前图像的匹配结果
        return matched_indices  # 返回所有图像的匹配结果

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Same interface as :meth:`RetinaNet.label_anchors`, but implemented with FCOS
        anchor matching rule.
        与RetinaNet的label_anchors方法接口相同，但使用FCOS的锚点匹配规则实现

        Unlike RetinaNet, there are no ignored anchors.
        与RetinaNet不同，FCOS没有被忽略的锚点
        """
        matched_indices = self.match_anchors(anchors, gt_instances)  # 获取锚点匹配结果

        matched_labels, matched_boxes = [], []  # 存储匹配的标签和边界框
        for gt_index, gt_per_image in zip(matched_indices, gt_instances):  # 遍历每张图像的匹配结果
            label = gt_per_image.gt_classes[gt_index.clip(min=0)]  # 获取匹配的类别标签
            label[gt_index < 0] = self.num_classes  # background，将未匹配的锚点标记为背景类

            matched_gt_boxes = gt_per_image.gt_boxes[gt_index.clip(min=0)]  # 获取匹配的真实框

            matched_labels.append(label)  # 添加标签
            matched_boxes.append(matched_gt_boxes)  # 添加边界框
        return matched_labels, matched_boxes  # 返回匹配的标签和边界框

    def losses(
        self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centerness
    ):
        """
        This method is almost identical to :meth:`RetinaNet.losses`, with an extra
        "loss_centerness" in the returned dict.
        此方法与RetinaNet的losses方法几乎相同，但在返回的字典中额外添加了中心度损失
        """
        num_images = len(gt_labels)  # 获取图像数量
        gt_labels = torch.stack(gt_labels)  # (N, R)，将标签堆叠为张量

        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)  # 创建正样本掩码
        num_pos_anchors = pos_mask.sum().item()  # 计算正样本数量
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)  # 记录每张图像的平均正样本数
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 300)  # 使用指数移动平均更新损失归一化因子

        # classification and regression loss
        # 分类和回归损失
        gt_labels_target = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[
            :, :, :-1
        ]  # no loss for the last (background) class，不计算背景类的损失
        loss_cls = sigmoid_focal_loss_jit(  # 计算Focal Loss分类损失
            torch.cat(pred_logits, dim=1),
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = _dense_box_regression_loss(  # 计算边界框回归损失
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            [x.tensor for x in gt_boxes],
            pos_mask,
            box_reg_loss_type="giou",  # 使用GIoU损失
        )

        ctrness_targets = self.compute_ctrness_targets(anchors, gt_boxes)  # NxR，计算中心度目标
        pred_centerness = torch.cat(pred_centerness, dim=1).squeeze(dim=2)  # NxR，合并中心度预测
        ctrness_loss = F.binary_cross_entropy_with_logits(  # 计算中心度损失
            pred_centerness[pos_mask], ctrness_targets[pos_mask], reduction="sum"
        )
        return {  # 返回归一化后的损失
            "loss_fcos_cls": loss_cls / normalizer,  # 分类损失
            "loss_fcos_loc": loss_box_reg / normalizer,  # 定位损失
            "loss_fcos_ctr": ctrness_loss / normalizer,  # 中心度损失
        }

    def compute_ctrness_targets(self, anchors, gt_boxes):  # NxR
        anchors = Boxes.cat(anchors).tensor  # Rx4，将所有锚点合并为一个张量
        # 计算每个锚点到对应真实框的回归目标
        reg_targets = [self.box2box_transform.get_deltas(anchors, m.tensor) for m in gt_boxes]
        reg_targets = torch.stack(reg_targets, dim=0)  # NxRx4，堆叠所有图像的回归目标
        if len(reg_targets) == 0:  # 处理空标注的情况
            return reg_targets.new_zeros(len(reg_targets))
        left_right = reg_targets[:, :, [0, 2]]  # 提取左右边界的回归目标
        top_bottom = reg_targets[:, :, [1, 3]]  # 提取上下边界的回归目标
        # 计算中心度得分：根据边界回归目标的最小值与最大值的比值
        ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
        )
        return torch.sqrt(ctrness)  # 返回中心度得分的平方根

    def forward_inference(
        self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]
    ):
        # 调整预测结果的维度顺序
        pred_logits, pred_anchor_deltas, pred_centerness = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4, 1]  # 分别对应类别预测、边界框预测和中心度预测
        )
        anchors = self.anchor_generator(features)  # 生成锚点

        results: List[Instances] = []  # 存储每张图像的检测结果
        for img_idx, image_size in enumerate(images.image_sizes):  # 遍历每张图像
            scores_per_image = [
                # Multiply and sqrt centerness & classification scores
                # (See eqn. 4 in https://arxiv.org/abs/2006.09214)
                # 将分类得分和中心度得分相乘并开平方（参见论文公式4）
                torch.sqrt(x[img_idx].sigmoid_() * y[img_idx].sigmoid_())
                for x, y in zip(pred_logits, pred_centerness)
            ]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]  # 获取当前图像的边界框预测
            results_per_image = self.inference_single_image(  # 对单张图像进行推理
                anchors, scores_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)  # 添加当前图像的检测结果
        return results  # 返回所有图像的检测结果

    def inference_single_image(
        self,
        anchors: List[Boxes],  # 多层级的锚点
        box_cls: List[Tensor],  # 类别预测分数
        box_delta: List[Tensor],  # 边界框回归预测
        image_size: Tuple[int, int],  # 图像尺寸
    ):
        """
        Identical to :meth:`RetinaNet.inference_single_image.
        与RetinaNet的单图像推理方法相同
        """
        # 解码多层级的预测结果
        pred = self._decode_multi_level_predictions(
            anchors,
            box_cls,
            box_delta,
            self.test_score_thresh,  # 分数阈值
            self.test_topk_candidates,  # 每层保留的候选框数量
            image_size,  # 用于裁剪预测框
        )
        # 执行NMS，去除重叠的检测框
        keep = batched_nms(
            pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh
        )
        # 返回每张图像保留的最大检测框数量
        return pred[keep[: self.max_detections_per_image]]


class FCOSHead(RetinaNetHead):
    """
    The head used in :paper:`fcos`. It adds an additional centerness
    prediction branch on top of :class:`RetinaNetHead`.
    FCOS检测头网络，在RetinaNetHead的基础上增加了中心度预测分支。
    """

    def __init__(self, *, input_shape: List[ShapeSpec], conv_dims: List[int], **kwargs):
        # 初始化FCOS检测头，继承自RetinaNetHead
        # input_shape: 输入特征图的形状列表
        # conv_dims: 卷积层的通道数列表
        super().__init__(input_shape=input_shape, conv_dims=conv_dims, num_anchors=1, **kwargs)
        # Unlike original FCOS, we do not add an additional learnable scale layer
        # because it's found to have no benefits after normalizing regression targets by stride.
        # 与原始FCOS不同，这里不添加额外的可学习尺度层，因为在按步长归一化回归目标后发现没有明显收益
        self._num_features = len(input_shape)  # 记录输入特征层的数量
        # 创建中心度预测分支，使用3x3卷积层
        self.ctrness = nn.Conv2d(conv_dims[-1], 1, kernel_size=3, stride=1, padding=1)
        # 初始化中心度预测层的权重和偏置
        torch.nn.init.normal_(self.ctrness.weight, std=0.01)  # 权重使用标准差为0.01的正态分布初始化
        torch.nn.init.constant_(self.ctrness.bias, 0)  # 偏置初始化为0

    def forward(self, features):
        # 前向传播函数，处理多尺度特征图
        # features: 输入的特征图列表
        assert len(features) == self._num_features  # 确保输入特征图数量正确
        logits = []      # 存储分类预测结果
        bbox_reg = []    # 存储边界框回归预测结果
        ctrness = []     # 存储中心度预测结果
        for feature in features:  # 遍历每个尺度的特征图
            logits.append(self.cls_score(self.cls_subnet(feature)))  # 进行分类预测
            bbox_feature = self.bbox_subnet(feature)  # 提取边界框特征
            bbox_reg.append(self.bbox_pred(bbox_feature))  # 预测边界框回归值
            ctrness.append(self.ctrness(bbox_feature))  # 预测中心度值
        return logits, bbox_reg, ctrness
