# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
from typing import Dict, List
import torch

# 导入必要的模块和类
from detectron2.config import configurable  # 导入配置相关的装饰器
from detectron2.layers import ShapeSpec, batched_nms_rotated, cat  # 导入形状规范、旋转NMS和张量拼接函数
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated  # 导入实例、旋转框和IoU计算相关类
from detectron2.utils.memory import retry_if_cuda_oom  # 导入CUDA内存不足时的重试函数

from ..box_regression import Box2BoxTransformRotated  # 导入旋转框变换类
from .build import PROPOSAL_GENERATOR_REGISTRY  # 导入建议生成器注册表
from .proposal_utils import _is_tracing  # 导入追踪状态检查函数
from .rpn import RPN  # 导入基础RPN类

logger = logging.getLogger(__name__)  # 创建日志记录器


def find_top_rrpn_proposals(
    proposals,
    pred_objectness_logits,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_size,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.
    对每个特征图，选择得分最高的`pre_nms_topk`个候选框，应用NMS，裁剪候选框，并移除小框。
    如果是训练模式，返回所有特征图中得分最高的`post_nms_topk`个候选框；
    否则，返回每个特征图中得分最高的`post_nms_topk`个候选框。

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 5).
            All proposal predictions on the feature maps.
            L个张量的列表。张量i的形状为(N, Hi*Wi*A, 5)，包含特征图上的所有候选框预测。
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
            L个张量的列表。张量i的形状为(N, Hi*Wi*A)，包含候选框的目标性得分。
        image_sizes (list[tuple]): sizes (h, w) for each image
            每张图像的尺寸(h, w)列表。
        nms_thresh (float): IoU threshold to use for NMS
            NMS使用的IoU阈值。
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
            在应用NMS之前保留的得分最高的k个候选框数量。当RRPN在多个特征图上运行时（如FPN），
            这个数字是每个特征图的候选框数量。
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
            在应用NMS之后保留的得分最高的k个候选框数量。当RRPN在多个特征图上运行时（如FPN），
            这个数字是所有特征图的总候选框数量。
        min_box_size(float): minimum proposal box side length in pixels (absolute units wrt
            input images).
            候选框的最小边长（像素单位，相对于输入图像的绝对单位）。
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.
            如果候选框用于训练则为True，否则为False。此参数仅用于支持一个遗留bug。

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
            N个Instances的列表。第i个Instances存储了图像i的post_nms_topk个候选框。
    """
    # 获取图像数量和设备信息
    num_images = len(image_sizes)  # 获取批次中的图像数量
    device = proposals[0].device  # 获取设备信息（CPU或GPU）

    # 1. Select top-k anchor for every level and every image
    # 1. 为每个特征层级和每张图像选择top-k个锚点
    topk_scores = []  # #lvl Tensor, each of shape N x topk
                     # 每个层级的张量，形状为N x topk，存储分数
    topk_proposals = []  # 存储每个层级的候选框
    level_ids = []  # #lvl Tensor, each of shape (topk,)
                    # 每个层级的张量，形状为(topk,)，存储层级ID
    batch_idx = torch.arange(num_images, device=device)  # 创建批次索引张量
    # 遍历每个特征层级的候选框和对应的目标性得分
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]  # 获取当前层级的候选框数量
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
                                              # 在追踪模式下是张量
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)  # 限制候选框数量不超过pre_nms_topk
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)  # 取候选框数量和pre_nms_topk中的较小值

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # 排序比topk操作更快：https://github.com/pytorch/pytorch/issues/22812
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)  # 对logits进行降序排序
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]  # 获取前num_proposals_i个分数
        topk_idx = idx[batch_idx, :num_proposals_i]  # 获取前num_proposals_i个索引

        # each is N x topk
        # 每个张量的形状为N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 5
                                                                      # 根据索引获取对应的候选框

        topk_proposals.append(topk_proposals_i)  # 添加当前层级的候选框
        topk_scores.append(topk_scores_i)  # 添加当前层级的分数
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))  # 创建并添加层级ID张量

    # 2. Concat all levels together
    # 2. 将所有层级的结果连接在一起
    topk_scores = cat(topk_scores, dim=1)  # 连接所有层级的分数
    topk_proposals = cat(topk_proposals, dim=1)  # 连接所有层级的候选框
    level_ids = cat(level_ids, dim=0)  # 连接所有层级的ID

    # 3. For each image, run a per-level NMS, and choose topk results.
    # 3. 对每张图像执行每个层级的NMS，并选择topk个结果
    results = []  # 存储处理结果
    for n, image_size in enumerate(image_sizes):  # 遍历每张图像
        boxes = RotatedBoxes(topk_proposals[n])  # 获取当前图像的候选框
        scores_per_img = topk_scores[n

        # filter empty boxes
        # 过滤空的框
        keep = boxes.nonempty(threshold=min_box_size)  # 移除小于阈值的框
        lvl = level_ids  # 获取层级ID
        if _is_tracing() or keep.sum().item() != len(boxes):  # 如果在追踪模式下或存在需要移除的框
            boxes, scores_per_img, lvl = (boxes[keep], scores_per_img[keep], level_ids[keep])  # 更新保留的框、分数和层级ID

        keep = batched_nms_rotated(boxes.tensor, scores_per_img, lvl, nms_thresh)  # 执行批量NMS操作
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        # 在Detectron1中，训练和测试时的行为不同：
        # 训练时，topk是在训练批次中所有图像的候选框上进行的；
        # 测试时，是在每张图像的候选框上单独进行的。
        # 这导致训练行为依赖于批次大小，配置"POST_NMS_TOPK_TRAIN"最终依赖于批次大小。
        # Detectron2修复了这个bug，使行为不再依赖于批次大小。
        keep = keep[:post_nms_topk]  # keep is already sorted
                                    # 保留排序后的前post_nms_topk个框

        res = Instances(image_size)  # 创建Instances对象存储结果
        res.proposal_boxes = boxes[keep]  # 保存筛选后的候选框
        res.objectness_logits = scores_per_img[keep]  # 保存筛选后的目标性得分
        results.append(res)  # 将结果添加到列表中
    return results


@PROPOSAL_GENERATOR_REGISTRY.register()  # 注册RRPN类到建议生成器注册表
class RRPN(RPN):  # 继承自基础RPN类
    """
    Rotated Region Proposal Network described in :paper:`RRPN`.
    旋转区域建议网络，在RRPN论文中描述。
    """

    @configurable  # 使用configurable装饰器使类可配置
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类RPN的初始化方法
        if self.anchor_boundary_thresh >= 0:  # 检查锚框边界阈值
            raise NotImplementedError(
                "anchor_boundary_thresh is a legacy option not implemented for RRPN."
                # 锚框边界阈值是一个未在RRPN中实现的遗留选项
            )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):  # 从配置创建RRPN实例的类方法
        ret = super().from_config(cfg, input_shape)  # 调用父类的配置加载方法
        ret["box2box_transform"] = Box2BoxTransformRotated(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)  # 使用旋转框变换替换原有的框变换
        return ret

    @torch.no_grad()  # 禁用梯度计算，提高效率和减少内存使用
    def label_and_sample_anchors(self, anchors: List[RotatedBoxes], gt_instances: List[Instances]):
        """
        Args:
            anchors (list[RotatedBoxes]): anchors for each feature map.
            # 每个特征图的旋转锚框列表
            gt_instances: the ground-truth instances for each image.
            # 每张图像的真实标注实例

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across feature maps. Label values are in {-1, 0, 1},
                with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
                # 图像数量的张量列表。第i个元素是标签向量，其长度是所有特征图上的锚框总数。
                # 标签值为{-1, 0, 1}，含义：-1=忽略；0=负类；1=正类。
            list[Tensor]:
                i-th element is a Nx5 tensor, where N is the total number of anchors across
                feature maps.  The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as 1.
                # 第i个元素是Nx5的张量，N是所有特征图上的锚框总数。
                # 这些值是每个锚框匹配的真实框。对于标签不为1的锚框，其值是未定义的。
        """
        anchors = RotatedBoxes.cat(anchors)  # 将所有特征图的锚框合并为一个张量

        gt_boxes = [x.gt_boxes for x in gt_instances]  # 提取每个实例的真实框
        del gt_instances  # 删除不再需要的实例对象，释放内存

        gt_labels = []  # 存储每个图像的锚框标签
        matched_gt_boxes = []  # 存储每个图像中与锚框匹配的真实框
        for gt_boxes_i in gt_boxes:
            """
            gt_boxes_i: ground-truth boxes for i-th image
            # 第i张图像的真实框
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou_rotated)(gt_boxes_i, anchors)  # 计算旋转框的IoU匹配质量矩阵
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)  # 根据IoU进行锚框匹配
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            # 匹配过程内存开销大，可能导致张量被移到CPU，但结果数据量小
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)  # 确保标签和真实框在同一设备上

            # A vector of labels (-1, 0, 1) for each anchor
            # 为每个锚框生成标签向量(-1, 0, 1)
            gt_labels_i = self._subsample_labels(gt_labels_i)  # 对标签进行子采样，保持正负样本平衡

            if len(gt_boxes_i) == 0:  # 如果当前图像没有真实框
                # These values won't be used anyway since the anchor is labeled as background
                # 这些值不会被使用，因为所有锚框都被标记为背景
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)  # 创建全零张量作为匹配的真实框
            else:
                # TODO wasted indexing computation for ignored boxes
                # TODO 对被忽略的框进行了不必要的索引计算
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor  # 获取与锚框匹配的真实框坐标

            gt_labels.append(gt_labels_i)  # N,AHW  # 添加当前图像的锚框标签
            matched_gt_boxes.append(matched_gt_boxes_i)  # 添加当前图像的匹配真实框
        return gt_labels, matched_gt_boxes  # 返回所有图像的锚框标签和匹配的真实框

    @torch.no_grad()  # 预测阶段不需要计算梯度
    def predict_proposals(self, anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes):
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)  # 将预测的偏移量解码为旋转框坐标
        return find_top_rrpn_proposals(  # 筛选得分最高的旋转区域建议
            pred_proposals,  # 预测的旋转建议框
            pred_objectness_logits,  # 预测的目标性得分
            image_sizes,  # 图像尺寸列表
            self.nms_thresh,  # NMS阈值
            self.pre_nms_topk[self.training],  # NMS前保留的建议框数量
            self.post_nms_topk[self.training],  # NMS后保留的建议框数量
            self.min_box_size,  # 最小框尺寸
            self.training,  # 是否处于训练模式
        )
