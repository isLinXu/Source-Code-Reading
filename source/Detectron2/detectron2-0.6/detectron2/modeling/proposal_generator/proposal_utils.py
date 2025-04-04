# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List, Tuple, Union
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances

logger = logging.getLogger(__name__)


def _is_tracing():
    # (fixed in TORCH_VERSION >= 1.9)
    # 在PyTorch 1.9及以上版本中已修复
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        # 如果是在脚本模式下，返回False
        return False
    else:
        # 返回是否在追踪模式下
        return torch.jit.is_tracing()


def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.
    对每个特征图，选择得分最高的`pre_nms_topk`个候选框，应用NMS，裁剪候选框，并移除小框。
    返回所有特征图中每张图像得分最高的`post_nms_topk`个候选框。

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
            L个张量的列表。张量i的形状为(N, Hi*Wi*A, 4)，包含特征图上的所有候选框预测。
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
            L个张量的列表。张量i的形状为(N, Hi*Wi*A)，包含候选框的目标性得分。
        image_sizes (list[tuple]): sizes (h, w) for each image
            每张图像的尺寸(h, w)列表。
        nms_thresh (float): IoU threshold to use for NMS
            NMS使用的IoU阈值。
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
            在应用NMS之前保留的得分最高的k个候选框数量。当RPN在多个特征图上运行时（如FPN），
            这个数字是每个特征图的候选框数量。
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
            在应用NMS之后保留的得分最高的k个候选框数量。当RPN在多个特征图上运行时（如FPN），
            这个数字是所有特征图的总候选框数量。
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
            候选框的最小边长（像素单位，相对于输入图像的绝对单位）。
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.
            如果候选框用于训练则为True，否则为False。此参数仅用于支持一个遗留bug。

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
            N个Instances的列表。第i个Instances存储了图像i的post_nms_topk个候选框，
            按目标性得分降序排序。
    """
    num_images = len(image_sizes)  # 获取图像数量
    device = proposals[0].device  # 获取设备信息

    # 1. Select top-k anchor for every level and every image
    # 1. 为每个层级和每张图像选择top-k个锚点
    topk_scores = []  # #lvl Tensor, each of shape N x topk
                     # 每个层级的张量，形状为N x topk，存储分数
    topk_proposals = []  # 存储每个层级的候选框
    level_ids = []  # #lvl Tensor, each of shape (topk,)
                    # 每个层级的张量，形状为(topk,)，存储层级ID
    batch_idx = torch.arange(num_images, device=device)  # 创建批次索引张量
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]  # 获取当前层级的候选框数量
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
                                              # 在追踪模式下是张量
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)  # 限制候选框数量不超过pre_nms_topk
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)  # 取候选框数量和pre_nms_topk中的较小值

        # sort is faster than topk: https://github.com/pytorch/pytorch/issues/22812
        # 排序比topk操作更快：https://github.com/pytorch/pytorch/issues/22812
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)  # 对logits进行降序排序
        topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)  # 获取前num_proposals_i个分数
        topk_idx = idx.narrow(1, 0, num_proposals_i)  # 获取前num_proposals_i个索引

        # each is N x topk
        # 每个张量的形状为N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4
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
    results: List[Instances] = []  # 存储处理结果
    for n, image_size in enumerate(image_sizes):  # 遍历每张图像
        boxes = Boxes(topk_proposals[n])  # 获取当前图像的候选框
        scores_per_img = topk_scores[n]  # 获取当前图像的分数
        lvl = level_ids  # 获取层级ID

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)  # 创建有效性掩码，排除无穷和NaN值
        if not valid_mask.all():  # 如果存在无效值
            if training:  # 如果在训练模式下
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                    # 预测的框或分数包含Inf/NaN值，训练已发散
                )
            boxes = boxes[valid_mask]  # 保留有效的框
            scores_per_img = scores_per_img[valid_mask]  # 保留有效的分数

        # filter empty boxes
        # 过滤空的框
        keep = boxes.nonempty(threshold=min_box_size)  # 移除小于阈值的框
        if _is_tracing() or keep.sum().item() != len(boxes):  # 如果在追踪模式下或存在需要移除的框
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]  # 更新保留的框、分数和层级ID

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)  # 执行批量NMS操作
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


def add_ground_truth_to_proposals(
    gt: Union[List[Instances], List[Boxes]], proposals: List[Instances]
) -> List[Instances]:
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.
    调用`add_ground_truth_to_proposals_single_image`处理所有图像。

    Args:
        gt(Union[List[Instances], List[Boxes]): list of N elements. Element i is a Instances
            representing the ground-truth for image i.
            N个元素的列表。元素i是表示图像i的真实标注的Instances对象。
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.
            N个元素的列表。元素i是表示图像i的候选框的Instances对象。

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
            N个Instances的列表。每个元素是图像的候选框，包含"proposal_boxes"和"objectness_logits"字段。
    """
    assert gt is not None  # 确保真实标注不为空

    if len(proposals) != len(gt):
        raise ValueError("proposals and gt should have the same length as the number of images!")
        # 确保候选框和真实标注的数量与图像数量相同
    if len(proposals) == 0:
        return proposals  # 如果没有候选框，直接返回

    return [
        add_ground_truth_to_proposals_single_image(gt_i, proposals_i)
        for gt_i, proposals_i in zip(gt, proposals)  # 对每张图像的真实标注和候选框进行处理
    ]


def add_ground_truth_to_proposals_single_image(
    gt: Union[Instances, Boxes], proposals: Instances
) -> Instances:
    """
    Augment `proposals` with `gt`.
    使用真实标注`gt`增强候选框`proposals`。

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt and proposals
        per image.
        与`add_ground_truth_to_proposals`相同，但处理单张图像的真实标注和候选框。

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
        与`add_ground_truth_to_proposals`相同，但只处理一张图像。
    """
    if isinstance(gt, Boxes):
        # convert Boxes to Instances
        # 将Boxes转换为Instances对象
        gt = Instances(proposals.image_size, gt_boxes=gt)

    gt_boxes = gt.gt_boxes  # 获取真实框
    device = proposals.objectness_logits.device  # 获取设备信息
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    # 为所有真实框分配目标性得分，使得P(object) = sigmoid(logit) ≈ 1
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))  # 计算目标性得分值
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)  # 创建真实框的目标性得分张量

    # Concatenating gt_boxes with proposals requires them to have the same fields
    # 将真实框与候选框连接需要它们具有相同的字段
    gt_proposal = Instances(proposals.image_size, **gt.get_fields())  # 创建真实框的Instances对象
    gt_proposal.proposal_boxes = gt_boxes  # 设置候选框
    gt_proposal.objectness_logits = gt_logits  # 设置目标性得分

    for key in proposals.get_fields().keys():  # 检查字段一致性
        assert gt_proposal.has(
            key
        ), "The attribute '{}' in `proposals` does not exist in `gt`".format(key)

    # NOTE: Instances.cat only use fields from the first item. Extra fields in latter items
    # will be thrown away.
    # 注意：Instances.cat只使用第一个项目的字段，后面项目的额外字段将被丢弃
    new_proposals = Instances.cat([proposals, gt_proposal])  # 连接候选框和真实框

    return new_proposals  # 返回增强后的候选框
