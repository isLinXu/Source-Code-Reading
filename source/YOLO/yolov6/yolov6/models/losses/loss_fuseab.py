#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.tal_assigner import TaskAlignedAssigner


class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=0,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5},
                ):
        """
        初始化损失计算类。

        参数:
        - fpn_strides: FPN各层的步长，默认为 [8, 16, 32]
        - grid_cell_size: 网格单元大小，默认为 5.0
        - grid_cell_offset: 网格单元偏移量，默认为 0.5
        - num_classes: 类别数量，默认为 80
        - ori_img_size: 原始图像大小，默认为 640
        - warmup_epoch: 预热轮数，默认为 0
        - use_dfl: 是否使用分布焦点损失，默认为 True
        - reg_max: 分布焦点损失的最大值，默认为 16
        - iou_type: IOU 类型，默认为 'giou'
        - loss_weight: 损失权重，默认为 {'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        """
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        # 任务对齐分配器用于目标分配
        self.formal_assigner = TaskAlignedAssigner(topk=26, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        # 初始化分布焦点损失的投影参数proj
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        # 变焦损失用于分类
        self.varifocal_loss = VarifocalLoss().cuda()
        # 边界框损失用于边界框回归
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight

    def __call__(
        self,
        outputs,
        targets,
        epoch_num,
        step_num,
        batch_height,
        batch_width
    ):
        """
        计算给定输入的损失。

        参数:
        - outputs: 模型输出，包括特征图、预测分数和预测分布
        - targets: 目标标签和边界框
        - epoch_num: 当前训练轮数
        - step_num: 当前训练步数
        - batch_height: 批次高度
        - batch_width: 批次宽度

        返回:
        - loss: 总损失
        - losses: 各项损失的详细信息
        """
        feats, pred_scores, pred_distri = outputs
        # 生成输入特征图的锚点
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device, is_eval=False, mode='ab')

        assert pred_scores.type() == pred_distri.type()
        # 缩放真值边界框
        gt_bboxes_scale = torch.tensor([batch_width, batch_height, batch_width, batch_height]).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # targets
        # 处理目标数据，包括标签、边界框和掩码
        targets =self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:] #xyxy # xyxy 格式
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        # 计算预测边界框
        anchor_points_s = anchor_points / stride_tensor
        pred_distri[..., :2] += anchor_points_s
        pred_bboxes = xywh2xyxy(pred_distri)

        try:
            # 使用任务对齐分配器分配目标
            target_labels, target_bboxes, target_scores, fg_mask = \
                self.formal_assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    gt_labels,
                    gt_bboxes,
                    mask_gt)

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")

            _pred_scores = pred_scores.detach().cpu().float()
            _pred_bboxes = pred_bboxes.detach().cpu().float()
            _anchor_points = anchor_points.cpu().float()
            _gt_labels = gt_labels.cpu().float()
            _gt_bboxes = gt_bboxes.cpu().float()
            _mask_gt = mask_gt.cpu().float()
            _stride_tensor = stride_tensor.cpu().float()

            target_labels, target_bboxes, target_scores, fg_mask = \
                self.formal_assigner(
                    _pred_scores,
                    _pred_bboxes * _stride_tensor,
                    _anchor_points,
                    _gt_labels,
                    _gt_bboxes,
                    _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()

        #Dynamic release GPU memory
        # 动态释放 GPU 内存
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        # 重新缩放边界框
        target_bboxes /= stride_tensor

        # cls loss
        # 计算分类损失
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        # 避免除以零错误，如果 target_scores_sum 为 0，loss_cls 也为 0
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum

        # bbox loss
        # 计算边界框损失
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        # 计算总损失
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl

        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                         (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        对目标数据进行预处理，以适应模型训练或推理的需求。

        参数:
        targets (Tensor): 原始目标数据，包含目标信息。
        batch_size (int): 批处理大小，用于定义输出的批量维度。
        scale_tensor (Tensor): 用于缩放目标数据的张量。

        返回:
        Tensor: 预处理后的目标数据。
        """
        # 初始化一个空的列表，用于存放每个样本的目标数据
        targets_list = np.zeros((batch_size, 1, 5)).tolist()

        # 遍历目标数据，将其添加到对应的样本索引位置
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])

        # 计算样本中目标数量的最大值，并将每个样本的目标数据扩展到最大长度
        max_len = max((len(l) for l in targets_list))

        # 对每个样本的目标数据进行填充，以确保每个样本有相同数量的目标
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)

        # 对目标数据的坐标进行缩放处理
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)

        # 将目标数据的坐标格式从中心点+宽高转换为左上角+右下角的格式
        targets[..., 1:] = xywh2xyxy(batch_target)

        # 返回预处理后的目标数据
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        """
        解码边界框。

        对给定的锚点和预测分布进行解码，以获得最终的边界框坐标。

        参数:
        - anchor_points: 锚点坐标，用于边界框的解码。
        - pred_dist: 预测的边界框分布，如果使用了分布式 focal loss (DFL)，则需要对其进行转换。

        返回:
        解码后的边界框坐标。
        """
        # 如果使用了分布式 focal loss (DFL)，则对预测分布进行转换
        if self.use_dfl:
            # 获取预测分布的形状信息
            batch_size, n_anchors, _ = pred_dist.shape
            # 对预测分布进行softmax处理，并将其转换为边界框坐标
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(
                self.proj.to(pred_dist.device))
        # 根据预测分布和锚点，计算最终的边界框坐标
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    """
    实现Varifocal Loss的PyTorch模块。

    Varifocal Loss是一种用于处理类别不平衡问题的损失函数，特别适用于目标检测任务。
    它通过调整样本的权重来减少简单样本的贡献，同时增加困难样本的贡献，从而提高模型的性能。
    """
    def __init__(self):
        """
        初始化VarifocalLoss模块。
        """
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):
        """
        计算Varifocal Loss。

        参数:
        pred_score (Tensor): 模型预测的得分。
        gt_score (Tensor): 地面真实得分。
        label (Tensor): 标签，指示每个样本属于哪个类别。
        alpha (float): 损失函数的alpha参数，用于调整负样本的权重，默认值为0.75。
        gamma (float): 损失函数的gamma参数，用于调整困难样本的权重，默认值为2.0。

        返回:
        loss (Tensor): 计算得到的Varifocal Loss。
        """
        # 计算样本权重，对正样本和负样本分别计算权重，alpha是负样本的权重，gamma是困难样本的权重
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

        # 禁用自动混合精度，确保损失计算的准确性
        with torch.cuda.amp.autocast(enabled=False):

            # 计算二元交叉熵损失，并应用样本权重
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    """
    BboxLoss类用于计算边界框的损失，包括IoU损失和分布焦点损失（DFL）。

    参数:
    - num_classes: 类别的数量
    - reg_max: 分布焦点损失的最大回归值
    - use_dfl: 是否使用分布焦点损失，默认为False
    - iou_type: 计算IoU损失的类型，默认为'giou'
    """
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        # 初始化BboxLoss类，并初始化相关参数
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        # 初始化IoU损失计算对象
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        前向传播函数，计算损失。

        参数:
        - pred_dist: 预测的分布
        - pred_bboxes: 预测的边界框
        - anchor_points: 锚点坐标
        - target_bboxes: 目标边界框
        - target_scores: 目标分数
        - target_scores_sum: 目标分数的总和
        - fg_mask: 前景掩码，用于区分正样本和负样本

        返回:
        - loss_iou: IoU损失
        - loss_dfl: 分布焦点损失（如果使用）
        """
        # 选择正样本掩码
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # 计算IoU损失
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
            # 计算DFL损失
            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            # 当没有正样本时，将IoU损失和DFL损失都设为0
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        # 将目标值转换为长整型
        target_left = target.to(torch.long)
        # 计算右侧的目标值
        target_right = target_left + 1
        # 计算左侧权重
        weight_left = target_right.to(torch.float) - target
        # 计算右侧权重
        weight_right = 1 - weight_left

        # 计算左侧损失
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        # 计算右侧损失
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right

        # 返回左侧损失和右侧损失的加权平均值
        return (loss_left + loss_right).mean(-1, keepdim=True)
