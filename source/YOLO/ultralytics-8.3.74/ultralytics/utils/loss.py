# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块

from ultralytics.utils.metrics import OKS_SIGMA  # 从ultralytics.utils.metrics导入OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh  # 从ultralytics.utils.ops导入操作函数
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors  # 从ultralytics.utils.tal导入任务分配器和相关函数
from ultralytics.utils.torch_utils import autocast  # 从ultralytics.utils.torch_utils导入自动类型转换

from .metrics import bbox_iou, probiou  # 从当前模块导入边界框IoU计算函数
from .tal import bbox2dist  # 从当前模块导入边界框到距离的转换函数


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.
    Zhang等人提出的变焦损失。

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class.
        初始化VarifocalLoss类。"""
        super().__init__()  # 调用父类构造函数

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss.
        计算变焦损失。"""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label  # 计算权重
        with autocast(enabled=False):  # 禁用自动类型转换
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)  # 计算二元交叉熵损失
                .mean(1)  # 对每个样本取平均
                .sum()  # 求和
            )
        return loss  # 返回损失


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters.
        FocalLoss类的初始化器，无参数。"""
        super().__init__()  # 调用父类构造函数

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks.
        计算并更新目标检测/分类任务的混淆矩阵。"""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")  # 计算二元交叉熵损失
        # p_t = torch.exp(-loss)  # 计算p_t
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # 非零幂以确保梯度稳定

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # 从logits计算概率
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)  # 计算p_t
        modulating_factor = (1.0 - p_t) ** gamma  # 计算调制因子
        loss *= modulating_factor  # 应用调制因子
        if alpha > 0:  # 如果alpha大于0
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)  # 计算alpha因子
            loss *= alpha_factor  # 应用alpha因子
        return loss.mean(1).sum()  # 返回损失的平均值


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training.
    训练期间计算DFL损失的标准类。"""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module.
        初始化DFL模块。"""
        super().__init__()  # 调用父类构造函数
        self.reg_max = reg_max  # 设置最大正则化值

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.
        返回左侧和右侧DFL损失的总和。

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)  # 限制目标值范围
        tl = target.long()  # target left 目标左侧
        tr = tl + 1  # target right 目标右侧
        wl = tr - target  # weight left 左侧权重
        wr = 1 - wl  # weight right 右侧权重
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl  # 计算左侧DFL损失
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr  # 计算右侧DFL损失
        ).mean(-1, keepdim=True)  # 返回平均损失


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training.
    训练期间计算训练损失的标准类。"""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.
        使用正则化最大值和DFL设置初始化BboxLoss模块。"""
        super().__init__()  # 调用父类构造函数
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None  # 初始化DFL损失

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss.
        计算IoU损失。"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # 计算权重
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)  # 计算IoU
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum  # 计算IoU损失

        # DFL loss
        if self.dfl_loss:  # 如果存在DFL损失
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)  # 将目标边界框转换为距离
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight  # 计算DFL损失
            loss_dfl = loss_dfl.sum() / target_scores_sum  # 归一化DFL损失
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)  # 如果没有DFL损失，返回0.0

        return loss_iou, loss_dfl  # 返回IoU损失和DFL损失


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training.
    训练期间计算训练损失的标准类。"""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.
        使用正则化最大值和DFL设置初始化BboxLoss模块。"""
        super().__init__(reg_max)  # 调用父类构造函数

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss.
        计算IoU损失。"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # 计算权重
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])  # 计算IoU
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum  # 计算IoU损失

        # DFL loss
        if self.dfl_loss:  # 如果存在DFL损失
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)  # 将目标边界框转换为距离
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight  # 计算DFL损失
            loss_dfl = loss_dfl.sum() / target_scores_sum  # 归一化DFL损失
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)  # 如果没有DFL损失，返回0.0

        return loss_iou, loss_dfl  # 返回IoU损失和DFL损失


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses.
    计算训练损失的标准类。"""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class.
        初始化KeypointLoss类。"""
        super().__init__()  # 调用父类构造函数
        self.sigmas = sigmas  # 设置sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints.
        计算预测和实际关键点的关键点损失因子和欧几里得距离损失。"""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)  # 计算欧几里得距离
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)  # 计算关键点损失因子
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()  # 返回关键点损失


class v8DetectionLoss:
    """Criterion class for computing training losses.
    计算训练损失的标准类。"""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function.
        使用模型初始化v8DetectionLoss，定义与模型相关的属性和BCE损失函数。"""
        device = next(model.parameters()).device  # get model device 获取模型设备
        h = model.args  # hyperparameters 超参数

        m = model.model[-1]  # Detect() module 检测模块
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # 初始化二元交叉熵损失
        self.hyp = h  # 保存超参数
        self.stride = m.stride  # model strides 模型步幅
        self.nc = m.nc  # number of classes 类别数量
        self.no = m.nc + m.reg_max * 4  # 输出数量
        self.reg_max = m.reg_max  # 最大正则化值
        self.device = device  # 保存设备

        self.use_dfl = m.reg_max > 1  # 是否使用DFL

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)  # 初始化任务分配器
        self.bbox_loss = BboxLoss(m.reg_max).to(device)  # 初始化边界框损失
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)  # 创建投影数组

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor.
        预处理目标计数并与输入批量大小匹配以输出张量。"""
        nl, ne = targets.shape  # 获取目标的行数和列数
        if nl == 0:  # 如果没有目标
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)  # 返回一个空的张量
        else:
            i = targets[:, 0]  # image index 图像索引
            _, counts = i.unique(return_counts=True)  # 计算每个图像的目标数量
            counts = counts.to(dtype=torch.int32)  # 转换为int32类型
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)  # 创建输出张量
            for j in range(batch_size):  # 遍历每个批次
                matches = i == j  # 找到当前批次的匹配目标
                if n := matches.sum():  # 如果有匹配的目标
                    out[j, :n] = targets[matches, 1:]  # 将匹配的目标填入输出张量
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))  # 将边界框从xywh转换为xyxy
        return out  # 返回处理后的张量

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution.
        从锚点和分布解码预测的对象边界框坐标。"""
        if self.use_dfl:  # 如果使用DFL
            b, a, c = pred_dist.shape  # batch, anchors, channels 批次、锚点、通道
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))  # 计算预测分布
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)  # 将分布转换为边界框

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        计算边界框、类别和DFL的损失总和，乘以批量大小。"""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl 初始化损失张量
        feats = preds[1] if isinstance(preds, tuple) else preds  # 获取预测特征
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(  # 分割预测分布和得分
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # 调整得分张量的维度
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # 调整分布张量的维度

        dtype = pred_scores.dtype  # 获取数据类型
        batch_size = pred_scores.shape[0]  # 获取批量大小
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像尺寸（高，宽）
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # 创建锚点和步幅张量

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)  # 合并目标信息
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # 预处理目标
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy  分割目标标签和边界框
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # 创建目标掩码

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4) 解码预测边界框
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(  # 使用分配器获取目标信息
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),  # 获取预测得分
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  # 获取预测边界框
            anchor_points * stride_tensor,  # 获取锚点
            gt_labels,  # 获取目标标签
            gt_bboxes,  # 获取目标边界框
            mask_gt,  # 获取目标掩码
        )

        target_scores_sum = max(target_scores.sum(), 1)  # 计算目标得分总和

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE 计算类别损失

        # Bbox loss
        if fg_mask.sum():  # 如果前景掩码有值
            target_bboxes /= stride_tensor  # 归一化目标边界框
            loss[0], loss[2] = self.bbox_loss(  # 计算边界框损失和DFL损失
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain  乘以边界框增益
        loss[1] *= self.hyp.cls  # cls gain  乘以类别增益
        loss[2] *= self.hyp.dfl  # dfl gain  乘以DFL增益

        return loss.sum() * batch_size, loss.detach()  # 返回损失总和和分离的损失


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""
    # 该类用于计算训练损失的标准类

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        # 初始化v8SegmentationLoss类，接受一个去并行化的模型作为参数
        super().__init__(model)
        self.overlap = model.args.overlap_mask  # 保存模型的重叠掩码参数

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        # 计算并返回YOLO模型的损失
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        # 初始化损失张量，包含框损失、类别损失和分布损失
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        # 解包预测结果，获取特征、预测掩码和原型
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        # 获取批次大小、掩码数量、掩码高度和宽度
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # 合并预测分布和得分

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # 重新排列得分张量
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # 重新排列预测分布张量
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()  # 重新排列预测掩码张量

        dtype = pred_scores.dtype  # 获取得分张量的数据类型
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # 计算图像尺寸（高度和宽度）
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # 创建锚点和步幅张量

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)  # 获取批次索引
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)  # 合并目标信息
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            # 预处理目标数据
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            # 分离出类别标签和边界框
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # 生成掩码
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e
            # 捕获运行时错误并抛出类型错误，提示数据集格式不正确

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # 解码预测边界框

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )  # 进行目标分配

        target_scores_sum = max(target_scores.sum(), 1)  # 计算目标得分总和

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # 计算类别损失

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )  # 计算边界框损失
            # Masks loss
            masks = batch["masks"].to(self.device).float()  # 获取真实掩码并转换为浮点型
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]  # 如果需要，调整掩码尺寸

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )  # 计算分割损失

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            # 如果没有前景掩码，避免产生无用梯度

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain
        # 根据超参数调整损失

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            # 真实掩码，形状为(n, H, W)，n为对象数量
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            # 预测掩码系数，形状为(n, 32)
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            # 原型掩码，形状为(32, H, W)
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            # 真实边界框，格式为xyxy，归一化到[0, 1]，形状为(n, 4)
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).
            # 每个真实边界框的面积，形状为(n,)

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.
            # 返回计算出的单张图像的掩码损失
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        # 通过原型和预测系数计算预测掩码
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")  # 计算二元交叉熵损失
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()  # 返回归一化后的损失

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            # 二元张量，形状为(BS, N_anchors)，指示哪些锚点是正样本
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            # 真实掩码，形状为(BS, H, W)如果`overlap`为False，否则为(BS, ?, H, W)
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            # 每个锚点的真实对象索引，形状为(BS, N_anchors)
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            # 每个锚点的真实边界框，形状为(BS, N_anchors, 4)
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            # 批次索引，形状为(N_labels_in_batch, 1)
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            # 原型掩码，形状为(BS, 32, H, W)
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            # 每个锚点的预测掩码，形状为(BS, N_anchors, 32)
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            # 输入图像的大小，形状为(2)，即(H, W)
            overlap (bool): Whether the masks in `masks` tensor overlap.
            # 掩码张量中的掩码是否重叠

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.
            # 返回计算出的实例分割损失
        """
        _, _, mask_h, mask_w = proto.shape  # 获取掩码的高度和宽度
        loss = 0  # 初始化损失

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]  # 将目标边界框归一化到0-1范围

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)  # 计算目标边界框的面积

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)  # 将边界框归一化到掩码大小

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            # 解包前景掩码、目标索引、预测掩码、原型、归一化边界框和面积
            if fg_mask_i.any():  # 如果有前景掩码
                mask_idx = target_gt_idx_i[fg_mask_i]  # 获取目标索引
                if overlap:  # 如果掩码重叠
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)  # 生成真实掩码
                    gt_mask = gt_mask.float()  # 转换为浮点型
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]  # 获取非重叠的真实掩码

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )  # 计算单个掩码的损失并累加

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
                # 如果没有前景掩码，避免产生无用梯度

        return loss / fg_mask.sum()  # 返回平均损失

class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""
    # 该类用于计算训练损失的标准类

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        # 初始化v8PoseLoss，设置关键点变量并声明关键点损失实例
        super().__init__(model)  # 调用父类构造函数
        self.kpt_shape = model.model[-1].kpt_shape  # 获取关键点的形状
        self.bce_pose = nn.BCEWithLogitsLoss()  # 初始化二元交叉熵损失
        is_pose = self.kpt_shape == [17, 3]  # 判断是否为姿态估计
        nkpt = self.kpt_shape[0]  # 关键点的数量
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        # 根据是否为姿态估计设置sigma值
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)  # 初始化关键点损失

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        # 计算总损失并返回
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        # 初始化损失张量，包含框损失、类别损失、分布损失、关键点位置损失和关键点可见性损失
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        # 解包预测结果，获取特征和预测关键点
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # 合并预测分布和得分

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # 重新排列得分张量
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # 重新排列预测分布张量
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()  # 重新排列预测关键点张量

        dtype = pred_scores.dtype  # 获取得分张量的数据类型
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # 计算图像尺寸（高度和宽度）
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # 创建锚点和步幅张量

        # Targets
        batch_size = pred_scores.shape[0]  # 获取批次大小
        batch_idx = batch["batch_idx"].view(-1, 1)  # 获取批次索引
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)  # 合并目标信息
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # 预处理目标数据
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 分离出类别标签和边界框
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # 生成掩码

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # 解码预测边界框
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)
        # 解码预测关键点

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )  # 进行目标分配

        target_scores_sum = max(target_scores.sum(), 1)  # 计算目标得分总和

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # 计算类别损失

        # Bbox loss
        if fg_mask.sum():  # 如果有前景掩码
            target_bboxes /= stride_tensor  # 将目标边界框除以步幅
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )  # 计算边界框损失
            keypoints = batch["keypoints"].to(self.device).float().clone()  # 获取关键点并转换为浮点型
            keypoints[..., 0] *= imgsz[1]  # 将关键点的x坐标乘以图像宽度
            keypoints[..., 1] *= imgsz[0]  # 将关键点的y坐标乘以图像高度

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )  # 计算关键点损失

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        # 根据超参数调整损失

        return loss.sum() * batch_size, loss.detach()  # 返回总损失和分离的损失

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        # 解码预测关键点到图像坐标
        y = pred_kpts.clone()  # 克隆预测关键点
        y[..., :2] *= 2.0  # 将前两个坐标乘以2
        y[..., 0] += anchor_points[:, [0]] - 0.5  # 调整x坐标
        y[..., 1] += anchor_points[:, [1]] - 0.5  # 调整y坐标
        return y  # 返回调整后的关键点

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            # 二元掩码张量，指示对象的存在，形状为(BS, N_anchors)
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            # 索引张量，将锚点映射到真实对象，形状为(BS, N_anchors)
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            # 真实关键点，形状为(N_kpts_in_batch, N_kpts_per_object, kpts_dim)
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            # 关键点的批次索引张量，形状为(N_kpts_in_batch, 1)
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            # 锚点的步幅张量，形状为(N_anchors, 1)
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            # 真实边界框，形状为(BS, N_anchors, 4)，格式为(x1, y1, x2, y2)
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
            # 预测关键点，形状为(BS, N_anchors, N_kpts_per_object, kpts_dim)

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            # 返回关键点损失
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            # 返回关键点对象损失
        """
        batch_idx = batch_idx.flatten()  # 将批次索引展平
        batch_size = len(masks)  # 获取批次大小

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()  # 找到单张图像中关键点的最大数量

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )  # 创建用于存放批次关键点的张量

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):  # 遍历每个批次
            keypoints_i = keypoints[batch_idx == i]  # 获取当前批次的关键点
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i  # 填充批次关键点

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)  # 扩展目标索引的维度

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )  # 根据目标索引从批次关键点中选择关键点

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)  # 将坐标除以步幅

        kpts_loss = 0  # 初始化关键点损失
        kpts_obj_loss = 0  # 初始化关键点对象损失

        if masks.any():  # 如果有掩码
            gt_kpt = selected_keypoints[masks]  # 获取真实关键点
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)  # 计算目标边界框的面积
            pred_kpt = pred_kpts[masks]  # 获取预测关键点
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            # 创建关键点掩码，指示关键点是否存在
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # 计算姿态损失

            if pred_kpt.shape[-1] == 3:  # 如果预测关键点有z坐标
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # 计算关键点对象损失

        return kpts_loss, kpts_obj_loss  # 返回关键点损失和关键点对象损失


class v8ClassificationLoss:
    """Criterion class for computing training losses."""
    # 该类用于计算训练损失的标准类

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        # 计算预测结果与真实标签之间的分类损失
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds  # 解包预测结果
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")  # 计算交叉熵损失
        loss_items = loss.detach()  # 分离损失
        return loss, loss_items  # 返回损失和分离的损失


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""
    # 计算旋转YOLO模型中的目标检测、分类和边界框分布的损失

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        # 初始化v8OBBLoss，包含模型、分配器和旋转边界框损失；注意模型必须去并行化
        super().__init__(model)  # 调用父类构造函数
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)  # 初始化分配器
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)  # 初始化旋转边界框损失

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # 预处理目标计数并与输入批次大小匹配以输出张量
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)  # 如果没有目标，输出全零张量
        else:
            i = targets[:, 0]  # 获取图像索引
            _, counts = i.unique(return_counts=True)  # 计算每个图像的目标数量
            counts = counts.to(dtype=torch.int32)  # 转换为整型
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)  # 初始化输出张量
            for j in range(batch_size):
                matches = i == j  # 找到当前图像的匹配目标
                if n := matches.sum():  # 如果有匹配目标
                    bboxes = targets[matches, 2:]  # 获取匹配目标的边界框
                    bboxes[..., :4].mul_(scale_tensor)  # 根据缩放因子调整边界框
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)  # 填充输出张量
        return out  # 返回处理后的张量

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        # 计算并返回YOLO模型的损失
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # 初始化损失张量，包含框损失、类别损失和分布损失
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]  # 解包预测结果
        batch_size = pred_angle.shape[0]  # 获取批次大小
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # 合并预测分布和得分

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # 重新排列得分张量
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # 重新排列预测分布张量
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()  # 重新排列预测角度张量

        dtype = pred_scores.dtype  # 获取得分张量的数据类型
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # 计算图像尺寸（高度和宽度）
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # 创建锚点和步幅张量

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)  # 获取批次索引
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)  # 合并目标信息
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()  # 计算目标的宽高
            targets = targets[(rw >= 2) & (rh >= 2)]  # 过滤掉小尺寸的边界框以稳定训练
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # 预处理目标数据
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # 生成掩码
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e  # 捕获运行时错误并抛出类型错误，提示数据集格式不正确

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)
        # 解码预测边界框

        bboxes_for_assigner = pred_bboxes.clone().detach()  # 克隆预测边界框
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor  # 仅调整前四个元素
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )  # 进行目标分配

        target_scores_sum = max(target_scores.sum(), 1)  # 计算目标得分总和

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # 计算类别损失

        # Bbox loss
        if fg_mask.sum():  # 如果有前景掩码
            target_bboxes[..., :4] /= stride_tensor  # 将目标边界框除以步幅
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )  # 计算边界框损失
        else:
            loss[0] += (pred_angle * 0).sum()  # 如果没有前景掩码，避免产生无用梯度

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        # 根据超参数调整损失

        return loss.sum() * batch_size, loss.detach()  # 返回总损失和分离的损失

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            # 锚点，形状为(h*w, 2)
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            # 预测的旋转距离，形状为(bs, h*w, 4)
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
            # 预测的角度，形状为(bs, h*w, 1)

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
            # 返回带有角度的预测旋转边界框，形状为(bs, h*w, 5)
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)  # 返回拼接后的边界框和角度


class E2EDetectLoss:
    """Criterion class for computing training losses."""
    # 该类用于计算训练损失的标准类

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        # 使用提供的模型初始化E2EDetectLoss，包含一对多和一对一的检测损失
        self.one2many = v8DetectionLoss(model, tal_topk=10)  # 初始化一对多损失
        self.one2one = v8DetectionLoss(model, tal_topk=1)  # 初始化一对一损失

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # 计算框、类别和分布损失的总和并乘以批次大小
        preds = preds[1] if isinstance(preds, tuple) else preds  # 解包预测结果
        one2many = preds["one2many"]  # 获取一对多预测结果
        loss_one2many = self.one2many(one2many, batch)  # 计算一对多损失
        one2one = preds["one2one"]  # 获取一对一预测结果
        loss_one2one = self.one2one(one2one, batch)  # 计算一对一损失
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]  # 返回总损失