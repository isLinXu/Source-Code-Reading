# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

from . import LOGGER  # 导入日志记录器
from .checks import check_version  # 导入版本检查函数
from .metrics import bbox_iou, probiou  # 导入 IoU 计算函数
from .ops import xywhr2xyxyxyxy  # 导入坐标转换函数

TORCH_1_10 = check_version(torch.__version__, "1.10.0")  # 检查 PyTorch 版本是否为 1.10.0


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.
    用于目标检测的任务对齐分配器。

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.
    该类根据任务对齐度量将真实（gt）对象分配给锚点，该度量结合了分类和定位信息。

    Attributes:
        topk (int): The number of top candidates to consider.
        topk（整数）：要考虑的最佳候选者数量。
        num_classes (int): The number of object classes.
        num_classes（整数）：对象类别的数量。
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        alpha（浮动）：任务对齐度量中分类组件的 alpha 参数。
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        beta（浮动）：任务对齐度量中定位组件的 beta 参数。
        eps (float): A small value to prevent division by zero.
        eps（浮动）：防止除零的一个小值。
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()  # 调用父类构造函数
        self.topk = topk  # 设置最佳候选者数量
        self.num_classes = num_classes  # 设置对象类别数量
        self.bg_idx = num_classes  # 设置背景索引
        self.alpha = alpha  # 设置 alpha 参数
        self.beta = beta  # 设置 beta 参数
        self.eps = eps  # 设置小值以防止除零

    @torch.no_grad()  # 在前向传播时不计算梯度
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.
        计算任务对齐的分配。参考代码可在 https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py 获取。

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_scores（张量）：形状（bs，总锚点数量，类别数量）
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            pd_bboxes（张量）：形状（bs，总锚点数量，4）
            anc_points (Tensor): shape(num_total_anchors, 2)
            anc_points（张量）：形状（总锚点数量，2）
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_labels（张量）：形状（bs，最大框数，1）
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_bboxes（张量）：形状（bs，最大框数，4）
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            mask_gt（张量）：形状（bs，最大框数，1）

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_labels（张量）：形状（bs，总锚点数量）
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_bboxes（张量）：形状（bs，总锚点数量，4）
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            target_scores（张量）：形状（bs，总锚点数量，类别数量）
            fg_mask (Tensor): shape(bs, num_total_anchors)
            fg_mask（张量）：形状（bs，总锚点数量）
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
            target_gt_idx（张量）：形状（bs，总锚点数量）
        """
        self.bs = pd_scores.shape[0]  # 获取批次大小
        self.n_max_boxes = gt_bboxes.shape[1]  # 获取最大框数
        device = gt_bboxes.device  # 获取设备信息

        if self.n_max_boxes == 0:  # 如果没有真实框
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),  # 返回背景索引
                torch.zeros_like(pd_bboxes),  # 返回零的边界框
                torch.zeros_like(pd_scores),  # 返回零的分数
                torch.zeros_like(pd_scores[..., 0]),  # 返回零的前景掩码
                torch.zeros_like(pd_scores[..., 0]),  # 返回零的目标索引
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)  # 调用私有前向函数
        except torch.OutOfMemoryError:  # 如果出现内存不足错误
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")  # 记录警告信息
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]  # 将张量移动到 CPU
            result = self._forward(*cpu_tensors)  # 在 CPU 上计算结果
            return tuple(t.to(device) for t in result)  # 将结果移动回原设备并返回

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_scores（张量）：形状（bs，总锚点数量，类别数量）
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            pd_bboxes（张量）：形状（bs，总锚点数量，4）
            anc_points (Tensor): shape(num_total_anchors, 2)
            anc_points（张量）：形状（总锚点数量，2）
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_labels（张量）：形状（bs，最大框数，1）
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_bboxes（张量）：形状（bs，最大框数，4）
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            mask_gt（张量）：形状（bs，最大框数，1）

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_labels（张量）：形状（bs，总锚点数量）
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_bboxes（张量）：形状（bs，总锚点数量，4）
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            target_scores（张量）：形状（bs，总锚点数量，类别数量）
            fg_mask (Tensor): shape(bs, num_total_anchors)
            fg_mask（张量）：形状（bs，总锚点数量）
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
            target_gt_idx（张量）：形状（bs，总锚点数量）
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(  # 获取正样本掩码、对齐度量和重叠度
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)  # 选择最高重叠的目标

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)  # 获取目标标签、边界框和分数

        # Normalize
        align_metric *= mask_pos  # 对齐度量乘以正样本掩码
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj 获取正样本对齐度量的最大值
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj 获取正样本重叠度的最大值
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)  # 归一化对齐度量
        target_scores = target_scores * norm_align_metric  # 更新目标分数

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx  # 返回目标标签、边界框、分数、前景掩码和目标索引

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)  # 获取锚点在真实框内的掩码
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)  # 获取锚点对齐度量和重叠度
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())  # 选择 top-k 候选者的掩码
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt  # 合并所有掩码为最终掩码

        return mask_pos, align_metric, overlaps  # 返回正样本掩码、对齐度量和重叠度

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]  # 获取锚点数量
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w 将掩码转换为布尔类型
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)  # 初始化重叠度张量
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)  # 初始化边界框分数张量

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj 初始化索引张量
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj 扩展批次索引
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj 获取真实标签索引
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w 获取每个真实类别的边界框分数

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]  # 扩展预测边界框
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]  # 扩展真实边界框
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)  # 计算重叠度

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)  # 计算对齐度量
        return align_metric, overlaps  # 返回对齐度量和重叠度

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)  # 计算 IoU 并进行压缩和限制

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.
        根据给定的度量选择 top-k 候选者。

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            metrics（张量）：形状为（b，最大对象数，h*w）的张量，其中 b 是批次大小，
                              max_num_obj 是最大对象数，h*w 表示总锚点数量。
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            largest（布尔值）：如果为 True，则选择最大的值；否则选择最小的值。
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.
            topk_mask（张量）：可选的布尔张量，形状为（b，最大对象数，topk），其中
                                topk 是要考虑的最佳候选者数量。如果未提供，
                                则根据给定的度量自动计算 top-k 值。

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
            返回形状为（b，最大对象数，h*w）的张量，包含所选的 top-k 候选者。
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)  # 获取 top-k 候选者的度量和索引
        if topk_mask is None:  # 如果未提供 topk_mask
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)  # 计算 top-k 值的掩码
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)  # 用 0 填充无效索引

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)  # 初始化计数张量
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)  # 创建全为 1 的张量
        for k in range(self.topk):  # 遍历每个 top-k 候选者
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)  # 在指定位置添加 1
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)  # 将计数大于 1 的位置填充为 0

        return count_tensor.to(metrics.dtype)  # 返回计数张量

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.
        计算正锚点的目标标签、目标边界框和目标分数。

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_labels（张量）：形状为（b，最大对象数，1）的真实标签，其中 b 是批次大小，
                                max_num_obj 是最大对象数。
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            gt_bboxes（张量）：形状为（b，最大对象数，4）的真实边界框。
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            target_gt_idx（张量）：正锚点分配的真实对象的索引，形状为（b，h*w），其中 h*w 是总
                                    锚点数量。
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.
            fg_mask（张量）：形状为（b，h*w）的布尔张量，指示正锚点（前景）。

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                target_labels（张量）：形状为（b，h*w），包含正锚点的目标标签。
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                target_bboxes（张量）：形状为（b，h*w，4），包含正锚点的目标边界框。
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
                target_scores（张量）：形状为（b，h*w，类别数量），包含正锚点的目标分数，其中类别数量是对象类别的数量。
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]  # 创建批次索引
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w) 更新目标索引
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w) 获取目标标签

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]  # 获取目标边界框

        # Assigned target scores
        target_labels.clamp_(0)  # 限制目标标签的值

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),  # (b, h*w, 80)
            dtype=torch.int64,
            device=target_labels.device,
        )  # 初始化目标分数张量
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)  # 将目标标签的分数设置为 1

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80) 创建前景分数掩码
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)  # 更新目标分数

        return target_labels, target_bboxes, target_scores  # 返回目标标签、边界框和分数

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.
        选择真实边界框内的正锚点中心。

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            xy_centers（张量）：锚点中心坐标，形状为（h*w，2）。
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            gt_bboxes（张量）：真实边界框，形状为（b，n_boxes，4）。
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.
            eps（浮动）：用于数值稳定性的小值。默认为 1e-9。

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).
            返回正锚点的布尔掩码，形状为（b，n_boxes，h*w）。

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            b：批次大小，n_boxes：真实框数量，h：高度，w：宽度。
            Bounding box format: [x_min, y_min, x_max, y_max].
            边界框格式：[x_min，y_min，x_max，y_max]。
        """
        n_anchors = xy_centers.shape[0]  # 获取锚点数量
        bs, n_boxes, _ = gt_bboxes.shape  # 获取批次大小和真实框数量
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom 获取左上角和右下角坐标
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)  # 计算锚点与真实框的距离
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)  # 返回正锚点的布尔掩码

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.
        选择在分配给多个真实框时具有最高 IoU 的锚框。

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            mask_pos（张量）：正掩码，形状为（b，n_max_boxes，h*w）。
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            overlaps（张量）：IoU 重叠度，形状为（b，n_max_boxes，h*w）。
            n_max_boxes (int): Maximum number of ground truth boxes.
            n_max_boxes（整数）：最大真实框数量。

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            target_gt_idx（张量）：分配的真实对象的索引，形状为（b，h*w）。
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            fg_mask（张量）：前景掩码，形状为（b，h*w）。
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
            mask_pos（张量）：更新后的正掩码，形状为（b，n_max_boxes，h*w）。

        Note:
            b: batch size, h: height, w: width.
            b：批次大小，h：高度，w：宽度。
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)  # 计算前景掩码
        if fg_mask.max() > 1:  # 如果一个锚点分配给多个真实框
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # 创建多重真实框的掩码
            max_overlaps_idx = overlaps.argmax(1)  # 获取最大重叠的索引

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # 初始化最大重叠掩码
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)  # 在最大重叠位置填充 1

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # 更新正掩码
            fg_mask = mask_pos.sum(-2)  # 重新计算前景掩码
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w) 获取目标真实框的索引
        return target_gt_idx, fg_mask, mask_pos  # 返回目标索引、前景掩码和正掩码


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""
    # 用于将真实对象分配给旋转边界框的任务对齐分配器。

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)  # 计算旋转边界框的 IoU

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.
        为旋转边界框选择真实框中的正锚点中心。

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            xy_centers（张量）：形状为（h*w，2）
            gt_bboxes (Tensor): shape(b, n_boxes, 5)
            gt_bboxes（张量）：形状为（b，n_boxes，5）

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
            返回形状为（b，n_boxes，h*w）的张量
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)  # 将真实框转换为四个角坐标
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)  # 获取四个角的坐标
        ab = b - a  # 计算边 AB
        ad = d - a  # 计算边 AD

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a  # 计算锚点与角 A 的距离
        norm_ab = (ab * ab).sum(dim=-1)  # 计算边 AB 的平方长度
        norm_ad = (ad * ad).sum(dim=-1)  # 计算边 AD 的平方长度
        ap_dot_ab = (ap * ab).sum(dim=-1)  # 计算锚点与边 AB 的点积
        ap_dot_ad = (ap * ad).sum(dim=-1)  # 计算锚点与边 AD 的点积
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # 判断锚点是否在框内


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []  # 初始化锚点和步幅张量
    assert feats is not None  # 确保特征不为空
    dtype, device = feats[0].dtype, feats[0].device  # 获取数据类型和设备信息
    for i, stride in enumerate(strides):  # 遍历每个步幅
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))  # 获取高度和宽度
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x 移动 x 坐标
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y 移动 y 坐标
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)  # 创建网格
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))  # 将 x 和 y 坐标堆叠并调整形状
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))  # 创建步幅张量
    return torch.cat(anchor_points), torch.cat(stride_tensor)  # 返回拼接后的锚点和步幅张量


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)  # 将距离拆分为左上角和右下角
    x1y1 = anchor_points - lt  # 计算左上角坐标
    x2y2 = anchor_points + rb  # 计算右下角坐标
    if xywh:  # 如果需要 xywh 格式
        c_xy = (x1y1 + x2y2) / 2  # 计算中心坐标
        wh = x2y2 - x1y1  # 计算宽高
        return torch.cat((c_xy, wh), dim)  # 返回 xywh 格式的边界框
    return torch.cat((x1y1, x2y2), dim)  # 返回 xyxy 格式的边界框


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)  # 将边界框拆分为左上角和右下角
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # 计算距离并限制在最大值范围内


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted rotated bounding box coordinates from anchor points and distribution.
    从锚点和分布中解码预测的旋转边界框坐标。

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, shape (bs, h*w, 4).
        pred_dist（张量）：预测的旋转距离，形状为（bs，h*w，4）。
        pred_angle (torch.Tensor): Predicted angle, shape (bs, h*w, 1).
        pred_angle（张量）：预测的角度，形状为（bs，h*w，1）。
        anchor_points (torch.Tensor): Anchor points, shape (h*w, 2).
        anchor_points（张量）：锚点，形状为（h*w，2）。
        dim (int, optional): Dimension along which to split. Defaults to -1.
        dim（整数，可选）：拆分的维度。默认为 -1。

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, shape (bs, h*w, 4).
        返回预测的旋转边界框，形状为（bs，h*w，4）。
    """
    lt, rb = pred_dist.split(2, dim=dim)  # 拆分预测的距离
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)  # 计算余弦和正弦值
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)  # 计算边界框的宽和高
    x, y = xf * cos - yf * sin, xf * sin + yf * cos  # 计算旋转后的坐标
    xy = torch.cat([x, y], dim=dim) + anchor_points  # 计算最终坐标
    return torch.cat([xy, lt + rb], dim=dim)  # 返回旋转边界框