# Copyright (c) Facebook, Inc. and its affiliates.
# 导入必要的库
import math
from typing import List, Tuple  # 导入类型提示相关的工具
import torch
from fvcore.nn import giou_loss, smooth_l1_loss  # 导入损失函数
from torch.nn import functional as F

from detectron2.layers import cat, ciou_loss, diou_loss  # 导入自定义层和损失函数
from detectron2.structures import Boxes  # 导入边界框数据结构

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
# 用于限制大的dw和dh预测值的阈值。这个启发式规则确保dw和dh不会超过将16px的框变换到1000px的框所需的值
# （基于一个小的锚框16px和典型的图像尺寸1000px）
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


__all__ = ["Box2BoxTransform", "Box2BoxTransformRotated", "Box2BoxTransformLinear"]  # 定义模块的公共接口


@torch.jit.script  # 使用TorchScript进行即时编译，提高运行效率
class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    R-CNN中定义的边界框转换。该转换由4个偏移量参数化：(dx, dy, dw, dh)。
    转换通过exp(dw)和exp(dh)缩放框的宽度和高度，并通过偏移量(dx * width, dy * height)移动框的中心。
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        参数：
            weights (4元素元组)：应用于(dx, dy, dw, dh)偏移量的缩放因子。
                在Fast R-CNN中，这些值最初被设置为使偏移量具有单位方差；
                现在它们被视为系统的超参数。
            scale_clamp (float)：在预测偏移量时，预测的框缩放因子(dw和dh)
                被限制在不超过scale_clamp的范围内。
        """
        self.weights = weights  # 存储缩放权重
        self.scale_clamp = scale_clamp  # 存储缩放限制值

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        获取边界框回归变换的偏移量(dx, dy, dw, dh)，这些偏移量可用于将src_boxes
        转换为target_boxes。即满足关系：
        target_boxes == self.apply_deltas(deltas, src_boxes)
        （除非某些偏移量太大而被限制）

        参数：
            src_boxes (Tensor)：源边界框，例如目标建议框
            target_boxes (Tensor)：变换的目标框，例如真实标注框
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)  # 确保输入是张量类型
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        # 计算源框的宽度、高度和中心坐标
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]  # 宽度 = x2 - x1
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]  # 高度 = y2 - y1
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths  # 中心x坐标
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights  # 中心y坐标

        # 计算目标框的宽度、高度和中心坐标
        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        # 应用权重计算偏移量
        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths  # 中心点x方向的相对偏移
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights  # 中心点y方向的相对偏移
        dw = ww * torch.log(target_widths / src_widths)  # 宽度的对数比值
        dh = wh * torch.log(target_heights / src_heights)  # 高度的对数比值

        # 将所有偏移量堆叠成一个张量
        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        将变换偏移量`deltas` (dx, dy, dw, dh)应用到`boxes`上。

        参数：
            deltas (Tensor)：形状为(N, k*4)的变换偏移量，其中k >= 1。
                deltas[i]表示对单个框boxes[i]的k个可能不同的类别特定框变换。
            boxes (Tensor)：要变换的框，形状为(N, 4)
        """
        deltas = deltas.float()  # ensure fp32 for decoding precision，确保使用fp32进行解码以保证精度
        boxes = boxes.to(deltas.dtype)  # 将boxes转换为与deltas相同的数据类型

        # 计算源框的宽度、高度和中心坐标
        widths = boxes[:, 2] - boxes[:, 0]  # 宽度
        heights = boxes[:, 3] - boxes[:, 1]  # 高度
        ctr_x = boxes[:, 0] + 0.5 * widths  # 中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # 中心y坐标

        # 应用权重并计算偏移量
        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx  # x方向的偏移
        dy = deltas[:, 1::4] / wy  # y方向的偏移
        dw = deltas[:, 2::4] / ww  # 宽度的缩放
        dh = deltas[:, 3::4] / wh  # 高度的缩放

        # Prevent sending too large values into torch.exp()
        # 防止将过大的值输入到torch.exp()中
        dw = torch.clamp(dw, max=self.scale_clamp)  # 限制宽度缩放的最大值
        dh = torch.clamp(dh, max=self.scale_clamp)  # 限制高度缩放的最大值

        # 计算预测框的中心坐标和尺寸
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]  # 预测的中心x坐标
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]  # 预测的中心y坐标
        pred_w = torch.exp(dw) * widths[:, None]  # 预测的宽度
        pred_h = torch.exp(dh) * heights[:, None]  # 预测的高度

        # 计算预测框的四个角点坐标
        x1 = pred_ctr_x - 0.5 * pred_w  # 左上角x坐标
        y1 = pred_ctr_y - 0.5 * pred_h  # 左上角y坐标
        x2 = pred_ctr_x + 0.5 * pred_w  # 右下角x坐标
        y2 = pred_ctr_y + 0.5 * pred_h  # 右下角y坐标
        pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)  # 堆叠成预测框张量
        return pred_boxes.reshape(deltas.shape)  # 重塑为与输入相同的形状


@torch.jit.script
class Box2BoxTransformRotated(object):
    """
    The box-to-box transform defined in Rotated R-CNN. The transformation is parameterized
    by 5 deltas: (dx, dy, dw, dh, da). The transformation scales the box's width and height
    by exp(dw), exp(dh), shifts a box's center by the offset (dx * width, dy * height),
    and rotate a box's angle by da (radians).
    Note: angles of deltas are in radians while angles of boxes are in degrees.
    Rotated R-CNN中定义的边界框变换。该变换由5个偏移量参数化：(dx, dy, dw, dh, da)。
    变换通过exp(dw)和exp(dh)缩放框的宽度和高度，通过偏移量(dx * width, dy * height)移动框的中心，
    并通过da（弧度）旋转框的角度。
    注意：偏移量中的角度使用弧度制，而边界框中的角度使用角度制。
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float, float],
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
    ):
        """
        Args:
            weights (5-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh, da) deltas. These are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        参数：
            weights (5元素元组)：应用于(dx, dy, dw, dh, da)偏移量的缩放因子。
                这些被视为系统的超参数。
            scale_clamp (float)：在预测偏移量时，预测的框缩放因子(dw和dh)
                被限制在不超过scale_clamp的范围内。
        """
        self.weights = weights  # 存储缩放权重
        self.scale_clamp = scale_clamp  # 存储缩放限制值

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh, da) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): Nx5 source boxes, e.g., object proposals
            target_boxes (Tensor): Nx5 target of the transformation, e.g., ground-truth
                boxes.
        获取边界框回归变换的偏移量(dx, dy, dw, dh, da)，这些偏移量可用于将src_boxes
        转换为target_boxes。即满足关系：
        target_boxes == self.apply_deltas(deltas, src_boxes)
        （除非某些偏移量太大而被限制）

        参数：
            src_boxes (Tensor)：Nx5的源边界框，例如目标建议框
            target_boxes (Tensor)：Nx5的变换目标框，例如真实标注框
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)  # 确保输入是张量类型
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        # 解包源框的参数：中心坐标、宽度、高度和角度
        src_ctr_x, src_ctr_y, src_widths, src_heights, src_angles = torch.unbind(src_boxes, dim=1)

        # 解包目标框的参数
        target_ctr_x, target_ctr_y, target_widths, target_heights, target_angles = torch.unbind(
            target_boxes, dim=1
        )

        # 应用权重计算偏移量
        wx, wy, ww, wh, wa = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths  # 中心点x方向的相对偏移
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights  # 中心点y方向的相对偏移
        dw = ww * torch.log(target_widths / src_widths)  # 宽度的对数比值
        dh = wh * torch.log(target_heights / src_heights)  # 高度的对数比值
        # Angles of deltas are in radians while angles of boxes are in degrees.
        # the conversion to radians serve as a way to normalize the values
        # 偏移量中的角度使用弧度制，而边界框中的角度使用角度制。
        # 转换为弧度制作为归一化值的方式
        da = target_angles - src_angles  # 计算角度差
        da = (da + 180.0) % 360.0 - 180.0  # make it in [-180, 180)，将角度限制在[-180, 180)范围内
        da *= wa * math.pi / 180.0  # 转换为弧度并应用权重

        # 将所有偏移量堆叠成一个张量
        deltas = torch.stack((dx, dy, dw, dh, da), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransformRotated are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh, da) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*5).
                deltas[i] represents box transformation for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        将变换偏移量`deltas` (dx, dy, dw, dh, da)应用到`boxes`上。

        参数：
            deltas (Tensor)：形状为(N, k*5)的变换偏移量。
                deltas[i]表示对单个框boxes[i]的变换。
            boxes (Tensor)：要变换的框，形状为(N, 5)
        """
        assert deltas.shape[1] % 5 == 0 and boxes.shape[1] == 5  # 确保输入形状正确

        boxes = boxes.to(deltas.dtype).unsqueeze(2)  # 转换数据类型并增加维度

        # 解包边界框参数
        ctr_x = boxes[:, 0]  # 中心x坐标
        ctr_y = boxes[:, 1]  # 中心y坐标
        widths = boxes[:, 2]  # 宽度
        heights = boxes[:, 3]  # 高度
        angles = boxes[:, 4]  # 角度

        # 应用权重
        wx, wy, ww, wh, wa = self.weights

        # 计算归一化后的偏移量
        dx = deltas[:, 0::5] / wx  # x方向的偏移
        dy = deltas[:, 1::5] / wy  # y方向的偏移
        dw = deltas[:, 2::5] / ww  # 宽度的缩放
        dh = deltas[:, 3::5] / wh  # 高度的缩放
        da = deltas[:, 4::5] / wa  # 角度的偏移

        # Prevent sending too large values into torch.exp()
        # 防止将过大的值输入到torch.exp()中
        dw = torch.clamp(dw, max=self.scale_clamp)  # 限制宽度缩放的最大值
        dh = torch.clamp(dh, max=self.scale_clamp)  # 限制高度缩放的最大值

        # 计算预测框的参数
        pred_boxes = torch.zeros_like(deltas)  # 创建预测框张量
        pred_boxes[:, 0::5] = dx * widths + ctr_x  # x_ctr，预测的中心x坐标
        pred_boxes[:, 1::5] = dy * heights + ctr_y  # y_ctr，预测的中心y坐标
        pred_boxes[:, 2::5] = torch.exp(dw) * widths  # width，预测的宽度
        pred_boxes[:, 3::5] = torch.exp(dh) * heights  # height，预测的高度

        # Following original RRPN implementation,
        # angles of deltas are in radians while angles of boxes are in degrees.
        # 按照原始RRPN实现，偏移量中的角度使用弧度制，而边界框中的角度使用角度制
        pred_angle = da * 180.0 / math.pi + angles  # 将弧度转换为角度并加到原始角度上
        pred_angle = (pred_angle + 180.0) % 360.0 - 180.0  # make it in [-180, 180)，将角度限制在[-180, 180)范围内

        pred_boxes[:, 4::5] = pred_angle  # 设置预测框的角度

        return pred_boxes


class Box2BoxTransformLinear:
    """
    The linear box-to-box transform defined in FCOS. The transformation is parameterized
    by the distance from the center of (square) src box to 4 edges of the target box.
    FCOS中定义的线性边界框变换。该变换由源框（正方形）中心到目标框四个边的距离来参数化。
    """

    def __init__(self, normalize_by_size=True):
        """
        Args:
            normalize_by_size: normalize deltas by the size of src (anchor) boxes.
        参数：
            normalize_by_size：是否通过源框（锚框）的尺寸来归一化偏移量。
        """
        self.normalize_by_size = normalize_by_size  # 是否进行尺寸归一化

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx1, dy1, dx2, dy2) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true.
        The center of src must be inside target boxes.

        Args:
            src_boxes (Tensor): square source boxes, e.g., anchors
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        获取边界框回归变换的偏移量(dx1, dy1, dx2, dy2)，这些偏移量可用于将src_boxes
        转换为target_boxes。即满足关系：
        target_boxes == self.apply_deltas(deltas, src_boxes)
        源框的中心必须在目标框内部。

        参数：
            src_boxes (Tensor)：正方形源框，例如锚框
            target_boxes (Tensor)：变换的目标框，例如真实标注框
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)  # 确保输入是张量类型
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        # 计算源框的中心坐标
        src_ctr_x = 0.5 * (src_boxes[:, 0] + src_boxes[:, 2])  # 中心x坐标
        src_ctr_y = 0.5 * (src_boxes[:, 1] + src_boxes[:, 3])  # 中心y坐标

        # 计算源框中心到目标框四个边的距离
        target_l = src_ctr_x - target_boxes[:, 0]  # 到左边的距离
        target_t = src_ctr_y - target_boxes[:, 1]  # 到上边的距离
        target_r = target_boxes[:, 2] - src_ctr_x  # 到右边的距离
        target_b = target_boxes[:, 3] - src_ctr_y  # 到下边的距离

        # 将四个距离堆叠成偏移量
        deltas = torch.stack((target_l, target_t, target_r, target_b), dim=1)
        if self.normalize_by_size:  # 如果需要进行尺寸归一化
            stride = (src_boxes[:, 2] - src_boxes[:, 0]).unsqueeze(1)  # 计算源框的宽度（步长）
            deltas = deltas / stride  # 通过步长归一化偏移量
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx1, dy1, dx2, dy2) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        将变换偏移量`deltas` (dx1, dy1, dx2, dy2)应用到`boxes`上。

        参数：
            deltas (Tensor)：形状为(N, k*4)的变换偏移量，其中k >= 1。
                deltas[i]表示对单个框boxes[i]的k个可能不同的类别特定框变换。
            boxes (Tensor)：要变换的框，形状为(N, 4)
        """
        # Ensure the output is a valid box. See Sec 2.1 of https://arxiv.org/abs/2006.09214
        # 确保输出是有效的边界框。参见论文Sec 2.1
        deltas = F.relu(deltas)  # 使用ReLU确保偏移量非负
        boxes = boxes.to(deltas.dtype)  # 转换数据类型

        # 计算源框的中心坐标
        ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])  # 中心x坐标
        ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])  # 中心y坐标
        if self.normalize_by_size:  # 如果需要进行尺寸归一化
            stride = (boxes[:, 2] - boxes[:, 0]).unsqueeze(1)  # 计算源框的宽度（步长）
            deltas = deltas * stride  # 还原归一化的偏移量
        # 解包四个方向的偏移量
        l = deltas[:, 0::4]  # 左边偏移
        t = deltas[:, 1::4]  # 上边偏移
        r = deltas[:, 2::4]  # 右边偏移
        b = deltas[:, 3::4]  # 下边偏移

        # 计算预测框的坐标
        pred_boxes = torch.zeros_like(deltas)  # 创建预测框张量
        pred_boxes[:, 0::4] = ctr_x[:, None] - l  # x1，左上角x坐标
        pred_boxes[:, 1::4] = ctr_y[:, None] - t  # y1，左上角y坐标
        pred_boxes[:, 2::4] = ctr_x[:, None] + r  # x2，右下角x坐标
        pred_boxes[:, 3::4] = ctr_y[:, None] + b  # y2，右下角y坐标
        return pred_boxes


def _dense_box_regression_loss(
    anchors: List[Boxes],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_mask: torch.Tensor,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    计算密集多层级边界框回归的损失。
    损失在前景掩码``fg_mask``上累积。

    参数：
        anchors：每个层级的锚框，每个形状为(HixWixA, 4)
        pred_anchor_deltas：每个层级的预测，每个形状为(N, HixWixA, 4)
        gt_boxes：N个真实标注框，每个形状为(R, 4) (R = sum(Hi * Wi * A))
        fg_mask：形状为(N, R)的前景布尔掩码，用于计算损失
        box_reg_loss_type (str)：使用的损失类型。支持的损失："smooth_l1"、"giou"、
            "diou"、"ciou"。
        smooth_l1_beta (float)：smooth L1回归损失的beta参数。默认使用L1损失。
            仅在`box_reg_loss_type`为"smooth_l1"时使用
    """
    anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)，将所有层级的锚框合并
    if box_reg_loss_type == "smooth_l1":  # 如果使用smooth_l1损失
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]  # 计算真实偏移量
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)，堆叠真实偏移量
        loss_box_reg = smooth_l1_loss(  # 计算smooth L1损失
            cat(pred_anchor_deltas, dim=1)[fg_mask],  # 预测偏移量
            gt_anchor_deltas[fg_mask],  # 真实偏移量
            beta=smooth_l1_beta,  # beta参数
            reduction="sum",  # 使用求和归约
        )
    elif box_reg_loss_type == "giou":  # 如果使用GIoU损失
        pred_boxes = [  # 计算预测框
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = giou_loss(  # 计算GIoU损失
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
        )
    elif box_reg_loss_type == "diou":  # 如果使用DIoU损失
        pred_boxes = [  # 计算预测框
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = diou_loss(  # 计算DIoU损失
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
        )
    elif box_reg_loss_type == "ciou":  # 如果使用CIoU损失
        pred_boxes = [  # 计算预测框
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = ciou_loss(  # 计算CIoU损失
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
        )
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg
