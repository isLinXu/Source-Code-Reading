#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math  # 导入math模块，用于数学计算
import torch  # 导入PyTorch库


class IOUloss:
    """ Calculate IoU loss.
    """
    # 计算IoU损失的类

    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        # 类的初始化设置
        self.box_format = box_format  # 设置框的格式
        self.iou_type = iou_type.lower()  # 设置IoU类型并转换为小写
        self.reduction = reduction  # 设置损失的归约方式
        self.eps = eps  # 设置防止除以零的值

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [N, 4].
        """
        # 计算IoU，box1和box2是形状为[M, 4]和[N, 4]的torch张量
        if box1.shape[0] != box2.shape[0]:  # 如果两个框的数量不相等
            box2 = box2.T  # 转置box2
            if self.box_format == 'xyxy':  # 如果框格式为xyxy
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]  # 解包box1的坐标
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]  # 解包box2的坐标
            elif self.box_format == 'xywh':  # 如果框格式为xywh
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2  # 计算box1的左上和右下角坐标
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2  # 计算box2的左上和右下角坐标
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:  # 如果两个框的数量相等
            if self.box_format == 'xyxy':  # 如果框格式为xyxy
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)  # 拆分box1的坐标
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)  # 拆分box2的坐标
            elif self.box_format == 'xywh':  # 如果框格式为xywh
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)  # 拆分box1的坐标和宽高
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)  # 拆分box2的坐标和宽高
                b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2  # 计算box1的左上和右下角坐标
                b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
                b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2  # 计算box2的左上和右下角坐标
                b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # 计算交集面积
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # 计算并集面积
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps  # box1的宽和高
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps  # box2的宽和高
        union = w1 * h1 + w2 * h2 - inter + self.eps  # 计算并集面积
        iou = inter / union  # 计算IoU

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 计算凸包的宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 计算凸包的高度
        if self.iou_type == 'giou':  # 如果IoU类型为GIoU
            c_area = cw * ch + self.eps  # 计算凸包的面积
            iou = iou - (c_area - union) / c_area  # 更新IoU值
        elif self.iou_type in ['diou', 'ciou']:  # 如果IoU类型为DIoU或CIoU
            c2 = cw ** 2 + ch ** 2 + self.eps  # 计算凸包的对角线平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 计算中心距离平方
            if self.iou_type == 'diou':  # 如果IoU类型为DIoU
                iou = iou - rho2 / c2  # 更新IoU值
            elif self.iou_type == 'ciou':  # 如果IoU类型为CIoU
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)  # 计算角度损失
                with torch.no_grad():  # 禁用梯度计算
                    alpha = v / (v - iou + (1 + self.eps))  # 计算平衡因子
                iou = iou - (rho2 / c2 + v * alpha)  # 更新IoU值
        elif self.iou_type == 'siou':  # 如果IoU类型为SIoU
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps  # 计算宽度差的一半
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps  # 计算高度差的一半
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)  # 计算sigma
            sin_alpha_1 = torch.abs(s_cw) / sigma  # 计算角度的正弦值
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2  # 阈值
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)  # 选择较大的正弦值
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)  # 计算角度成本
            rho_x = (s_cw / cw) ** 2  # 计算宽度的比例
            rho_y = (s_ch / ch) ** 2  # 计算高度的比例
            gamma = angle_cost - 2  # 计算gamma
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)  # 计算距离成本
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # 计算宽度的相对差异
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # 计算高度的相对差异
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)  # 计算形状成本
            iou = iou - 0.5 * (distance_cost + shape_cost)  # 更新IoU值
        loss = 1.0 - iou  # 计算损失

        if self.reduction == 'sum':  # 如果归约方式为sum
            loss = loss.sum()  # 计算总损失
        elif self.reduction == 'mean':  # 如果归约方式为mean
            loss = loss.mean()  # 计算平均损失

        return loss  # 返回损失


def pairwise_bbox_iou(box1, box2, box_format='xywh'):
    """Calculate iou.
    This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
    """
    # 计算IoU
    if box_format == 'xyxy':  # 如果框格式为xyxy
        lt = torch.max(box1[:, None, :2], box2[:, :2])  # 计算左上角坐标
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # 计算右下角坐标
        area_1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)  # 计算box1的面积
        area_2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)  # 计算box2的面积

    elif box_format == 'xywh':  # 如果框格式为xywh
        lt = torch.max(
            (box1[:, None, :2] - box1[:, None, 2:] / 2),  # 计算左上角坐标
            (box2[:, :2] - box2[:, 2:] / 2),
        )
        rb = torch.min(
            (box1[:, None, :2] + box1[:, None, 2:] / 2),  # 计算右下角坐标
            (box2[:, :2] + box2[:, 2:] / 2),
        )

        area_1 = torch.prod(box1[:, 2:], 1)  # 计算box1的面积
        area_2 = torch.prod(box2[:, 2:], 1)  # 计算box2的面积
    valid = (lt < rb).type(lt.type()).prod(dim=2)  # 判断有效框
    inter = torch.prod(rb - lt, 2) * valid  # 计算交集面积
    return inter / (area_1[:, None] + area_2 - inter)  # 返回IoU