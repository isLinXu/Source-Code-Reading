#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()  # 调用父类的初始化方法
        self.reduction = reduction  # 设置损失的归约方式
        self.loss_type = loss_type  # 设置损失类型（IoU或GIoU）

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]  # 确保预测和目标的批次大小相同

        pred = pred.view(-1, 4)  # 将预测结果重塑为(-1, 4)的形状
        target = target.view(-1, 4)  # 将目标结果重塑为(-1, 4)的形状
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )  # 计算左上角坐标
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )  # 计算右下角坐标

        area_p = torch.prod(pred[:, 2:], 1)  # 计算预测框的面积
        area_g = torch.prod(target[:, 2:], 1)  # 计算目标框的面积

        en = (tl < br).type(tl.type()).prod(dim=1)  # 判断是否有重叠区域
        area_i = torch.prod(br - tl, 1) * en  # 计算交集的面积
        area_u = area_p + area_g - area_i  # 计算并集的面积
        iou = (area_i) / (area_u + 1e-16)  # 计算IoU，加入小常数以避免除零错误

        if self.loss_type == "iou":
            loss = 1 - iou ** 2  # 计算IoU损失
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )  # 计算包围框的左上角坐标
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )  # 计算包围框的右下角坐标
            area_c = torch.prod(c_br - c_tl, 1)  # 计算包围框的面积
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)  # 计算GIoU
            loss = 1 - giou.clamp(min=-1.0, max=1.0)  # 计算GIoU损失

        if self.reduction == "mean":
            loss = loss.mean()  # 如果选择均值归约，计算损失的均值
        elif self.reduction == "sum":
            loss = loss.sum()  # 如果选择求和归约，计算损失的总和

        return loss  # 返回计算得到的损失