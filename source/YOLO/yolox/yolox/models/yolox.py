#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn
# 导入 PyTorch 的 nn 模块，用于构建神经网络。

from .yolo_head import YOLOXHead
# 从当前包中导入 YOLOXHead 类，用于定义 YOLOX 的头部。

from .yolo_pafpn import YOLOPAFPN
# 从当前包中导入 YOLOPAFPN 类，用于定义 YOLOX 的骨干网络。

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    # YOLOX 模型模块。模块列表由 create_yolov3_modules 函数定义。
    # 该网络在训练期间返回来自三个 YOLO 层的损失值，在测试期间返回检测结果。

    def __init__(self, backbone=None, head=None):
        super().__init__()
        # 初始化 YOLOX 类，调用父类构造函数。

        if backbone is None:
            backbone = YOLOPAFPN()
        # 如果没有提供骨干网络，则使用 YOLOPAFPN 作为默认骨干网络。

        if head is None:
            head = YOLOXHead(80)
        # 如果没有提供头部，则使用 YOLOXHead 作为默认头部，类别数为 80。

        self.backbone = backbone
        # 将骨干网络赋值给实例变量。

        self.head = head
        # 将头部赋值给实例变量。

    def forward(self, x, targets=None):
        # 定义前向传播函数，接收输入 x 和目标 targets。

        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        # 通过骨干网络处理输入，获取特征图。

        if self.training:
            # 如果模型处于训练状态，则执行以下操作。

            assert targets is not None
            # 确保目标不为 None。

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            # 调用头部，计算损失值，包括总损失、IoU 损失、置信度损失、类别损失、L1 损失和前景数量。

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            # 将损失值存储在字典中。

        else:
            outputs = self.head(fpn_outs)
            # 如果模型不在训练状态，则仅获取检测结果。

        return outputs
        # 返回输出结果。

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        # 定义可视化函数，接收输入 x、目标和保存前缀。

        fpn_outs = self.backbone(x)
        # 通过骨干网络处理输入，获取特征图。

        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
        # 调用头部的可视化函数，绘制分配结果并保存。