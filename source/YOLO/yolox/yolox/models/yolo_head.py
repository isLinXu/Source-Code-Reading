#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
# 导入数学模块，用于数学计算。

from loguru import logger
# 从 loguru 模块导入 logger，用于日志记录。

import torch
# 导入 PyTorch 库，用于深度学习模型的构建和训练。

import torch.nn as nn
# 导入 PyTorch 的 nn 模块，用于构建神经网络。

import torch.nn.functional as F
# 导入 PyTorch 的功能性模块，用于实现各种神经网络操作。

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign
# 从 yolox.utils 模块导入各种实用函数，包括计算框的 IoU、坐标转换、网格生成和可视化分配结果的函数。

from .losses import IOUloss
# 从当前包中导入 IOUloss 类，用于计算 IoU 损失。

from .network_blocks import BaseConv, DWConv
# 从当前包中导入 BaseConv 和 DWConv 类，用于构建卷积层。

class YOLOXHead(nn.Module):
    # 定义 YOLOXHead 类，继承自 nn.Module。

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        # 初始化 YOLOXHead 类，定义构造函数，接收多个参数。
        
        super().__init__()
        # 调用父类构造函数。

        self.num_classes = num_classes
        # 设置类别数量。

        self.decode_in_inference = True  # for deploy, set to False
        # 设置是否在推理阶段解码，默认为 True。

        self.cls_convs = nn.ModuleList()
        # 初始化类别卷积层的模块列表。

        self.reg_convs = nn.ModuleList()
        # 初始化回归卷积层的模块列表。

        self.cls_preds = nn.ModuleList()
        # 初始化类别预测层的模块列表。

        self.reg_preds = nn.ModuleList()
        # 初始化回归预测层的模块列表。

        self.obj_preds = nn.ModuleList()
        # 初始化目标预测层的模块列表。

        self.stems = nn.ModuleList()
        # 初始化主干网络的模块列表。

        Conv = DWConv if depthwise else BaseConv
        # 根据是否使用深度卷积选择卷积层类型。

        for i in range(len(in_channels)):
            # 遍历输入通道列表，构建网络层。

            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            # 添加主干卷积层，输入通道数根据宽度调整，输出通道数为 256。

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            # 添加类别卷积层，包含两个卷积层，每个卷积层的输入和输出通道数均为 256。

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            # 添加回归卷积层，包含两个卷积层，结构与类别卷积层相同。

            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # 添加类别预测层，输出通道数为类别数量。

            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # 添加回归预测层，输出通道数为 4（表示边界框的坐标）。

            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # 添加目标预测层，输出通道数为 1（表示目标存在的置信度）。

        self.use_l1 = False
        # 设置是否使用 L1 损失，默认为 False。

        self.l1_loss = nn.L1Loss(reduction="none")
        # 初始化 L1 损失函数。

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        # 初始化二元交叉熵损失函数。

        self.iou_loss = IOUloss(reduction="none")
        # 初始化 IoU 损失函数。

        self.strides = strides
        # 设置每个层的步幅。

        self.grids = [torch.zeros(1)] * len(in_channels)
        # 初始化网格，长度与输入通道数量相同，初始值为零。

    def initialize_biases(self, prior_prob):
        # 定义初始化偏置的函数，接收先验概率作为参数。
        
        for conv in self.cls_preds:
            # 遍历类别预测层的卷积层。
            
            b = conv.bias.view(1, -1)
            # 将偏置转换为一维数组。
            
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            # 根据先验概率计算偏置值，并填充到偏置中。
            
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            # 将偏置重新设置为可训练的参数。
    
        for conv in self.obj_preds:
            # 遍历目标预测层的卷积层。
            
            b = conv.bias.view(1, -1)
            # 将偏置转换为一维数组。
            
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            # 根据先验概率计算偏置值，并填充到偏置中。
            
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            # 将偏置重新设置为可训练的参数。
    
    def forward(self, xin, labels=None, imgs=None):
        # 定义前向传播函数，接收输入 xin、标签 labels 和图像 imgs。
    
        outputs = []
        # 初始化输出列表，用于存储每个层的输出。
    
        origin_preds = []
        # 初始化原始预测列表，用于存储回归预测。
    
        x_shifts = []
        # 初始化 x 轴偏移列表。
    
        y_shifts = []
        # 初始化 y 轴偏移列表。
    
        expanded_strides = []
        # 初始化扩展步幅列表。
    
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            # 遍历类别卷积层、回归卷积层、当前层步幅和输入，构建每个层的输出。
    
            x = self.stems[k](x)
            # 通过主干卷积层处理输入。
    
            cls_x = x
            # 将处理后的输入赋值给类别特征。
    
            reg_x = x
            # 将处理后的输入赋值给回归特征。
    
            cls_feat = cls_conv(cls_x)
            # 通过类别卷积层获取类别特征。
    
            cls_output = self.cls_preds[k](cls_feat)
            # 通过类别预测层获取类别输出。
    
            reg_feat = reg_conv(reg_x)
            # 通过回归卷积层获取回归特征。
    
            reg_output = self.reg_preds[k](reg_feat)
            # 通过回归预测层获取回归输出。
    
            obj_output = self.obj_preds[k](reg_feat)
            # 通过目标预测层获取目标输出。
    
            if self.training:
                # 如果模型处于训练状态，则执行以下操作。
    
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                # 将回归输出、目标输出和类别输出按通道拼接。
    
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                # 获取输出和网格信息。
    
                x_shifts.append(grid[:, :, 0])
                # 将网格的 x 轴偏移添加到列表中。
    
                y_shifts.append(grid[:, :, 1])
                # 将网格的 y 轴偏移添加到列表中。
    
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                # 将当前层的步幅添加到列表中。
    
                if self.use_l1:
                    # 如果使用 L1 损失，则执行以下操作。
    
                    batch_size = reg_output.shape[0]
                    # 获取当前批次的大小。
    
                    hsize, wsize = reg_output.shape[-2:]
                    # 获取回归输出的高度和宽度。
    
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    # 将回归输出重塑为适合计算的形状。
    
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    # 调整输出的维度顺序并重塑为二维数组。
    
                    origin_preds.append(reg_output.clone())
                    # 将当前的回归输出克隆并添加到原始预测列表中。
    
            else:
                # 如果模型不在训练状态，则执行以下操作。
    
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                # 将回归输出、目标输出和类别输出按通道拼接，并对目标和类别输出应用 sigmoid 激活函数。
    
            outputs.append(output)
            # 将当前层的输出添加到输出列表中。
    
        if self.training:
            # 如果模型处于训练状态，则返回损失。
    
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            # 如果模型不在训练状态，则执行以下操作。
    
            self.hw = [x.shape[-2:] for x in outputs]
            # 获取每个输出的高度和宽度。
    
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            # 将所有输出按通道拼接并调整维度顺序。
    
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
                # 如果在推理阶段解码，则返回解码后的输出。
    
            else:
                return outputs
                # 否则，返回输出。

    def get_output_and_grid(self, output, k, stride, dtype):
        # 定义获取输出和网格的函数，接收输出、索引 k、步幅和数据类型。
    
        grid = self.grids[k]
        # 获取当前层的网格。
    
        batch_size = output.shape[0]
        # 获取当前批次的大小。
    
        n_ch = 5 + self.num_classes
        # 计算输出的通道数，包括 4 个边界框坐标和 1 个目标置信度，加上类别数量。
    
        hsize, wsize = output.shape[-2:]
        # 获取输出的高度和宽度。
    
        if grid.shape[2:4] != output.shape[2:4]:
            # 如果网格的高度和宽度与输出不匹配，则执行以下操作。
    
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # 生成网格坐标。
    
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            # 创建网格，并调整形状为 (1, 1, hsize, wsize, 2)。
    
            self.grids[k] = grid
            # 更新当前层的网格。
    
        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        # 将输出重塑为 (batch_size, 1, n_ch, hsize, wsize) 的形状。
    
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        # 调整输出的维度顺序并重塑为 (batch_size, hsize * wsize, -1) 的形状。
    
        grid = grid.view(1, -1, 2)
        # 将网格重塑为 (1, -1, 2) 的形状。
    
        output[..., :2] = (output[..., :2] + grid) * stride
        # 更新输出的前两个通道（边界框中心坐标），将其转换为实际坐标。
    
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        # 更新输出的第三和第四个通道（边界框宽度和高度），使用指数函数将其转换为实际尺寸。
    
        return output, grid
        # 返回输出和网格。
    
    def decode_outputs(self, outputs, dtype):
        # 定义解码输出的函数，接收输出和数据类型。
    
        grids = []
        # 初始化网格列表。
    
        strides = []
        # 初始化步幅列表。
    
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            # 遍历每个输出的高度、宽度和对应的步幅。
    
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # 生成网格坐标。
    
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            # 创建网格，并调整形状为 (1, -1, 2)。
    
            grids.append(grid)
            # 将网格添加到网格列表中。
    
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
            # 创建与网格相同形状的步幅张量，并添加到步幅列表中。
    
        grids = torch.cat(grids, dim=1).type(dtype)
        # 将所有网格按列拼接，并转换为指定的数据类型。
    
        strides = torch.cat(strides, dim=1).type(dtype)
        # 将所有步幅按列拼接，并转换为指定的数据类型。
    
        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        # 将输出的边界框坐标、宽度和高度与网格和步幅结合，生成最终的输出。
    
        return outputs
        # 返回解码后的输出。


    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
        ):
        # 定义计算损失的函数，接收图像、偏移量、步幅、标签、输出、原始预测和数据类型。
    
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        # 提取边界框预测，形状为 [batch, n_anchors_all, 4]。
    
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        # 提取目标置信度预测，形状为 [batch, n_anchors_all, 1]。
    
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]
        # 提取类别预测，形状为 [batch, n_anchors_all, n_cls]。
    
        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        # 计算每个批次中目标的数量。
    
        total_num_anchors = outputs.shape[1]
        # 获取总的锚框数量。
    
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        # 将 x 轴偏移量拼接为一个张量。
    
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        # 将 y 轴偏移量拼接为一个张量。
    
        expanded_strides = torch.cat(expanded_strides, 1)
        # 将扩展步幅拼接为一个张量。
    
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        # 如果使用 L1 损失，则将原始预测拼接为一个张量。
    
        cls_targets = []
        # 初始化类别目标列表。
    
        reg_targets = []
        # 初始化回归目标列表。
    
        l1_targets = []
        # 初始化 L1 目标列表。
    
        obj_targets = []
        # 初始化目标置信度目标列表。
    
        fg_masks = []
        # 初始化前景掩码列表。
    
        num_fg = 0.0
        # 初始化前景数量。
    
        num_gts = 0.0
        # 初始化真实目标数量。
    
        for batch_idx in range(outputs.shape[0]):
            # 遍历每个批次的输出。
    
            num_gt = int(nlabel[batch_idx])
            # 获取当前批次的真实目标数量。
    
            num_gts += num_gt
            # 更新总的真实目标数量。
    
            if num_gt == 0:
                # 如果当前批次没有真实目标，则执行以下操作。
    
                cls_target = outputs.new_zeros((0, self.num_classes))
                # 创建空的类别目标。
    
                reg_target = outputs.new_zeros((0, 4))
                # 创建空的回归目标。
    
                l1_target = outputs.new_zeros((0, 4))
                # 创建空的 L1 目标。
    
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                # 创建全零的目标置信度目标。
    
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                # 创建全零的前景掩码。
    
            else:
                # 如果当前批次有真实目标，则执行以下操作。
    
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                # 获取当前批次的真实边界框。
    
                gt_classes = labels[batch_idx, :num_gt, 0]
                # 获取当前批次的真实类别。
    
                bboxes_preds_per_image = bbox_preds[batch_idx]
                # 获取当前批次的边界框预测。
    
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                    # 调用 get_assignments 函数进行目标匹配。
    
                except RuntimeError as e:
                    # 捕获运行时错误。
    
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # 如果不是由于 CUDA 内存不足导致的错误，则抛出异常。
    
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    # 记录错误信息，提示可能的内存不足问题。
    
                    torch.cuda.empty_cache()
                    # 清空 CUDA 缓存。
    
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )
                    # 在 CPU 模式下重新调用 get_assignments 函数进行目标匹配。
    
                torch.cuda.empty_cache()
                # 清空 CUDA 缓存。
    
                num_fg += num_fg_img
                # 更新前景数量。
    
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                # 创建类别目标，使用 one-hot 编码并乘以预测 IoU。
    
                obj_target = fg_mask.unsqueeze(-1)
                # 创建目标置信度目标。
    
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                # 获取回归目标。
    
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
                    # 如果使用 L1 损失，则计算 L1 目标。
    
            cls_targets.append(cls_target)
            # 将类别目标添加到列表中。
    
            reg_targets.append(reg_target)
            # 将回归目标添加到列表中。
    
            obj_targets.append(obj_target.to(dtype))
            # 将目标置信度目标添加到列表中，并转换为指定的数据类型。
    
            fg_masks.append(fg_mask)
            # 将前景掩码添加到列表中。
    
            if self.use_l1:
                l1_targets.append(l1_target)
                # 如果使用 L1 损失，则将 L1 目标添加到列表中。
    
        cls_targets = torch.cat(cls_targets, 0)
        # 将所有类别目标拼接为一个张量。
    
        reg_targets = torch.cat(reg_targets, 0)
        # 将所有回归目标拼接为一个张量。
    
        obj_targets = torch.cat(obj_targets, 0)
        # 将所有目标置信度目标拼接为一个张量。
    
        fg_masks = torch.cat(fg_masks, 0)
        # 将所有前景掩码拼接为一个张量。
    
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
            # 如果使用 L1 损失，则将所有 L1 目标拼接为一个张量。
    
        num_fg = max(num_fg, 1)
        # 确保前景数量至少为 1。
    
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        # 计算 IoU 损失。
    
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        # 计算目标置信度损失。
    
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        # 计算类别损失。
    
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
            # 计算 L1 损失。
        else:
            loss_l1 = 0.0
            # 如果不使用 L1 损失，则将 L1 损失设置为 0。
    
        reg_weight = 5.0
        # 设置回归损失的权重。
    
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        # 计算总损失。
    
        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )
        # 返回总损失、各个损失分量和前景比例。

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        # 定义获取 L1 目标的函数，接收 L1 目标、真实边界框、步幅、x 轴偏移、y 轴偏移和一个小的 epsilon 值。
        
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        # 计算 x 轴坐标的 L1 目标。
    
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        # 计算 y 轴坐标的 L1 目标。
    
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        # 计算宽度的 L1 目标，使用对数变换。
    
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        # 计算高度的 L1 目标，使用对数变换。
    
        return l1_target
        # 返回计算后的 L1 目标。
    
    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        # 定义获取目标分配的函数，接收批次索引、真实目标数量、真实边界框、真实类别、边界框预测、扩展步幅、x 轴偏移、y 轴偏移、类别预测、目标预测和模式（CPU/GPU）。
    
        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            # 如果模式为 CPU，则打印信息。
    
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            # 将真实边界框转换为 CPU 张量并转为浮点型。
    
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            # 将边界框预测转换为 CPU 张量并转为浮点型。
    
            gt_classes = gt_classes.cpu().float()
            # 将真实类别转换为 CPU 张量并转为浮点型。
    
            expanded_strides = expanded_strides.cpu().float()
            # 将扩展步幅转换为 CPU 张量并转为浮点型。
    
            x_shifts = x_shifts.cpu()
            # 将 x 轴偏移转换为 CPU 张量。
    
            y_shifts = y_shifts.cpu()
            # 将 y 轴偏移转换为 CPU 张量。
    
        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        # 获取前景掩码和几何关系约束。
    
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # 根据前景掩码筛选边界框预测。
    
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        # 根据前景掩码筛选类别预测。
    
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        # 根据前景掩码筛选目标预测。
    
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        # 获取当前批次中锚框的数量。
    
        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            # 如果模式为 CPU，则将真实边界框转换为 CPU 张量。
    
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
            # 将边界框预测转换为 CPU 张量。
    
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # 计算真实边界框与预测边界框之间的成对 IoU。
    
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        # 创建真实类别的 one-hot 编码。
    
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        # 计算成对 IoU 损失。
    
        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
            # 如果模式为 CPU，则将类别预测和目标预测转换为 CPU 张量。
    
        with torch.cuda.amp.autocast(enabled=False):
            # 禁用自动混合精度。
    
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            # 计算类别预测，使用 sigmoid 激活函数并开平方。
    
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
            # 计算成对类别损失。
    
        del cls_preds_
        # 删除临时变量以释放内存。
    
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )
        # 计算总成本，包括类别损失、IoU 损失和几何关系约束。
    
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        # 调用 simota_matching 函数进行目标匹配。
    
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        # 删除临时变量以释放内存。
    
        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
            # 如果模式为 CPU，则将匹配的类别和掩码转换为 GPU 张量。
    
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
        # 返回匹配的类别、前景掩码、预测 IoU、匹配的真实目标索引和前景数量。

    def get_geometry_constraint(
            self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
        ):
        # 定义获取几何约束的函数，接收真实边界框、扩展步幅、x 轴偏移和 y 轴偏移。
    
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        # 计算物体中心是否位于锚框的固定范围内，用于避免不当匹配，并减少候选锚框的数量，从而节省 GPU 内存。
    
        expanded_strides_per_image = expanded_strides[0]
        # 获取当前图像的扩展步幅。
    
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        # 计算当前图像中锚框中心的 x 坐标。
    
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        # 计算当前图像中锚框中心的 y 坐标。
    
        # in fixed center
        center_radius = 1.5
        # 设置中心半径为 1.5。
    
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        # 计算中心距离。
    
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        # 计算真实边界框左侧的坐标。
    
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        # 计算真实边界框右侧的坐标。
    
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        # 计算真实边界框上侧的坐标。
    
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist
        # 计算真实边界框下侧的坐标。
    
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        # 计算锚框中心与真实边界框左侧的距离。
    
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        # 计算锚框中心与真实边界框右侧的距离。
    
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        # 计算锚框中心与真实边界框上侧的距离。
    
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        # 计算锚框中心与真实边界框下侧的距离。
    
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        # 将四个方向的距离堆叠成一个张量。
    
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        # 判断锚框中心是否在真实边界框的范围内。
    
        anchor_filter = is_in_centers.sum(dim=0) > 0
        # 过滤出有效的锚框。
    
        geometry_relation = is_in_centers[:, anchor_filter]
        # 获取几何关系。
    
        return anchor_filter, geometry_relation
        # 返回有效锚框的过滤器和几何关系。
    
    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # 定义 SimOTA 匹配的函数，接收成本矩阵、成对 IoU、真实类别、真实目标数量和前景掩码。
    
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # 初始化匹配矩阵，形状与成本矩阵相同。
    
        n_candidate_k = min(10, pair_wise_ious.size(1))
        # 获取候选锚框的数量，最多为 10。
    
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        # 获取每个真实目标的前 k 个 IoU。
    
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        # 计算动态 k 值，确保至少为 1。
    
        for gt_idx in range(num_gt):
            # 遍历每个真实目标。
    
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            # 获取当前真实目标的最小成本索引。
    
            matching_matrix[gt_idx][pos_idx] = 1
            # 更新匹配矩阵。
    
        del topk_ious, dynamic_ks, pos_idx
        # 删除临时变量以释放内存。
    
        anchor_matching_gt = matching_matrix.sum(0)
        # 计算每个锚框匹配的真实目标数量。
    
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            # 如果一个锚框匹配多个真实目标，则执行以下操作。
    
            multiple_match_mask = anchor_matching_gt > 1
            # 创建多重匹配掩码。
    
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            # 找到成本最小的真实目标。
    
            matching_matrix[:, multiple_match_mask] *= 0
            # 将多重匹配的锚框的匹配矩阵置为 0。
    
            matching_matrix[cost_argmin, multiple_match_mask] = 1
            # 仅保留成本最小的匹配。
    
        fg_mask_inboxes = anchor_matching_gt > 0
        # 获取前景掩码。
    
        num_fg = fg_mask_inboxes.sum().item()
        # 计算前景数量。
    
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        # 更新前景掩码。
    
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 获取匹配的真实目标索引。
    
        gt_matched_classes = gt_classes[matched_gt_inds]
        # 获取匹配的真实目标类别。
    
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        # 计算当前匹配的 IoU。
    
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        # 返回前景数量、匹配的真实目标类别、当前匹配的 IoU 和匹配的真实目标索引。

    
    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        # 原始的前向逻辑
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.
        # TODO: 在这里使用前向逻辑

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            # 遍历每个层级的卷积和输入
            x = self.stems[k](x)  # 通过当前层的stem处理输入
            cls_x = x  # 分类特征
            reg_x = x  # 回归特征

            cls_feat = cls_conv(cls_x)  # 通过分类卷积层获取分类特征
            cls_output = self.cls_preds[k](cls_feat)  # 通过分类预测层获取分类输出
            reg_feat = reg_conv(reg_x)  # 通过回归卷积层获取回归特征
            reg_output = self.reg_preds[k](reg_feat)  # 通过回归预测层获取回归输出
            obj_output = self.obj_preds[k](reg_feat)  # 通过目标预测层获取目标输出

            output = torch.cat([reg_output, obj_output, cls_output], 1)  # 将回归、目标和分类输出拼接
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())  # 获取输出和网格
            x_shifts.append(grid[:, :, 0])  # 添加x方向的网格偏移
            y_shifts.append(grid[:, :, 1])  # 添加y方向的网格偏移
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])  # 扩展步幅
            )
            outputs.append(output)  # 添加当前层的输出

        outputs = torch.cat(outputs, 1)  # 将所有层的输出拼接
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4] 获取边界框预测
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1] 获取目标预测
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls] 获取分类预测

        # calculate targets
        # 计算目标
        total_num_anchors = outputs.shape[1]  # 总锚点数量
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all] 将x方向的偏移拼接
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all] 将y方向的偏移拼接
        expanded_strides = torch.cat(expanded_strides, 1)  # 拼接扩展步幅

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # 计算物体数量
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            # 遍历每个批次的图像、真实物体数量和标签
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)  # 调整图像维度并转换为uint8类型
            num_gt = int(num_gt)  # 将真实物体数量转换为整数
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()  # 如果没有真实物体，创建全零的前景掩码
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]  # 获取当前图像的真实边界框
                gt_classes = label[:num_gt, 0]  # 获取当前图像的真实类别
                bboxes_preds_per_image = bbox_preds[batch_idx]  # 获取当前图像的边界框预测
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )  # 获取分配结果

            img = img.cpu().numpy().copy()  # 将图像从GPU转移到CPU并复制
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],  # 计算前景坐标
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)  # 将中心坐标格式转换为xyxy格式
            save_name = save_prefix + str(batch_idx) + ".png"  # 生成保存文件名
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)  # 可视化分配结果
            logger.info(f"save img to {save_name}")  # 记录保存信息