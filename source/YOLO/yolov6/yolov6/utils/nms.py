#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# The code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/general.py

import os  # 导入os模块，用于处理操作系统相关的功能
import time  # 导入time模块，用于时间相关的操作
import numpy as np  # 导入numpy库，用于数值计算
import cv2  # 导入OpenCV库，用于图像处理
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于计算机视觉相关功能


# Settings
# 设置打印选项
torch.set_printoptions(linewidth=320, precision=5, profile='long')  # 设置PyTorch打印选项
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5 设置numpy打印选项
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)  # 防止OpenCV多线程（与PyTorch DataLoader不兼容）
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads  # 设置NumExpr最大线程数


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 克隆输入x，如果是torch.Tensor则克隆，否则复制
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x  # 计算左上角x坐标
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y  # 计算左上角y坐标
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x  # 计算右下角x坐标
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y  # 计算右下角y坐标
    return y  # 返回转换后的坐标


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes  # 类别数量
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates  # 选择候选框
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'  # 检查置信度阈值
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'  # 检查IoU阈值

    # Function settings.
    max_wh = 4096  # maximum box width and height  # 最大框宽高
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()  # 最大框数量
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.  # 超过时间限制则退出
    multi_label &= num_classes > 1  # multiple labels per box  # 每个框多个标签

    tik = time.time()  # 记录开始时间
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]  # 初始化输出
    for img_idx, x in enumerate(prediction):  # 遍历每张图片的预测结果
        x = x[pred_candidates[img_idx]]  # 选择置信度高的框

        # If no box remains, skip the next process.
        if not x.shape[0]:  # 如果没有框
            continue  # 跳过

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf  # 计算最终置信度

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])  # 将框从中心坐标和宽高转换为左上角和右下角坐标

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T  # 获取多标签框的索引
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)  # 合并框、置信度和类别索引
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)  # 选取置信度最高的类别
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]  # 合并框、置信度和类别索引

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]  # 过滤类别

        # Check shape
        num_box = x.shape[0]  # number of boxes  # 框的数量
        if not num_box:  # no boxes kept.
            continue  # 跳过
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence  # 按照置信度排序

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes  # 类别偏移量
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores  # 获取框和置信度
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS  # 进行非极大值抑制
        if keep_box_idx.shape[0] > max_det:  # limit detections  # 限制检测框数量
            keep_box_idx = keep_box_idx[:max_det]  # 只保留最大数量的框

        output[img_idx] = x[keep_box_idx]  # 保存结果
        if (time.time() - tik) > time_limit:  # 检查时间限制
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')  # 打印警告信息
            break  # 超过时间限制则退出

    return output  # 返回检测结果
