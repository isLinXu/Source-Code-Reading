#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

# import math
# import random

# import cv2
# import numpy as np

# from yolox.utils import xyxy2cxcywh


# def augment_hsv(img, hgain=5, sgain=30, vgain=30):
#     hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
#     hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
#     hsv_augs = hsv_augs.astype(np.int16)
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

#     img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
#     img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
#     img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

#     cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


# def get_aug_params(value, center=0):
#     if isinstance(value, float):
#         return random.uniform(center - value, center + value)
#     elif len(value) == 2:
#         return random.uniform(value[0], value[1])
#     else:
#         raise ValueError(
#             "Affine params should be either a sequence containing two values\
#              or single float values. Got {}".format(value)
#         )


# def get_affine_matrix(
#     target_size,
#     degrees=10,
#     translate=0.1,
#     scales=0.1,
#     shear=10,
# ):
#     twidth, theight = target_size

#     # Rotation and Scale
#     angle = get_aug_params(degrees)
#     scale = get_aug_params(scales, center=1.0)

#     if scale <= 0.0:
#         raise ValueError("Argument scale should be positive")

#     R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

#     M = np.ones([2, 3])
#     # Shear
#     shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
#     shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

#     M[0] = R[0] + shear_y * R[1]
#     M[1] = R[1] + shear_x * R[0]

#     # Translation
#     translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
#     translation_y = get_aug_params(translate) * theight  # y translation (pixels)

#     M[0, 2] = translation_x
#     M[1, 2] = translation_y

#     return M, scale


import math  # 导入数学模块
import random  # 导入随机数模块

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库

from yolox.utils import xyxy2cxcywh  # 从yolox.utils模块导入xyxy2cxcywh函数


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # 随机增益
    hsv_augs *= np.random.randint(0, 2, 3)  # 随机选择h, s, v
    hsv_augs = hsv_augs.astype(np.int16)  # 转换增益为int16类型
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)  # 将图像从BGR转换为HSV格式并转换为int16类型

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180  # 调整H通道并确保在0-179范围内
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)  # 调整S通道并确保在0-255范围内
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)  # 调整V通道并确保在0-255范围内

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # 将处理后的HSV图像转换回BGR格式并更新原图像


def get_aug_params(value, center=0):
    if isinstance(value, float):  # 如果value是浮点数
        return random.uniform(center - value, center + value)  # 返回center附近的随机浮点数
    elif len(value) == 2:  # 如果value是一个包含两个元素的序列
        return random.uniform(value[0], value[1])  # 返回value范围内的随机浮点数
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )  # 抛出异常，提示参数格式不正确


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size  # 获取目标尺寸的宽度和高度

    # Rotation and Scale
    angle = get_aug_params(degrees)  # 获取随机旋转角度
    scale = get_aug_params(scales, center=1.0)  # 获取随机缩放因子

    if scale <= 0.0:  # 如果缩放因子小于等于0
        raise ValueError("Argument scale should be positive")  # 抛出异常，提示缩放因子应为正数

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)  # 获取旋转矩阵

    M = np.ones([2, 3])  # 初始化仿射矩阵
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)  # 获取随机剪切因子（x方向）
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)  # 获取随机剪切因子（y方向）

    M[0] = R[0] + shear_y * R[1]  # 更新仿射矩阵的第一行
    M[1] = R[1] + shear_x * R[0]  # 更新仿射矩阵的第二行

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x方向的平移（像素）
    translation_y = get_aug_params(translate) * theight  # y方向的平移（像素）

    M[0, 2] = translation_x  # 设置仿射矩阵的x方向平移
    M[1, 2] = translation_y  # 设置仿射矩阵的y方向平移

    return M, scale  # 返回仿射矩阵和缩放因子


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)  # 获取目标框的数量

    # warp corner points
    twidth, theight = target_size  # 获取目标尺寸的宽度和高度
    corner_points = np.ones((4 * num_gts, 3))  # 创建一个数组，用于存储角点
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform 应用仿射变换
    corner_points = corner_points.reshape(num_gts, 8)  # 重塑角点数组的形状

    # create new boxes
    corner_xs = corner_points[:, 0::2]  # 提取角点的 x 坐标
    corner_ys = corner_points[:, 1::2]  # 提取角点的 y 坐标
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )  # 创建新的边界框

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)  # 限制 x 坐标在图像宽度范围内
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)  # 限制 y 坐标在图像高度范围内

    targets[:, :4] = new_bboxes  # 更新目标框

    return targets  # 返回更新后的目标框




def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)  # 获取仿射变换矩阵和缩放因子

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))  # 应用仿射变换到图像，设置边界值为114

    # Transform label coordinates
    if len(targets) > 0:  # 如果目标框存在
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)  # 对目标框应用仿射变换

    return img, targets  # 返回变换后的图像和目标框


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape  # 获取图像的宽度
    if random.random() < prob:  # 根据概率决定是否进行镜像翻转
        image = image[:, ::-1]  # 镜像翻转图像
        boxes[:, 0::2] = width - boxes[:, 2::-2]  # 更新目标框的坐标
    return image, boxes  # 返回处理后的图像和目标框


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:  # 如果图像是三维的（彩色图像）
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114  # 创建一个填充图像，初始值为114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114  # 创建一个填充图像，初始值为114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])  # 计算缩放比例
    resized_img = cv2.resize(  # 调整图像大小
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),  # 根据缩放比例计算新的宽度和高度
        interpolation=cv2.INTER_LINEAR,  # 线性插值
    ).astype(np.uint8)  # 转换为uint8类型
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img  # 将调整大小后的图像放入填充图像中

    padded_img = padded_img.transpose(swap)  # 交换维度
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)  # 转换为连续数组并指定数据类型为float32
    return padded_img, r  # 返回处理后的图像和缩放比例



class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels  # 最大标签数量
        self.flip_prob = flip_prob  # 翻转概率
        self.hsv_prob = hsv_prob  # HSV增强概率

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()  # 复制目标框的坐标
        labels = targets[:, 4].copy()  # 复制目标框的标签
        if len(boxes) == 0:  # 如果没有目标框
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)  # 创建一个空的目标框数组
            image, r_o = preproc(image, input_dim)  # 预处理图像
            return image, targets  # 返回预处理后的图像和空目标框

        image_o = image.copy()  # 复制原始图像
        targets_o = targets.copy()  # 复制原始目标框
        height_o, width_o, _ = image_o.shape  # 获取原始图像的高度和宽度
        boxes_o = targets_o[:, :4]  # 获取原始目标框的坐标
        labels_o = targets_o[:, 4]  # 获取原始目标框的标签
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)  # 将目标框格式从 [xyxy] 转换为 [c_x, c_y, w, h]

        if random.random() < self.hsv_prob:  # 根据 HSV 概率决定是否进行 HSV 增强
            augment_hsv(image)  # 对图像进行 HSV 增强
        image_t, boxes = _mirror(image, boxes, self.flip_prob)  # 根据翻转概率进行图像翻转
        height, width, _ = image_t.shape  # 获取翻转后图像的高度和宽度
        image_t, r_ = preproc(image_t, input_dim)  # 预处理翻转后的图像
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)  # 将翻转后的目标框格式从 [xyxy] 转换为 [c_x, c_y, w, h]
        boxes *= r_  # 根据预处理的比例调整目标框

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1  # 创建一个掩码，筛选出有效目标框
        boxes_t = boxes[mask_b]  # 过滤出有效目标框
        labels_t = labels[mask_b]  # 过滤出有效目标框的标签

        if len(boxes_t) == 0:  # 如果没有有效目标框
            image_t, r_o = preproc(image_o, input_dim)  # 预处理原始图像
            boxes_o *= r_o  # 根据预处理的比例调整原始目标框
            boxes_t = boxes_o  # 使用原始目标框
            labels_t = labels_o  # 使用原始标签

        labels_t = np.expand_dims(labels_t, 1)  # 扩展标签的维度

        targets_t = np.hstack((labels_t, boxes_t))  # 将标签和目标框合并
        padded_labels = np.zeros((self.max_labels, 5))  # 创建一个填充的标签数组
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]  # 将有效目标框填充到标签数组中
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)  # 确保标签数组是连续的
        return image_t, padded_labels  # 返回处理后的图像和填充的标签


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD  # 输入尺寸到 SSD
        rgb_means ((int,int,int)): average RGB of the dataset  # 数据集的平均 RGB
            (104,117,123)
        swap ((int,int,int)): final order of channels  # 最终的通道顺序

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data  # 可调用的变换，用于测试/验证数据
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap  # 设置通道交换顺序
        self.legacy = legacy  # 是否使用遗留模式

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)  # 预处理图像，调整尺寸并交换通道
        if self.legacy:  # 如果使用遗留模式
            img = img[::-1, :, :].copy()  # 反转图像的通道顺序
            img /= 255.0  # 将像素值归一化到 [0, 1]
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)  # 减去均值
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)  # 除以标准差
        return img, np.zeros((1, 5))  # 返回处理后的图像和一个空的目标框数组
