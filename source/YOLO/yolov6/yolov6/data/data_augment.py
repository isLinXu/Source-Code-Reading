#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import math
import random

import cv2
import numpy as np


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    '''HSV color-space augmentation.'''  # HSV颜色空间增强
    if hgain or sgain or vgain:
        # 生成随机增益值，范围在[1-gain, 1+gain]之间
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # 将图像从BGR转换到HSV颜色空间，并分离三个通道
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8  # 保存原始图像的数据类型

        # 创建查找表(LUT)用于颜色空间变换
        x = np.arange(0, 256, dtype=r.dtype)
        # 色调变换，需要对180取模，因为OpenCV中H通道范围是[0,180]
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        # 饱和度变换，需要裁剪到[0,255]范围
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # 明度变换，需要裁剪到[0,255]范围
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # 使用查找表对每个通道进行变换，然后合并通道
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # 将图像转换回BGR颜色空间，直接修改输入图像
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    '''Resize and pad image while meeting stride-multiple constraints.'''  # 调整图像大小并填充，同时满足步长约束
    shape = im.shape[:2]  # current shape [height, width]  # 获取当前图像的高度和宽度
    if isinstance(new_shape, int):
        # 如果new_shape是整数，转换为正方形尺寸元组
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       # 如果new_shape是只有一个元素的列表，转换为正方形尺寸元组
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)  # 计算缩放比例（新尺寸/原尺寸）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)  # 如果不放大，则只进行缩小操作（为了更好的验证mAP）
        r = min(r, 1.0)

    # Compute padding  # 计算填充大小
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 计算缩放后的宽度和高度
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding  # 计算需要填充的宽度和高度

    if auto:  # minimum rectangle  # 最小矩形模式
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding  # 确保填充后的尺寸是stride的倍数

    dw /= 2  # divide padding into 2 sides  # 将填充平均分配到两侧
    dh /= 2

    if shape[::-1] != new_unpad:  # resize  # 如果尺寸发生变化，则调整图像大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下左右填充的像素数，并四舍五入
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 使用指定颜色对图像进行填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r, (left, top)  # 返回处理后的图像、缩放比例和填充信息


def mixup(im, labels, im2, labels2):
    '''Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf.'''  # 应用MixUp数据增强方法
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0  # 使用Beta分布生成混合比例，alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)  # 按比例混合两张图片
    labels = np.concatenate((labels, labels2), 0)  # 合并两组标签
    return im, labels  # 返回混合后的图像和标签


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    '''Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio.'''  # 计算候选框：box1是增强前的框，box2是增强后的框，wh_thr是宽高阈值，ar_thr是长宽比阈值，area_thr是面积比阈值
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]  # 计算原始框的宽度和高度
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]  # 计算增强后框的宽度和高度
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio  # 计算长宽比，取最大值以处理长方形框
    # 返回满足条件的候选框：宽高大于阈值，面积比大于阈值，长宽比小于阈值
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def random_affine(img, labels=(), degrees=10, translate=.1, scale=.1, shear=10,
                  new_shape=(640, 640)):
    '''Applies Random affine transformation.'''  # 应用随机仿射变换
    n = len(labels)  # 获取标签数量
    if isinstance(new_shape, int):
        # 如果new_shape是整数，设置相同的宽高
        height = width = new_shape
    else:
        # 否则分别获取高度和宽度
        height, width = new_shape

    # 获取变换矩阵和缩放因子
    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # image changed  # 如果变换矩阵不是单位矩阵，说明图像需要变换
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))  # 应用仿射变换

    # Transform label coordinates  # 转换标签坐标
    if n:
        new = np.zeros((n, 4))  # 创建新的标签数组

        # 将标签坐标转换为齐次坐标系
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1  # 重排标签坐标点
        xy = xy @ M.T  # transform  # 应用变换矩阵
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine  # 将结果转回二维坐标

        # create new boxes  # 创建新的边界框
        x = xy[:, [0, 2, 4, 6]]  # 提取所有x坐标
        y = xy[:, [1, 3, 5, 7]]  # 提取所有y坐标
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # 计算新的边界框坐标

        # clip  # 裁剪坐标到图像范围内
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)  # x坐标裁剪
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)  # y坐标裁剪

        # filter candidates  # 过滤不合适的边界框
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.1)  # 使用box_candidates函数筛选
        labels = labels[i]  # 保留满足条件的标签
        labels[:, 1:5] = new[i]  # 更新标签坐标

    return img, labels  # 返回变换后的图像和标签


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    # 获取仿射变换矩阵的函数
    new_height, new_width = new_shape  # 解析目标尺寸
    # Center  # 中心变换矩阵
    C = np.eye(3)  # 创建3x3单位矩阵
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)  # x方向平移（将原点移到图像中心）
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)  # y方向平移（将原点移到图像中心）

    # Rotation and Scale  # 旋转和缩放矩阵
    R = np.eye(3)  # 创建3x3单位矩阵
    a = random.uniform(-degrees, degrees)  # 随机生成旋转角度
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations  # 可选：添加90度的旋转
    s = random.uniform(1 - scale, 1 + scale)  # 随机生成缩放比例
    # s = 2 ** random.uniform(-scale, scale)  # 可选：使用指数缩放
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)  # 获取2D旋转矩阵

    # Shear  # 剪切变换矩阵
    S = np.eye(3)  # 创建3x3单位矩阵
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)  # x方向剪切（角度）
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)  # y方向剪切（角度）

    # Translation  # 平移矩阵
    T = np.eye(3)  # 创建3x3单位矩阵
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)  # x方向随机平移
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)  # y方向随机平移

    # Combined rotation matrix  # 组合所有变换矩阵
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT  # 矩阵乘法顺序很重要：先中心化，再旋转缩放，再剪切，最后平移
    return M, s  # 返回变换矩阵和缩放因子


def mosaic_augmentation(shape, imgs, hs, ws, labels, hyp, specific_shape = False, target_height=640, target_width=640):
    '''Applies Mosaic augmentation.'''  # 应用Mosaic数据增强
    assert len(imgs) == 4, "Mosaic augmentation of current version only supports 4 images."  # 确保输入正好是4张图片
    labels4 = []  # 存储所有标签的列表
    if not specific_shape:
        # 确定目标尺寸
        if isinstance(shape, list) or isinstance(shape, np.ndarray):
            target_height, target_width = shape
        else:
            target_height = target_width = shape

    # 随机生成马赛克中心点坐标
    yc, xc = (int(random.uniform(x//2, 3*x//2)) for x in (target_height, target_width) )  # mosaic center x, y

    for i in range(len(imgs)):
        # Load image  # 加载图像
        img, h, w = imgs[i], hs[i], ws[i]
        # place img in img4  # 将图片放置在马赛克图像中
        if i == 0:  # top left  # 左上角
            img4 = np.full((target_height * 2, target_width * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles  # 创建马赛克底图

            # 计算大图和小图的坐标
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right  # 右上角
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left  # 左下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right  # 右下角
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(target_height * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]  # 将图片复制到对应位置
        padw = x1a - x1b  # 计算x方向的填充量
        padh = y1a - y1b  # 计算y方向的填充量

        # Labels  # 处理标签
        labels_per_img = labels[i].copy()  # 复制当前图片的标签
        if labels_per_img.size:  # 如果有标签
            boxes = np.copy(labels_per_img[:, 1:])  # 复制边界框坐标
            # 转换边界框坐标（考虑填充）
            boxes[:, 0] = w * (labels_per_img[:, 1] - labels_per_img[:, 3] / 2) + padw  # top left x
            boxes[:, 1] = h * (labels_per_img[:, 2] - labels_per_img[:, 4] / 2) + padh  # top left y
            boxes[:, 2] = w * (labels_per_img[:, 1] + labels_per_img[:, 3] / 2) + padw  # bottom right x
            boxes[:, 3] = h * (labels_per_img[:, 2] + labels_per_img[:, 4] / 2) + padh  # bottom right y
            labels_per_img[:, 1:] = boxes  # 更新标签中的坐标

        labels4.append(labels_per_img)  # 添加到标签列表

    # Concat/clip labels  # 合并和裁剪标签
    labels4 = np.concatenate(labels4, 0)  # 合并所有标签
    # 将坐标裁剪到有效范围内
    labels4[:, 1::2] = np.clip(labels4[:, 1::2], 0, 2 * target_width)  # x坐标
    labels4[:, 2::2] = np.clip(labels4[:, 2::2], 0, 2 * target_height)  # y坐标

    # Augment  # 对马赛克图像进行仿射变换增强
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=hyp['degrees'],
                                  translate=hyp['translate'],
                                  scale=hyp['scale'],
                                  shear=hyp['shear'],
                                  new_shape=(target_height, target_width))

    return img4, labels4  # 返回增强后的图像和标签
