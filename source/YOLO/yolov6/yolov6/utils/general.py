#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os  # 导入os模块，用于处理操作系统相关的功能
import glob  # 导入glob模块，用于文件路径匹配
import math  # 导入math模块，用于数学计算
import torch  # 导入PyTorch库
import requests  # 导入requests库，用于HTTP请求
import pkg_resources as pkg  # 导入pkg_resources模块，用于处理包资源
from pathlib import Path  # 从pathlib导入Path类，用于处理文件路径
from yolov6.utils.events import LOGGER  # 从yolov6.utils.events导入日志记录器

def increment_name(path):
    '''increase save directory's id'''
    # 增加保存目录的ID
    path = Path(path)  # 将路径转换为Path对象
    sep = ''  # 初始化分隔符
    if path.exists():  # 如果路径存在
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')  # 分离路径和后缀
        for n in range(1, 9999):  # 遍历1到9999
            p = f'{path}{sep}{n}{suffix}'  # 生成新的路径
            if not os.path.exists(p):  # 如果路径不存在
                break  # 找到可用路径，跳出循环
        path = Path(p)  # 更新路径为可用路径
    return path  # 返回新的路径

def find_latest_checkpoint(search_dir='.'):
    '''Find the most recent saved checkpoint in search_dir.'''
    # 在search_dir中查找最近保存的检查点
    checkpoint_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)  # 查找所有匹配的检查点文件
    return max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''  # 返回最新的检查点路径，如果没有则返回空字符串

def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    # 将距离(ltrb)转换为框(xywh或xyxy)
    lt, rb = torch.split(distance, 2, -1)  # 将距离分为左上角和右下角
    x1y1 = anchor_points - lt  # 计算左上角坐标
    x2y2 = anchor_points + rb  # 计算右下角坐标
    if box_format == 'xyxy':  # 如果框格式为xyxy
        bbox = torch.cat([x1y1, x2y2], -1)  # 连接左上角和右下角坐标
    elif box_format == 'xywh':  # 如果框格式为xywh
        c_xy = (x1y1 + x2y2) / 2  # 计算中心坐标
        wh = x2y2 - x1y1  # 计算宽和高
        bbox = torch.cat([c_xy, wh], -1)  # 连接中心坐标和宽高
    return bbox  # 返回框

def bbox2dist(anchor_points, bbox, reg_max):
    '''Transform bbox(xyxy) to dist(ltrb).'''
    # 将框(xyxy)转换为距离(ltrb)
    x1y1, x2y2 = torch.split(bbox, 2, -1)  # 将框分为左上角和右下角
    lt = anchor_points - x1y1  # 计算左上角距离
    rb = x2y2 - anchor_points  # 计算右下角距离
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)  # 连接距离并限制范围
    return dist  # 返回距离

def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    # 将框(xywh)转换为框(xyxy)
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5  # 计算左上角x坐标
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5  # 计算左上角y坐标
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]  # 计算右下角x坐标
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]  # 计算右下角y坐标
    return bboxes  # 返回转换后的框

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # 返回框的交并比（Jaccard指数）
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])  # 计算框的面积

    area1 = box_area(box1.T)  # 计算box1的面积
    area2 = box_area(box2.T)  # 计算box2的面积

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)  # 计算交集面积
    return inter / (area1[:, None] + area2 - inter)  # 返回IoU

def download_ckpt(path):
    """Download checkpoints of the pretrained models"""
    # 下载预训练模型的检查点
    basename = os.path.basename(path)  # 获取文件名
    dir = os.path.abspath(os.path.dirname(path))  # 获取保存路径的绝对路径
    os.makedirs(dir, exist_ok=True)  # 创建保存目录，如果已存在则忽略
    LOGGER.info(f"checkpoint {basename} not exist, try to downloaded it from github.")  # 记录信息，检查点不存在，将从GitHub下载
    # need to update the link with every release
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.4.0/{basename}"  # 设置下载链接
    LOGGER.warning(f"downloading url is: {url}, please make sure the version of the downloading model is corresponding to the code version!")  # 记录下载警告信息
    r = requests.get(url, allow_redirects=True)  # 发送GET请求下载文件
    assert r.status_code == 200, "Unable to download checkpoints, manually download it"  # 检查请求是否成功
    open(path, 'wb').write(r.content)  # 将下载的内容写入文件
    LOGGER.info(f"checkpoint {basename} downloaded and saved")  # 记录下载成功的信息

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor  # 返回最接近的可被divisor整除的数

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    # 验证图像大小在每个维度上是否是步幅s的倍数
    if isinstance(imgsz, int):  # 如果图像大小是整数
        new_size = max(make_divisible(imgsz, int(s)), floor)  # 计算新的图像大小
    else:  # 如果图像大小是列表
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]  # 计算每个维度的新大小
    if new_size != imgsz:  # 如果新大小与原大小不同
        LOGGER.warning(f'--img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')  # 记录警告信息
    return new_size  # 返回新的图像大小

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check whether the package's version is match the required version.
    # 检查包的版本是否符合要求
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))  # 解析当前版本和最低版本
    result = (current == minimum) if pinned else (current >= minimum)  # 判断版本是否符合要求
    if hard:  # 如果是强制检查
        info = f'⚠️ {name}{minimum} is required by YOLOv6, but {name}{current} is currently installed'  # 记录错误信息
        assert result, info  # 断言版本符合要求
    return result  # 返回检查结果