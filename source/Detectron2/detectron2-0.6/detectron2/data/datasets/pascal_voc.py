# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog  # 导入数据集目录和元数据目录相关的类
from detectron2.structures import BoxMode  # 导入边界框模式相关的类
from detectron2.utils.file_io import PathManager  # 导入路径管理器，用于处理文件路径

__all__ = ["load_voc_instances", "register_pascal_voc"]  # 定义模块的公开接口


# fmt: off
CLASS_NAMES = (  # 定义Pascal VOC数据集的20个类别名称
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
# fmt: on


def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    将Pascal VOC检测数据集的标注转换为Detectron2格式

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        dirname: 包含"Annotations"(标注)、"ImageSets"(图像集)、"JPEGImages"(图像)的目录
        split (str): one of "train", "test", "val", "trainval"
        split (str): 数据集划分，可以是"train"、"test"、"val"、"trainval"中的一个
        class_names: list or tuple of class names
        class_names: 类别名称的列表或元组
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:  # 打开数据集划分文件
        fileids = np.loadtxt(f, dtype=np.str)  # 读取图像ID列表

    # Needs to read many small annotation files. Makes sense at local
    # 需要读取多个小的标注文件，在本地处理更有效率
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))  # 获取标注文件目录的本地路径
    dicts = []  # 用于存储转换后的标注信息
    for fileid in fileids:  # 遍历每个图像ID
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")  # 构建XML标注文件路径
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")  # 构建图像文件路径

        with PathManager.open(anno_file) as f:  # 打开XML标注文件
            tree = ET.parse(f)  # 解析XML文件

        r = {  # 创建图像信息字典
            "file_name": jpeg_file,  # 图像文件路径
            "image_id": fileid,  # 图像ID
            "height": int(tree.findall("./size/height")[0].text),  # 图像高度
            "width": int(tree.findall("./size/width")[0].text),  # 图像宽度
        }
        instances = []  # 用于存储图像中的所有目标实例

        for obj in tree.findall("object"):  # 遍历XML中的每个目标对象
            cls = obj.find("name").text  # 获取目标类别名称
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # 我们在训练中包含了"difficult"样本
            # 根据有限的实验，这些样本不会影响准确率
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")  # 获取边界框信息
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]  # 提取边界框坐标
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            # 原始标注是范围在[1, W或H]的整数
            # 假设这些是基于1的像素索引（包含边界）
            # 标注为(xmin=1, xmax=W)的框覆盖整个图像
            # 在坐标空间中表示为(xmin=0, xmax=W)
            bbox[0] -= 1.0  # 将x坐标从1基转换为0基
            bbox[1] -= 1.0  # 将y坐标从1基转换为0基
            instances.append(  # 添加目标实例信息
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}  # 类别ID、边界框坐标和边界框模式
            )
        r["annotations"] = instances  # 将所有实例信息添加到图像信息字典
        dicts.append(r)  # 将图像信息添加到数据集列表
    return dicts  # 返回转换后的数据集信息


def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):  # 注册Pascal VOC数据集
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))  # 注册数据集加载函数
    MetadataCatalog.get(name).set(  # 设置数据集元数据
        thing_classes=list(class_names), dirname=dirname, year=year, split=split  # 设置类别名称、数据集目录、年份和数据集划分信息
    )
