# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
该文件在硬编码路径注册预定义的数据集及其元数据。

We hard-code metadata for common datasets. This will enable:
我们为常用数据集硬编码元数据。这将实现：
1. Consistency check when loading the datasets
1. 加载数据集时的一致性检查
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
2. 直接在这些标准数据集上使用模型并运行演示，无需下载数据集注释

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".
我们硬编码了一些数据集路径，这些数据集假定存在于"./datasets/"中。

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
用户不应使用此文件为新数据集创建新的数据集/元数据。
要添加新数据集，请参考教程"docs/DATASETS.md"。
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========
# ==== COCO数据集的预定义数据集和分割 ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # 定义COCO数据集的各个分割，包括训练集、验证集和测试集
    # 每个键值对的格式为：(图像目录, 注释文件)
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    # 定义COCO人体关键点数据集的各个分割
    # 包含人体关键点标注的训练集和验证集
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        # 这是原始的全景分割标注目录
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        # 此目录包含从全景分割标注转换而来的语义分割标注
        # 它被PanopticFPN使用
        # 你可以使用detectron2/datasets/prepare_panoptic_fpn.py脚本创建这些目录
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    # 注册所有COCO数据集
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            # 假设预定义数据集位于`./datasets`目录
            register_coco_instances(
                key,  # 数据集名称
                _get_builtin_metadata(dataset_name),  # 获取数据集元数据
                os.path.join(root, json_file) if "://" not in json_file else json_file,  # 注释文件路径
                os.path.join(root, image_root),  # 图像目录路径
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        # COCO全景分割数据集的"分离"版本，例如被Panoptic FPN使用
        register_coco_panoptic_separated(
            prefix,  # 数据集名称
            _get_builtin_metadata("coco_panoptic_separated"),  # 获取分离版本的元数据
            image_root,  # 图像目录
            os.path.join(root, panoptic_root),  # 全景分割标注目录
            os.path.join(root, panoptic_json),  # 全景分割注释文件
            os.path.join(root, semantic_root),  # 语义分割标注目录
            instances_json,  # 实例分割注释文件
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        # COCO全景分割数据集的"标准"版本，例如被Panoptic-DeepLab使用
        register_coco_panoptic(
            prefix,  # 数据集名称
            _get_builtin_metadata("coco_panoptic_standard"),  # 获取标准版本的元数据
            image_root,  # 图像目录
            os.path.join(root, panoptic_root),  # 全景分割标注目录
            os.path.join(root, panoptic_json),  # 全景分割注释文件
            instances_json,  # 实例分割注释文件
        )


# ==== Predefined datasets and splits for LVIS ==========
# ==== LVIS数据集的预定义数据集和分割 ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        # LVIS v1版本的数据集分割
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        # LVIS v0.5版本的数据集分割
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        # LVIS v0.5版本转换为COCO格式的数据集分割
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    # 注册所有LVIS数据集
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,  # 数据集名称
                get_lvis_instances_meta(dataset_name),  # 获取LVIS数据集元数据
                os.path.join(root, json_file) if "://" not in json_file else json_file,  # 注释文件路径
                os.path.join(root, image_root),  # 图像目录路径
            )


# ==== Predefined splits for raw cityscapes images ===========
# ==== 原始Cityscapes图像的预定义分割 ===========
_RAW_CITYSCAPES_SPLITS = {
    # 定义Cityscapes数据集的训练集、验证集和测试集
    # 每个分割包含左侧8位图像和精细标注
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    # 注册所有Cityscapes数据集
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")  # 获取Cityscapes数据集的元数据
        image_dir = os.path.join(root, image_dir)  # 构建图像目录的完整路径
        gt_dir = os.path.join(root, gt_dir)  # 构建标注目录的完整路径

        inst_key = key.format(task="instance_seg")  # 实例分割任务的数据集名称
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),  # 注册实例分割数据集的加载函数
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )  # 设置实例分割数据集的元数据

        sem_key = key.format(task="sem_seg")  # 语义分割任务的数据集名称
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )  # 注册语义分割数据集的加载函数
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )  # 设置语义分割数据集的元数据


# ==== Predefined splits for PASCAL VOC ===========
# ==== Pascal VOC数据集的预定义分割 ===========
def register_all_pascal_voc(root):
    # 定义Pascal VOC数据集的各个分割
    # 包括2007年和2012年的训练集、验证集和测试集
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012  # 根据数据集名称确定年份
        register_pascal_voc(name, os.path.join(root, dirname), split, year)  # 注册Pascal VOC数据集
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"  # 设置评估器类型


def register_all_ade20k(root):
    # 注册ADE20K语义分割数据集
    root = os.path.join(root, "ADEChallengeData2016")  # ADE20K数据集的根目录
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)  # 图像目录
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)  # 标注目录
        name = f"ade20k_sem_seg_{name}"  # 构建数据集名称
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )  # 注册数据集的加载函数
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],  # 设置语义类别
            image_root=image_dir,  # 设置图像根目录
            sem_seg_root=gt_dir,  # 设置语义分割标注根目录
            evaluator_type="sem_seg",  # 设置评估器类型
            ignore_label=255,  # 设置忽略标签
            **metadata,  # 其他元数据
        )


# True for open source;
# Internally at fb, we register them elsewhere
# 对于开源版本为True；
# 在Facebook内部，我们在其他地方注册这些数据集
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    # 假设预定义数据集位于`./datasets`目录
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")  # 获取数据集根目录
    register_all_coco(_root)  # 注册所有COCO数据集
    register_all_lvis(_root)  # 注册所有LVIS数据集
    register_all_cityscapes(_root)  # 注册所有Cityscapes数据集
    register_all_cityscapes_panoptic(_root)  # 注册所有Cityscapes全景分割数据集
    register_all_pascal_voc(_root)  # 注册所有Pascal VOC数据集
    register_all_ade20k(_root)  # 注册所有ADE20K数据集
