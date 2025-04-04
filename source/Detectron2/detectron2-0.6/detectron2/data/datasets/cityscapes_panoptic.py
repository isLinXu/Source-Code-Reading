# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog  # 导入数据集目录和元数据目录的管理类
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES  # 导入Cityscapes数据集的类别信息
from detectron2.utils.file_io import PathManager  # 导入文件路径管理器

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
本文件包含了将Cityscapes全景分割数据集注册到数据集目录的相关函数。
"""


logger = logging.getLogger(__name__)  # 初始化日志记录器


def get_cityscapes_panoptic_files(image_dir, gt_dir, json_info):
    # 获取Cityscapes全景分割数据集的文件路径
    files = []
    # scan through the directory
    # 扫描目录获取所有城市的数据
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")  # 记录找到的城市数量
    image_dict = {}  # 用于存储图像ID到图像文件路径的映射
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)  # 构建城市图像目录的完整路径
        for basename in PathManager.ls(city_img_dir):  # 遍历城市目录下的所有图像文件
            image_file = os.path.join(city_img_dir, basename)  # 构建图像文件的完整路径

            suffix = "_leftImg8bit.png"  # Cityscapes数据集图像的后缀名
            assert basename.endswith(suffix), basename  # 确保文件名具有正确的后缀
            basename = os.path.basename(basename)[: -len(suffix)]  # 获取不包含后缀的文件名

            image_dict[basename] = image_file  # 将图像ID和文件路径添加到字典中

    for ann in json_info["annotations"]:  # 遍历标注文件中的所有标注信息
        image_file = image_dict.get(ann["image_id"], None)  # 根据标注的图像ID获取对应的图像文件路径
        assert image_file is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )  # 确保每个标注都能找到对应的图像文件
        label_file = os.path.join(gt_dir, ann["file_name"])  # 构建标注文件的完整路径
        segments_info = ann["segments_info"]  # 获取分割信息

        files.append((image_file, label_file, segments_info))  # 将图像文件、标注文件和分割信息组成元组添加到列表中

    assert len(files), "No images found in {}".format(image_dir)  # 确保找到了图像文件
    assert PathManager.isfile(files[0][0]), files[0][0]  # 验证第一个图像文件路径是有效的
    assert PathManager.isfile(files[0][1]), files[0][1]  # 验证第一个标注文件路径是有效的
    return files  # 返回包含所有文件信息的列表


def load_cityscapes_panoptic(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        # 原始数据集路径，例如："~/cityscapes/leftImg8bit/train"
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        # 原始标注文件路径，例如："~/cityscapes/gtFine/cityscapes_panoptic_train"
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        # JSON标注文件路径，例如："~/cityscapes/gtFine/cityscapes_panoptic_train.json"
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.
        # 包含物体类和背景类ID映射的元数据字典，用于将类别ID映射为训练用的连续ID

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
        # 返回Detectron2标准格式的字典列表（参见自定义数据集文档）
    """

    def _convert_category_id(segment_info, meta):
        # 将原始类别ID转换为连续的训练ID
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            # 如果是物体类别，使用thing_dataset_id_to_contiguous_id进行映射
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            # 如果是背景类别，使用stuff_dataset_id_to_contiguous_id进行映射
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
    # 确保JSON标注文件存在，否则提示运行相应脚本生成
    with open(gt_json) as f:
        json_info = json.load(f)  # 加载JSON标注文件
    files = get_cityscapes_panoptic_files(image_dir, gt_dir, json_info)  # 获取数据集文件列表
    ret = []
    for image_file, label_file, segments_info in files:
        sem_label_file = (
            image_file.replace("leftImg8bit", "gtFine").split(".")[0] + "_labelTrainIds.png"
        )  # 构建语义分割标注文件路径
        segments_info = [_convert_category_id(x, meta) for x in segments_info]  # 转换所有分割信息中的类别ID
        ret.append(
            {
                "file_name": image_file,  # 图像文件路径
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
                ),  # 生成图像ID
                "sem_seg_file_name": sem_label_file,  # 语义分割标注文件路径
                "pan_seg_file_name": label_file,  # 全景分割标注文件路径
                "segments_info": segments_info,  # 分割信息（包含转换后的类别ID）
            }
        )
    assert len(ret), f"No images found in {image_dir}!"  # 确保找到了图像文件
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    # 确保语义分割标注文件存在，否则提示运行相应脚本生成
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    # 确保全景分割标注文件存在，否则提示运行相应脚本生成
    return ret  # 返回处理后的数据集信息


_RAW_CITYSCAPES_PANOPTIC_SPLITS = {
    "cityscapes_fine_panoptic_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/gtFine/cityscapes_panoptic_train",
        "cityscapes/gtFine/cityscapes_panoptic_train.json",
    ),
    "cityscapes_fine_panoptic_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/gtFine/cityscapes_panoptic_val",
        "cityscapes/gtFine/cityscapes_panoptic_val.json",
    ),
    # "cityscapes_fine_panoptic_test": not supported yet
}


def register_all_cityscapes_panoptic(root):
    # 初始化元数据字典
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    # 以下元数据将连续ID（从0到物体类别数+背景类别数）映射到它们的名称和颜色
    # 我们必须在"thing_*"和"stuff_*"下复制相同的名称和颜色，因为D2中的可视化函数
    # 由于全景FPN中使用的一些启发式方法，对物体类和背景类的处理方式不同
    # 我们保持相同的命名以便重用现有的可视化函数
    
    # 从CITYSCAPES_CATEGORIES中提取物体类的名称和颜色
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    # 从CITYSCAPES_CATEGORIES中提取背景类的名称和颜色
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    # 将提取的信息添加到元数据字典中
    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # Cityscapes全景分割中有三种类型的ID：
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    # (1) 类别ID：类似于语义分割，它是每个像素的类别ID。由于某些类别在评估中不使用，
    #     类别ID并不总是连续的，因此我们有两组类别ID：
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - 原始类别ID：原始数据集中的类别ID，主要用于评估
    #       - contiguous category id: [0, #classes), in order to train the classifier
    #       - 连续类别ID：范围为[0, 类别数)，用于训练分类器
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (2) 实例ID：用于区分同一类别的不同实例。对于背景类（"stuff"），实例ID始终为0；
    #     对于物体类（"thing"），实例ID从1开始，0保留给被忽略的实例（如群体标注）。
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    # (3) 全景ID：这是一个紧凑的ID，通过category_id * 1000 + instance_id编码类别和实例ID。
    
    # 初始化ID映射字典
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    # 遍历所有Cityscapes类别
    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            # 如果是物体类，添加到thing的ID映射中
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            # 如果是背景类，添加到stuff的ID映射中
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    # 将ID映射添加到元数据中
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    # 遍历数据集的不同分割（训练集和验证集）
    for key, (image_dir, gt_dir, gt_json) in _RAW_CITYSCAPES_PANOPTIC_SPLITS.items():
        # 构建完整的文件路径
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        # 注册数据集到DatasetCatalog
        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes_panoptic(x, y, z, meta)
        )
        # 设置数据集的元数据
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,          # 全景分割标注根目录
            image_root=image_dir,          # 图像根目录
            panoptic_json=gt_json,         # 全景分割JSON文件路径
            gt_dir=gt_dir.replace("cityscapes_panoptic_", ""),  # 真值目录
            evaluator_type="cityscapes_panoptic_seg",  # 评估器类型
            ignore_label=255,              # 忽略的标签值
            label_divisor=1000,            # 标签除数，用于分离类别ID和实例ID
            **meta,                        # 添加其他元数据
        )
