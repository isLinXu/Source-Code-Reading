# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes  # 导入检测框、掩码等基础结构
from detectron2.utils.file_io import PathManager  # 导入文件IO管理工具

from .. import DatasetCatalog, MetadataCatalog  # 导入数据集目录和元数据目录

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
该文件包含了将COCO格式注释解析为Detectron2格式字典的函数
"""


logger = logging.getLogger(__name__)  # 初始化日志记录器

__all__ = ["load_coco_json", "load_sem_seg", "convert_to_coco_json", "register_coco_instances"]  # 定义可导出的函数列表


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.
    加载COCO实例标注格式的json文件。
    目前支持实例检测、实例分割和人体关键点标注。

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        COCO实例标注格式的json文件的完整路径。
        image_root (str or path-like): the directory where the images in this json file exists.
        json文件中图像所在的目录路径。
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:
            数据集的名称（例如：coco_2017_train）。
            当提供此参数时，函数还会执行以下操作：

            * Put "thing_classes" into the metadata associated with this dataset.
            * 将"thing_classes"添加到与该数据集关联的元数据中。
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.
            * 将类别ID映射到连续范围（标准数据集格式所需），
              并将"thing_dataset_id_to_contiguous_id"添加到与该数据集关联的元数据中。

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
            通常应该提供此选项，除非用户需要手动加载原始json内容并进行更多处理。
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.
            需要加载到数据集字典中的额外标注键列表（除了"iscrowd"、"bbox"、"keypoints"、
            "category_id"、"segmentation"之外）。这些键的值将按原样返回。
            例如，DensePose标注就是通过这种方式加载的。

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.
        当dataset_name不为None时，返回Detectron2标准数据集字典格式的字典列表
        （参见`使用自定义数据集</tutorials/datasets.html>`_）。
        如果dataset_name为None，返回的category_ids可能不连续，
        且可能不符合Detectron2标准格式。

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
        1. 此函数不读取图像文件。
           结果中不包含"image"字段。
    """
    from pycocotools.coco import COCO  # 导入COCO API

    timer = Timer()  # 创建计时器
    json_file = PathManager.get_local_path(json_file)  # 获取json文件的本地路径
    with contextlib.redirect_stdout(io.StringIO()):  # 重定向标准输出以抑制COCO API的输出
        coco_api = COCO(json_file)  # 初始化COCO API对象
    if timer.seconds() > 1:  # 如果加载时间超过1秒
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))  # 记录加载时间

    id_map = None  # 初始化ID映射
    if dataset_name is not None:  # 如果提供了数据集名称
        meta = MetadataCatalog.get(dataset_name)  # 获取数据集的元数据
        cat_ids = sorted(coco_api.getCatIds())  # 获取并排序所有类别ID
        cats = coco_api.loadCats(cat_ids)  # 加载类别信息
        # The categories in a custom json file may not be sorted.
        # 自定义json文件中的类别可能未排序
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]  # 提取并排序类别名称
        meta.thing_classes = thing_classes  # 将类别名称列表添加到元数据中

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).
        # 在COCO中，某些类别ID被人为移除，按照惯例这些ID总是被忽略。
        # 我们处理COCO的ID问题，将类别ID转换为连续的ID，范围在[0, 80)内。

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        # 这通过查看json中的"categories"字段来工作，因此
        # 如果用户自己的json也有不连续的ID，我们也会
        # 应用这个映射，但会打印一个警告。
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):  # 检查类别ID是否连续
            if "coco" not in dataset_name:  # 如果不是COCO数据集
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}  # 创建类别ID到连续ID的映射
        meta.thing_dataset_id_to_contiguous_id = id_map  # 将ID映射添加到元数据中

    # sort indices for reproducible results
    # 对索引进行排序以获得可重现的结果
    img_ids = sorted(coco_api.imgs.keys())  # 获取并排序所有图像ID
    # imgs is a list of dicts, each looks something like:
    # imgs是字典列表，每个字典的格式如下：
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
            # 警告：json文件包含的标注数量与匹配到图像的标注数量不一致
        )

    if "minival" not in json_file:  # 如果不是minival数据集
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        # COCO2014的valminusminival和minival标注中存在这个bug。
        # 但是有问题的标注比例很小，不影响准确性。
        # 因此我们显式地将它们列入白名单。
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]  # 获取所有标注ID
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )  # 确保标注ID是唯一的

    imgs_anns = list(zip(imgs, anns))  # 将图像和标注数据配对
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))  # 记录加载的图像数量

    dataset_dicts = []  # 初始化数据集字典列表

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])  # 定义需要保留的标注键

    num_instances_without_valid_segmentation = 0  # 初始化无效分割实例计数器

    for (img_dict, anno_dict_list) in imgs_anns:  # 遍历图像和标注数据对
        record = {}  # 创建记录字典
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])  # 设置图像文件路径
        record["height"] = img_dict["height"]  # 设置图像高度
        record["width"] = img_dict["width"]  # 设置图像宽度
        image_id = record["image_id"] = img_dict["id"]  # 设置图像ID

        objs = []  # 初始化对象列表
        for anno in anno_dict_list:  # 遍历图像的所有标注
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            # 检查标注中的image_id是否与当前处理的image_id相同
            # 只有在数据解析逻辑或标注文件有问题时才会失败

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            # 原始的COCO valminusminival2014和minival2014标注文件
            # 实际上包含bug，在某些使用COCO API的方式下
            # 可能会触发这个断言
            assert anno["image_id"] == image_id  # 确保标注的image_id与当前图像ID匹配

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'  # 不支持COCO json文件中的"ignore"标记

            obj = {key: anno[key] for key in ann_keys if key in anno}  # 从标注中提取需要的字段
            if "bbox" in obj and len(obj["bbox"]) == 0:  # 检查边界框是否为空
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                    # 图像的标注包含空的边界框值，不符合COCO格式
                )

            segm = anno.get("segmentation", None)  # 获取分割标注
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):  # 如果分割标注是RLE格式
                    if isinstance(segm["counts"], list):  # 如果RLE编码是列表格式
                        # convert to compressed RLE
                        # 转换为压缩的RLE格式
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:  # 如果分割标注是多边形格式
                    # filter out invalid polygons (< 3 points)
                    # 过滤掉无效的多边形（少于3个点）
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:  # 如果过滤后没有有效的多边形
                        num_instances_without_valid_segmentation += 1  # 增加无效分割实例计数
                        continue  # ignore this instance  # 忽略这个实例
                obj["segmentation"] = segm  # 保存处理后的分割标注

            keypts = anno.get("keypoints", None)  # 获取关键点标注
            if keypts:  # list[int]  # 如果存在关键点标注
                for idx, v in enumerate(keypts):  # 遍历关键点坐标
                    if idx % 3 != 2:  # 对于x和y坐标（不是可见性标志）
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        # COCO的分割坐标是[0, H或W]范围内的浮点数，
                        # 但关键点坐标是[0, H-1或W-1]范围内的整数
                        # 因此我们假设坐标是"像素索引"
                        keypts[idx] = v + 0.5  # 添加0.5将整数索引转换为浮点坐标
                obj["keypoints"] = keypts  # 保存处理后的关键点标注

            obj["bbox_mode"] = BoxMode.XYWH_ABS  # 设置边界框模式为绝对坐标的XYWH格式
            if id_map:  # 如果存在类别ID映射
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]  # 将类别ID映射到连续ID
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)  # 将处理后的对象添加到列表中
        record["annotations"] = objs  # 将对象列表添加到记录中
        dataset_dicts.append(record)  # 将记录添加到数据集字典列表中

    if num_instances_without_valid_segmentation > 0:  # 如果存在无效的分割实例
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
            # 警告：过滤掉了没有有效分割的实例。
            # 数据集生成过程中可能存在问题，请仔细检查文档。
        )
    return dataset_dicts  # 返回处理后的数据集字典列表


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.
    加载语义分割数据集。所有在"gt_root"下具有"gt_ext"扩展名的文件被视为真值标注，
    所有在"image_root"下具有"image_ext"扩展名的文件被视为输入图像。
    真值和输入图像通过相对于"gt_root"和"image_root"的文件路径进行匹配，不考虑文件扩展名。
    这适用于COCO以及其他一些数据集。

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
            真值语义分割文件的完整路径。语义分割标注以图像形式存储，
            像素值为整数，表示对应的语义标签。
        image_root (str): the directory where the input images are.
            输入图像所在的目录。
        gt_ext (str): file extension for ground truth annotations.
            真值标注的文件扩展名。
        image_ext (str): file extension for input images.
            输入图像的文件扩展名。

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.
            返回detectron2标准格式的字典列表，不包含实例级别的标注。

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
        1. 此函数不读取图像和真值文件。
           结果中不包含"image"和"sem_seg"字段。
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    # 使用交集，以便val2017_100标注可以与val2017图像顺利运行
    if len(input_files) != len(gt_files):  # 如果输入图像和真值标注数量不一致
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]  # 获取输入图像的基本名称
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]  # 获取真值标注的基本名称
        intersect = list(set(input_basenames) & set(gt_basenames))  # 计算基本名称的交集
        # sort, otherwise each worker may obtain a list[dict] in different order
        # 排序，否则每个工作进程可能会得到不同顺序的字典列表
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))  # 记录使用交集文件的数量
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]  # 更新输入图像文件列表
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]  # 更新真值标注文件列表

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
        # 记录从指定目录加载的语义分割图像数量
    )

    dataset_dicts = []  # 初始化数据集字典列表
    for (img_path, gt_path) in zip(input_files, gt_files):  # 遍历配对的图像和标注文件
        record = {}  # 创建记录字典
        record["file_name"] = img_path  # 设置图像文件路径
        record["sem_seg_file_name"] = gt_path  # 设置语义分割标注文件路径
        dataset_dicts.append(record)  # 将记录添加到数据集字典列表

    return dataset_dicts  # 返回处理后的数据集字典列表


def convert_to_coco_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.
    将detectron2标准格式的实例检测/分割或关键点检测数据集转换为COCO json格式。

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset
    通用数据集描述可以在这里找到：
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data
    COCO数据格式描述可以在这里找到：
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
            源数据集的名称
            必须在DatastCatalog中注册并符合detectron2的标准格式。
            必须具有相应的元数据"thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
        coco_dict: 可序列化的COCO json格式字典
    """

    # 从数据集目录获取数据集字典和元数据
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    # 为COCO反向映射类别ID
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        # 创建从连续ID到原始COCO ID的反向映射
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        # 如果没有ID映射，则直接返回原ID
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    # 构建COCO格式的类别列表
    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    # 初始化COCO格式的图像和标注列表
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        # 构建COCO格式的图像信息
        coco_image = {
            "id": image_dict.get("image_id", image_id),  # 获取图像ID，如果没有则使用枚举索引
            "width": int(image_dict["width"]),  # 图像宽度
            "height": int(image_dict["height"]),  # 图像高度
            "file_name": str(image_dict["file_name"]),  # 图像文件名
        }
        coco_images.append(coco_image)

        # 获取每张图像的标注列表
        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            # 创建一个只包含COCO字段的新字典
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            # COCO要求：对于轴对齐的框使用XYWH格式，对于旋转的框使用XYWHA格式
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            # 获取边界框的模式并进行转换
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            # COCO要求：实例面积
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                # 通过计算像素点来计算实例面积
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                # TODO: 检查分割类型：RLE（游程编码）、二进制掩码或多边形
                if isinstance(segmentation, list):
                    # 如果是多边形格式的分割
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    # 如果是RLE格式的分割
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                # 使用边界框计算面积
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    # 对于轴对齐的框，转换为XYXY格式计算面积
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    # 对于旋转的框，直接计算面积
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        # COCO的分割坐标是[0, H或W]范围内的浮点数，
                        # 但关键点坐标是[0, H-1或W-1]范围内的整数
                        # 为了保持COCO格式的一致性，我们减去0.5
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    # 如果提供了关键点数量，直接使用
                    num_keypoints = annotation["num_keypoints"]
                else:
                    # 否则计算可见关键点的数量（每3个值中的第3个值大于0表示可见）
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            # COCO要求：
            #   将标注链接到图像
            #   "id"字段必须从1开始
            coco_annotation["id"] = len(coco_annotations) + 1  # 标注ID从1开始递增
            coco_annotation["image_id"] = coco_image["id"]  # 关联到对应的图像
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]  # 边界框坐标保留3位小数
            coco_annotation["area"] = float(area)  # 实例面积
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))  # 是否为群体标注
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))  # 类别ID

            # Add optional fields
            # 添加可选字段
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints  # 关键点坐标
                coco_annotation["num_keypoints"] = num_keypoints  # 可见关键点数量

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]  # 分割信息
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        # 确保RLE编码是JSON可序列化的字符串
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)  # 将标注添加到列表中

    # 记录转换完成的信息
    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    # 创建COCO数据集的基本信息
    info = {
        "date_created": str(datetime.datetime.now()),  # 创建时间
        "description": "Automatically generated COCO json file for Detectron2.",  # 描述信息
    }
    # 构建完整的COCO格式字典
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations  # 只有在有标注时才添加标注字段
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    将数据集转换为COCO格式并保存为json文件。
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.
    数据集名称必须在DatasetCatalog中注册并符合detectron2的标准格式。

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
            配置文件中对目录的引用
            必须在DatasetCatalog中注册并符合detectron2的标准格式
        output_file: path of json file that will be saved to
                    将要保存的json文件路径
        allow_cached: if json file is already present then skip conversion
                     如果json文件已存在则跳过转换
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data
    # TODO: 数据集或转换脚本可能会改变，
    # 使用校验和来验证缓存数据会很有用

    # 创建输出文件的目录
    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):  # 使用文件锁确保并发安全
        if PathManager.exists(output_file) and allow_cached:
            # 如果允许使用缓存且文件已存在，则使用缓存
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            # 否则进行转换
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            # 保存转换后的COCO格式标注
            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"  # 使用临时文件
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)  # 原子操作替换文件


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    注册COCO json标注格式的数据集，用于实例检测、实例分割和关键点检测。
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).
    （即http://cocodataset.org/#format-data中的类型1和类型2。
    数据集中的`instances*.json`和`person_keypoints*.json`文件）。

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.
    这是如何注册新数据集的示例。
    你可以参照这个函数来注册新的数据集。

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
                    数据集的标识名称，例如"coco_2014_train"。
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
                        与该数据集关联的额外元数据。可以留空为空字典。
        json_file (str): path to the json instance annotation file.
                        json实例标注文件的路径。
        image_root (str or path-like): directory which contains all the images.
                                      包含所有图像的目录。
    """
    # 类型检查
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    # 1. 注册一个返回字典的函数
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    # 2. 可选地，添加关于这个数据集的元数据，
    # 因为这些信息在评估、可视化或日志记录时可能有用
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.
    测试COCO json数据集加载器。

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
        "dataset_name"可以是"coco_2014_minival_100"或其他预注册的数据集名称
    """
    # 导入所需的工具和库
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata  # 导入预定义的元数据
    import sys

    # 设置日志记录器
    logger = setup_logger(name=__name__)
    # 确保提供的数据集名称在预注册的数据集列表中
    assert sys.argv[3] in DatasetCatalog.list()
    # 获取数据集的元数据信息
    meta = MetadataCatalog.get(sys.argv[3])

    # 加载COCO格式的JSON数据集
    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    # 记录加载的样本数量
    logger.info("Done loading {} samples.".format(len(dicts)))

    # 创建用于保存可视化结果的目录
    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    # 遍历数据集中的每个样本
    for d in dicts:
        # 读取图像文件并转换为numpy数组
        img = np.array(Image.open(d["file_name"]))
        # 创建可视化器实例
        visualizer = Visualizer(img, metadata=meta)
        # 绘制数据集字典中的标注信息
        vis = visualizer.draw_dataset_dict(d)
        # 构建输出文件路径
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        # 保存可视化结果
        vis.save(fpath)
