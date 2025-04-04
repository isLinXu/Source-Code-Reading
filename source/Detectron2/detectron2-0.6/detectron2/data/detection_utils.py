# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import logging
import numpy as np
from typing import List, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

# 导入detectron2中的数据结构相关类和函数
from detectron2.structures import (
    BitMasks,      # 用于表示二值掩码
    Boxes,         # 用于表示边界框
    BoxMode,       # 用于表示边界框的不同格式(XYXY_ABS, XYWH_ABS等)
    Instances,     # 用于表示图像中的实例集合
    Keypoints,     # 用于表示关键点
    PolygonMasks,  # 用于表示多边形掩码
    RotatedBoxes,  # 用于表示旋转边界框
    polygons_to_bitmask,  # 用于将多边形转换为二值掩码
)
# 导入文件IO工具
from detectron2.utils.file_io import PathManager

# 导入数据增强相关模块
from . import transforms as T
# 导入数据集元数据相关模块
from .catalog import MetadataCatalog

# 定义模块的公共接口
__all__ = [
    "SizeMismatchError",              # 图像尺寸不匹配错误
    "convert_image_to_rgb",           # 将图像转换为RGB格式
    "check_image_size",               # 检查图像尺寸
    "transform_proposals",            # 转换候选框
    "transform_instance_annotations", # 转换实例标注
    "annotations_to_instances",       # 将标注转换为实例
    "annotations_to_instances_rotated", # 将标注转换为旋转实例
    "build_augmentation",            # 构建数据增强
    "build_transform_gen",           # 构建转换生成器
    "create_keypoint_hflip_indices", # 创建关键点水平翻转索引
    "filter_empty_instances",         # 过滤空实例
    "read_image",                    # 读取图像
]


class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    当加载的图像与标注的宽度/高度不匹配时抛出此错误。
    """


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
# RGB和YUV颜色空间转换矩阵，遵循BT.601标准
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag，EXIF中表示图像方向的标签值


def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.
    将PIL图像转换为指定格式的numpy数组。

    Args:
        image (PIL.Image): a PIL image
                          PIL格式的图像对象
        format (str): the format of output image
                     输出图像的格式

    Returns:
        (np.ndarray): also see `read_image`
                      返回numpy数组格式的图像，详见`read_image`函数
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        # PIL只支持RGB格式，所以先转换为RGB，然后再根据需要调整通道顺序
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"  # 对于BGR和YUV格式，先转换为RGB
        image = image.convert(conversion_format)
    image = np.asarray(image)  # 将PIL图像转换为numpy数组
    
    # PIL squeezes out the channel dimension for "L", so make it HWC
    # 对于灰度图像("L"格式)，PIL会压缩掉通道维度，需要手动添加通道维度
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    # 处理PIL不直接支持的格式
    elif format == "BGR":
        # flip channels if needed
        # 将RGB转换为BGR，通过反转通道顺序实现
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0  # 归一化到[0,1]范围
        image = np.dot(image, np.array(_M_RGB2YUV).T)  # 使用转换矩阵将RGB转换为YUV

    return image


def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.
    将给定格式的图像转换为RGB格式。

    Args:
        image (np.ndarray or Tensor): an HWC image
                                     HWC格式的图像，可以是numpy数组或PyTorch张量
        format (str): the format of input image, also see `read_image`
                     输入图像的格式，详见`read_image`函数

    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
                      返回(H,W,3)格式的RGB图像，像素值范围为0-255，可以是float或uint8类型
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # 如果是PyTorch张量，转换为numpy数组
    if format == "BGR":
        image = image[:, :, [2, 1, 0]]  # BGR转RGB，通过重排通道顺序实现
    elif format == "YUV-BT.601":
        image = np.dot(image, np.array(_M_YUV2RGB).T)  # YUV转RGB，使用转换矩阵
        image = image * 255.0  # 将[0,1]范围转换为[0,255]范围
    else:
        if format == "L":
            image = image[:, :, 0]  # 对于灰度图像，取第一个通道
        image = image.astype(np.uint8)  # 转换为uint8类型
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))  # 使用PIL进行格式转换
    return image


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.
    正确应用EXIF方向信息。

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`
    这段代码是为了解决Pillow库中的一个bug，特别是在使用`tobytes`方法时会出现错误。

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527
    函数基于labelme和Pillow库的实现。

    Args:
        image (PIL.Image): a PIL image
                          PIL格式的图像对象

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
                    应用EXIF方向信息后的PIL图像
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()  # 尝试获取图像的EXIF信息
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None  # 如果获取EXIF信息失败，设置为None

    if exif is None:
        return image  # 如果没有EXIF信息，直接返回原图

    orientation = exif.get(_EXIF_ORIENT)  # 获取EXIF中的方向信息

    # 定义EXIF方向值对应的图像变换方法
    method = {
        2: Image.FLIP_LEFT_RIGHT,   # 水平翻转
        3: Image.ROTATE_180,         # 旋转180度
        4: Image.FLIP_TOP_BOTTOM,    # 垂直翻转
        5: Image.TRANSPOSE,          # 转置（主对角线翻转）
        6: Image.ROTATE_270,         # 旋转270度
        7: Image.TRANSVERSE,         # 转置（副对角线翻转）
        8: Image.ROTATE_90,          # 旋转90度
    }.get(orientation)

    if method is not None:
        return image.transpose(method)  # 如果存在对应的变换方法，应用变换
    return image  # 如果没有需要应用的变换，返回原图


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])  # 获取图像的实际宽高
        expected_wh = (dataset_dict["width"], dataset_dict["height"])  # 获取数据集字典中指定的宽高
        if not image_wh == expected_wh:  # 如果实际宽高与指定宽高不匹配
            raise SizeMismatchError(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation."
            )

    # To ensure bbox always remap to original image size
    # 确保边界框总是映射到原始图像尺寸
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]  # 如果字典中没有宽度信息，使用图像实际宽度
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]  # 如果字典中没有高度信息，使用图像实际高度


def transform_proposals(dataset_dict, image_shape, transforms, *, proposal_topk, min_box_size=0):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        proposal_topk (int): only keep top-K scoring proposals
        min_box_size (int): proposals with either side smaller than this
            threshold are removed

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_size)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.
    对单个实例的边界框、分割掩码和关键点标注应用变换。

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.
    将使用`transforms.apply_box`处理边界框，使用`transforms.apply_coords`处理分割多边形和关键点。
    如果需要针对特定数据结构的特殊处理，需要实现自己的版本。

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
                          单个实例的标注字典，将被原地修改。
        transforms (TransformList or list[Transform]):
                    变换列表或变换对象列表
        image_size (tuple): the height, width of the transformed image
                           变换后图像的高度和宽度
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
                                               关键点水平翻转的索引映射

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
            返回输入字典，其中的"bbox"、"segmentation"、"keypoints"字段已根据变换进行更新。
            "bbox_mode"字段将被设置为XYXY_ABS（绝对坐标）格式。
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)  # 将变换列表转换为TransformList对象
    # bbox is 1d (per-instance bounding box)
    # 处理边界框（每个实例一个边界框）
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)  # 将边界框转换为XYXY_ABS格式
    # clip transformed bbox to image size
    # 将变换后的边界框裁剪到图像范围内
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)  # 应用变换并确保坐标非负
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])  # 确保边界框不超出图像范围
    annotation["bbox_mode"] = BoxMode.XYXY_ABS  # 更新边界框模式为绝对坐标

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        # 处理分割标注（每个实例包含一个或多个多边形）
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            # 处理多边形格式的分割标注
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]  # 将多边形转换为numpy数组
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)  # 应用变换并重新整形
            ]
        elif isinstance(segm, dict):
            # RLE
            # 处理RLE（游程编码）格式的分割标注
            mask = mask_util.decode(segm)  # 解码RLE格式
            mask = transforms.apply_segmentation(mask)  # 应用变换
            assert tuple(mask.shape[:2]) == image_size  # 确保掩码尺寸与图像匹配
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        # 处理关键点标注
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation


def transform_keypoint_annotations(keypoints, transforms, image_size, keypoint_hflip_indices=None):
    """
    Transform keypoint annotations of an image.
    If a keypoint is transformed out of image boundary, it will be marked "unlabeled" (visibility=0)
    转换图像的关键点标注。如果关键点被变换到图像边界外，将被标记为"未标注"（可见性=0）。

    Args:
        keypoints (list[float]): Nx3 float in Detectron2's Dataset format.
            Each point is represented by (x, y, visibility).
            Nx3大小的浮点数列表，使用Detectron2的数据集格式。
            每个点由(x, y, visibility)表示。
        transforms (TransformList):
                    变换列表对象
        image_size (tuple): the height, width of the transformed image
                           变换后图像的高度和宽度
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
            When `transforms` includes horizontal flip, will use the index
            mapping to flip keypoints.
            关键点水平翻转的索引映射，当变换包含水平翻转时使用。
    """
    # (N*3,) -> (N, 3)
    # 将关键点数组重塑为(N,3)的形状
    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
    keypoints_xy = transforms.apply_coords(keypoints[:, :2])  # 对关键点坐标应用变换

    # Set all out-of-boundary points to "unlabeled"
    # 将所有超出图像边界的点标记为"未标注"
    inside = (keypoints_xy >= np.array([0, 0])) & (keypoints_xy <= np.array(image_size[::-1]))
    inside = inside.all(axis=1)  # 检查每个点是否在图像边界内
    keypoints[:, :2] = keypoints_xy  # 更新关键点坐标
    keypoints[:, 2][~inside] = 0  # 将超出边界的点标记为不可见

    # This assumes that HorizFlipTransform is the only one that does flip
    # 假设HorizFlipTransform是唯一执行翻转的变换
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1

    # Alternative way: check if probe points was horizontally flipped.
    # 另一种方法：检查探测点是否被水平翻转
    # probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
    # probe_aug = transforms.apply_coords(probe.copy())
    # do_hflip = np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0])  # noqa

    # If flipped, swap each keypoint with its opposite-handed equivalent
    # 如果进行了翻转，将每个关键点与其对应的对称点交换
    if do_hflip:
        if keypoint_hflip_indices is None:
            raise ValueError("Cannot flip keypoints without providing flip indices!")
        if len(keypoints) != len(keypoint_hflip_indices):
            raise ValueError(
                "Keypoint data has {} points, but metadata "
                "contains {} points!".format(len(keypoints), len(keypoint_hflip_indices))
            )
        keypoints = keypoints[np.asarray(keypoint_hflip_indices, dtype=np.int32), :]

    # Maintain COCO convention that if visibility == 0 (unlabeled), then x, y = 0
    # 保持COCO约定：如果可见性为0（未标注），则x,y坐标也设为0
    keypoints[keypoints[:, 2] == 0] = 0
    return keypoints


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target


def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only
    从数据集字典中的实例标注创建一个用于模型的Instances对象。
    与annotations_to_instances相比，此函数仅用于旋转框。

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
            一个图像中的实例标注列表，每个元素对应一个实例。
        image_size (tuple): height, width
            图像尺寸，包含高度和宽度。

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
            包含"gt_boxes"（旋转边界框）和"gt_classes"（类别）字段（如果能从annos中获取）。
            这是内置模型所期望的格式。
    """
    # 提取所有边界框
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)  # 创建Instances对象
    boxes = target.gt_boxes = RotatedBoxes(boxes)  # 创建旋转框对象并设置为目标的边界框
    boxes.clip(image_size)  # 将边界框裁剪到图像范围内

    # 提取类别ID并转换为张量
    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes  # 设置目标类别

    return target  # 返回处理完成的Instances对象


def filter_empty_instances(
    instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False
):
    """
    Filter out empty instances in an `Instances` object.
    过滤掉Instances对象中的空实例。

    Args:
        instances (Instances): 包含实例的对象
        by_box (bool): whether to filter out instances with empty boxes
                      是否过滤掉空边界框的实例
        by_mask (bool): whether to filter out instances with empty masks
                      是否过滤掉空掩码的实例
        box_threshold (float): minimum width and height to be considered non-empty
                             被认为是非空的最小宽度和高度阈值
        return_mask (bool): whether to return boolean mask of filtered instances
                          是否返回过滤后实例的布尔掩码

    Returns:
        Instances: the filtered instances.
                  过滤后的实例对象
        tensor[bool], optional: boolean mask of filtered instances
                               过滤实例的布尔掩码（可选）
    """
    # 确保至少启用了一种过滤方式
    assert by_box or by_mask
    r = []  # 存储过滤条件的列表
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))  # 添加非空边界框的过滤条件
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())  # 添加非空掩码的过滤条件

    # TODO: can also filter visible keypoints
    # 还可以过滤可见的关键点

    if not r:  # 如果没有过滤条件
        return instances
    m = r[0]  # 获取第一个过滤条件
    for x in r[1:]:  # 合并所有过滤条件
        m = m & x  # 使用与运算组合多个条件
    if return_mask:  # 如果需要返回掩码
        return instances[m], m  # 返回过滤后的实例和掩码
    return instances[m]  # 仅返回过滤后的实例


def create_keypoint_hflip_indices(dataset_names: Union[str, List[str]]) -> List[int]:
    """
    Args:
        dataset_names: list of dataset names
                      数据集名称列表

    Returns:
        list[int]: a list of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
        返回一个大小为关键点数量的整数列表，存储水平翻转后的关键点索引。
    """
    # 如果输入是字符串，转换为列表
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # 检查数据集之间的关键点名称和翻转映射是否一致
    check_metadata_consistency("keypoint_names", dataset_names)
    check_metadata_consistency("keypoint_flip_map", dataset_names)

    # 获取第一个数据集的元数据
    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names  # 获取关键点名称列表
    # TODO flip -> hflip
    flip_map = dict(meta.keypoint_flip_map)  # 获取关键点翻转映射
    flip_map.update({v: k for k, v in flip_map.items()})  # 添加反向映射
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]  # 获取翻转后的名称
    flip_indices = [names.index(i) for i in flipped_names]  # 将名称转换为索引
    return flip_indices


def gen_crop_transform_with_instance(crop_size, image_size, instance):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.
    生成一个CropTransform对象，使裁剪区域包含给定实例的中心。

    Args:
        crop_size (tuple): h, w in pixels
                          裁剪尺寸（高度和宽度，以像素为单位）
        image_size (tuple): h, w
                          图像尺寸（高度和宽度）
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
            一个实例的标注字典，使用Detectron2的数据集格式。
    """
    # 将裁剪尺寸转换为numpy数组
    crop_size = np.asarray(crop_size, dtype=np.int32)
    # 将边界框转换为绝对坐标系统
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    # 计算边界框的中心点坐标
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    # 确保边界框中心在图像内
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    # 确保裁剪尺寸不大于图像尺寸
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    # 计算裁剪区域的最小和最大边界
    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    # 在有效范围内随机选择裁剪起点
    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    # 返回裁剪变换对象
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.
    检查数据集之间的元数据是否一致。

    Args:
        key (str): a metadata key
                  元数据键名
        dataset_names (list[str]): a list of dataset names
                                  数据集名称列表

    Raises:
        AttributeError: if the key does not exist in the metadata
                       如果元数据中不存在该键名
        ValueError: if the given datasets do not have the same metadata values defined by key
                   如果给定的数据集在该键名下的元数据值不一致
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    从配置创建默认的数据增强列表。
    目前包括调整大小和翻转操作。

    Returns:
        list[Augmentation]: 数据增强操作列表
    """
    # 根据是否是训练模式选择不同的配置
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN  # 训练时的最小尺寸
        max_size = cfg.INPUT.MAX_SIZE_TRAIN  # 训练时的最大尺寸
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING  # 训练时的采样方式
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST  # 测试时的最小尺寸
        max_size = cfg.INPUT.MAX_SIZE_TEST  # 测试时的最大尺寸
        sample_style = "choice"  # 测试时固定使用choice采样方式
    # 创建调整最短边的变换
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    # 如果是训练模式且启用了随机翻转
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",  # 是否启用水平翻转
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",      # 是否启用垂直翻转
            )
        )
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""
