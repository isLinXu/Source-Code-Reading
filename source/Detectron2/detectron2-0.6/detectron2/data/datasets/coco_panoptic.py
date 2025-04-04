# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from .coco import load_coco_json, load_sem_seg

__all__ = ["register_coco_panoptic", "register_coco_panoptic_separated"]


def load_coco_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_coco_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_coco_panoptic_separated(
    name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json
):
    """
    Register a "separated" version of COCO panoptic segmentation dataset named `name`.
    The annotations in this registered dataset will contain both instance annotations and
    semantic annotations, each with its own contiguous ids. Hence it's called "separated".
    # 注册一个名为`name`的"分离"版本的COCO全景分割数据集
    # 该数据集的标注包含实例标注和语义标注，每种标注都有自己的连续ID，因此称为"分离"版本

    It follows the setting used by the PanopticFPN paper:
    # 遵循PanopticFPN论文中使用的设置：

    1. The instance annotations directly come from polygons in the COCO
       instances annotation task, rather than from the masks in the COCO panoptic annotations.
    # 1. 实例标注直接来自COCO实例标注任务中的多边形，而不是来自COCO全景标注中的掩码

       The two format have small differences:
       Polygons in the instance annotations may have overlaps.
       The mask annotations are produced by labeling the overlapped polygons
       with depth ordering.
    # 这两种格式有细微的差别：
    # 实例标注中的多边形可能有重叠
    # 掩码标注是通过对重叠的多边形进行深度排序标注产生的

    2. The semantic annotations are converted from panoptic annotations, where
       all "things" are assigned a semantic id of 0.
       All semantic categories will therefore have ids in contiguous
       range [1, #stuff_categories].
    # 2. 语义标注是从全景标注转换而来，其中所有的"things"类别被赋予语义ID 0
    # 所有语义类别的ID都在连续范围[1, #stuff_categories]内

    This function will also register a pure semantic segmentation dataset
    named ``name + '_stuffonly'``.
    # 该函数还会注册一个纯语义分割数据集，命名为``name + '_stuffonly'``

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
            # 数据集的标识名称，例如："coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
            # 与该数据集相关的额外元数据
        image_root (str): directory which contains all the images
            # 包含所有图像的目录
        panoptic_root (str): directory which contains panoptic annotation images
            # 包含全景标注图像的目录
        panoptic_json (str): path to the json panoptic annotation file
            # 全景标注JSON文件的路径
        sem_seg_root (str): directory which contains all the ground truth segmentation annotations.
            # 包含所有真实分割标注的目录
        instances_json (str): path to the json instance annotation file
            # 实例标注JSON文件的路径
    """
    # 构建全景数据集的名称，添加"_separated"后缀
    panoptic_name = name + "_separated"
    # 注册全景数据集
    DatasetCatalog.register(
        panoptic_name,
        # 使用lambda函数合并实例分割和语义分割数据
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, panoptic_name),  # 加载实例分割数据
            load_sem_seg(sem_seg_root, image_root),                    # 加载语义分割数据
        ),
    )
    # 设置全景数据集的元数据
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,    # 全景标注根目录
        image_root=image_root,          # 图像根目录
        panoptic_json=panoptic_json,    # 全景标注JSON文件路径
        sem_seg_root=sem_seg_root,      # 语义分割根目录
        json_file=instances_json,        # 实例标注JSON文件路径 TODO rename
        evaluator_type="coco_panoptic_seg",  # 评估器类型
        ignore_label=255,               # 忽略的标签值
        **metadata,                     # 其他元数据
    )

    # 构建纯语义分割数据集的名称，添加"_stuffonly"后缀
    semantic_name = name + "_stuffonly"
    # 注册纯语义分割数据集
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root))
    # 设置纯语义分割数据集的元数据
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,      # 语义分割根目录
        image_root=image_root,          # 图像根目录
        evaluator_type="sem_seg",       # 评估器类型
        ignore_label=255,               # 忽略的标签值
        **metadata,                     # 其他元数据
    )


def merge_to_panoptic(detection_dicts, sem_seg_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.
    # 通过使用"file_name"字段匹配条目来合并两个字典，创建全景分割数据集的字典

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        # 目标检测或实例分割的字典列表
        sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.
        # 语义分割的字典列表

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and sem_seg_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
        # 返回字典列表（每个输入图像对应一个字典）：每个字典包含来自detection_dicts和sem_seg_dicts中
        # 对应于同一图像的所有（键，值）对。函数假设不同字典中的相同键具有相同的值。
    """
    results = []  # 初始化结果列表
    # 创建文件名到语义分割条目的映射字典
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    # 确保语义分割字典不为空
    assert len(sem_seg_file_to_entry) > 0

    # 遍历每个检测字典
    for det_dict in detection_dicts:
        # 创建检测字典的浅拷贝
        dic = copy.copy(det_dict)
        # 使用文件名匹配并更新对应的语义分割信息
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        # 将合并后的字典添加到结果列表
        results.append(dic)
    return results  # 返回合并后的结果列表


if __name__ == "__main__":
    """
    Test the COCO panoptic dataset loader.
    测试COCO全景分割数据集加载器

    Usage:
        python -m detectron2.data.datasets.coco_panoptic \
            path/to/image_root path/to/panoptic_root path/to/panoptic_json dataset_name 10

        "dataset_name" can be "coco_2017_train_panoptic", or other
        pre-registered ones
        数据集名称可以是"coco_2017_train_panoptic"或其他预注册的数据集
    """
    # 导入所需的工具和库
    from detectron2.utils.logger import setup_logger  # 导入日志设置工具
    from detectron2.utils.visualizer import Visualizer  # 导入可视化工具
    import detectron2.data.datasets  # noqa # add pre-defined metadata  # 导入预定义的元数据
    import sys  # 导入系统模块
    from PIL import Image  # 导入图像处理库
    import numpy as np  # 导入数值计算库

    # 设置日志记录器
    logger = setup_logger(name=__name__)
    # 验证输入的数据集名称是否在已注册的数据集列表中
    assert sys.argv[4] in DatasetCatalog.list()
    # 获取数据集的元数据信息
    meta = MetadataCatalog.get(sys.argv[4])

    # 加载COCO全景分割数据集的JSON文件
    dicts = load_coco_panoptic_json(sys.argv[3], sys.argv[1], sys.argv[2], meta.as_dict())
    # 记录加载的样本数量
    logger.info("Done loading {} samples.".format(len(dicts)))

    # 创建可视化结果保存目录
    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    # 获取需要可视化的图像数量
    num_imgs_to_vis = int(sys.argv[5])
    # 遍历数据集中的图像
    for i, d in enumerate(dicts):
        # 读取图像并转换为numpy数组
        img = np.array(Image.open(d["file_name"]))
        # 创建可视化器实例
        visualizer = Visualizer(img, metadata=meta)
        # 绘制数据集字典中的信息到图像上
        vis = visualizer.draw_dataset_dict(d)
        # 构建保存路径
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        # 保存可视化结果
        vis.save(fpath)
        # 达到指定的可视化数量后退出循环
        if i + 1 >= num_imgs_to_vis:
            break
