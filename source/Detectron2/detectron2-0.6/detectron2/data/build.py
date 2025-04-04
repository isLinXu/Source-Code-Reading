# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
import operator
import pickle
import torch
import torch.utils.data as torchdata
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import _log_api_usage, log_first_n

from .catalog import DatasetCatalog, MetadataCatalog
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset
from .dataset_mapper import DatasetMapper
from .detection_utils import check_metadata_consistency
from .samplers import (
    InferenceSampler,
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)


"""
# 该文件包含构建用于训练或测试的数据加载器的默认逻辑。
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_batch_data_loader",
    "build_detection_train_loader",
    "build_detection_test_loader",
    "get_detection_dataset_dicts",
    "load_proposals_into_dataset",
    "print_instances_class_histogram",
]


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    过滤掉没有标注或仅包含人群标注的图像
    (即，没有非人群标注的图像)。
    在COCO数据集上的常见训练时间预处理。
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        # 检查是否存在非crowd标注（iscrowd=0表示有效标注）
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # 计算可见关键点数量（v>0表示可见）
        # 关键点格式为[x1,y1,v1, x2,y2,v2,...]，每3个元素为一个关键点
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()  # 取所有关键点的可见性标志
            for ann in annotations
            if "keypoints" in ann
        )

    dataset_dicts = [
        x for x in dataset_dicts if visible_keypoints_in_image(x) >= min_keypoints_per_image
    ]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with fewer than {} keypoints.".format(
            num_before - num_after, min_keypoints_per_image
        )
    )
    return dataset_dicts


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")  # 加载提议文件

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)  # 统一键名

    # 创建图像ID到提议索引的映射
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}

    # 处理边界框模式（默认XYXY绝对坐标）
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # 获取当前图像对应的提议索引
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # 按objectness分数降序排序
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)  # 创建直方图分箱
    histogram = np.zeros((num_classes,), dtype=np.int)  # 注意：np.int已过时，建议使用np.int_

    for entry in dataset_dicts:
        annos = entry["annotations"]
        # 提取非crowd标注的类别ID
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int
        )
        if len(classes):
            # 验证类别ID有效性
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)  # 控制表格列数

    def short_name(x):
        # 缩短长类名显示（用于LVIS等数据集）
        if len(x) > 13:
            return x[:11] + ".."
        return x

    # 准备表格数据
    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))  # 补齐空位
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])

    # 使用tabulate生成格式化的表格
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",  # 使用GitHub兼容的格式
        numalign="left",
        stralign="center",
    )
    log_first_n(  # 限制日志输出频率
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def get_detection_dataset_dicts(
    names,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    # 从DatasetCatalog获取数据集
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    # 验证数据集非空
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # 将提议加载到各个数据集
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        return torchdata.ConcatDataset(dataset_dicts)  # 合并多个数据集

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))  # 展平列表

    # 根据条件过滤数据
    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if check_consistency and has_instances:
        try:
            # 检查元数据一致性并打印类别分布
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # 数据集没有类别信息时跳过
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    # 获取分布式训练中的总进程数（GPU数量）
    world_size = get_world_size()  # 获取分布式训练中的world size
    # 验证总batch size必须能被GPU数量整除
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    # 计算每个GPU的batch size
    batch_size = total_batch_size // world_size  # 计算单GPU的batch size

    # 处理可迭代数据集的情况
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        # 将普通数据集转换为可迭代数据集
        dataset = ToIterableDataset(dataset, sampler)  # 转换为可迭代数据集

    # 宽高比分组处理逻辑
    if aspect_ratio_grouping:
        # 创建基础数据加载器（不进行批处理）
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # 不进行批处理
            worker_init_fn=worker_init_reset_seed,
        )
        # 添加宽高比分组功能
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)  # 按宽高比分组
        # 处理collate函数
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        # 标准数据加载方式（使用PyTorch默认批处理）
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,  # 丢弃最后不足一个batch的数据
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
        )


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    # 获取训练数据集
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,  # 过滤空标注
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,  # 关键点检测的最小关键点数
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])  # 记录API使用情况

    # 初始化数据映射器
    if mapper is None:
        mapper = DatasetMapper(cfg, True)  # 创建默认的数据映射器

    # 初始化采样器
    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        # 根据配置选择不同的采样策略
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))  # 基础训练采样器
        elif sampler_name == "RepeatFactorTrainingSampler":
            # 基于类别频率的重复采样
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        elif sampler_name == "RandomSubsetTrainingSampler":
            # 随机子集采样
            sampler = RandomSubsetTrainingSampler(len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # 返回训练数据加载器配置字典
    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,  # 从配置获取总batch size
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.
            No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    构建目标检测训练数据加载器，主要特性包括：
    - 支持数据映射转换
    - 可配置的采样策略
    - 分布式训练支持
    - 宽高比分组优化
    """
    # 处理列表类型数据集输入
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)  # 将字典列表转换为数据集对象
    
    # 应用数据映射转换
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)  # 使用mapper进行数据预处理和增强

    # 验证数据集与采样器的兼容性
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        # 设置默认训练采样器
        if sampler is None:
            sampler = TrainingSampler(len(dataset))  # 创建无限随机采样器
        # 验证采样器类型
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    
    # 构建最终批处理数据加载器
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    从配置生成测试数据加载器参数，特点包括：
    - 独立处理每个测试集
    - 自动加载提案文件（当配置开启时）
    - 使用验证阶段的映射器
    """
    # 统一数据集名为列表格式
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    # 获取测试数据集字典
    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,  # 测试时不过滤空标注
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] 
            for x in dataset_name
        ] if cfg.MODEL.LOAD_PROPOSALS else None,  # 按需加载提案文件
    )
    
    # 初始化测试用数据映射器
    if mapper is None:
        mapper = DatasetMapper(cfg, False)  # 创建无数据增强的映射器
    
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS  # 从配置获取worker数量
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, sampler=None, num_workers=0, collate_fn=None):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        num_workers (int): number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    构建测试数据加载器的核心实现：
    - 固定batch size为1
    - 使用推理采样器保证数据完整性
    - 最小化数据转换开销
    """
    # 转换列表类型数据集
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)  # 保持数据引用
    
    # 应用测试阶段数据映射
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)  # 应用测试预处理

    # 验证可迭代数据集的采样器设置
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        # 设置默认推理采样器
        if sampler is None:
            sampler = InferenceSampler(len(dataset))  # 保证分布式环境下数据完整划分
    
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    # 创建测试数据加载器（固定batch_size=1）
    return torchdata.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    不做任何处理的批处理collator（直接返回原始batch）
    """
    return batch


def worker_init_reset_seed(worker_id):
    # 初始化数据加载worker的随机种子
    initial_seed = torch.initial_seed() % 2 ** 31  # 获取基础随机种子
    seed_all_rng(initial_seed + worker_id)  # 为每个worker设置不同种子
