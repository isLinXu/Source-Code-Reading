# Copyright (c) Facebook, Inc. and its affiliates.
import copy  # 导入copy模块，用于深拷贝对象
import logging  # 导入logging模块，用于日志记录
import numpy as np  # 导入numpy库并简写为np，用于数值计算
from typing import List, Optional, Union  # 导入类型提示相关的模块
import torch  # 导入PyTorch库

from detectron2.config import configurable  # 从detectron2.config导入configurable装饰器

from . import detection_utils as utils  # 从当前包导入detection_utils并简写为utils
from . import transforms as T  # 从当前包导入transforms并简写为T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""
# 这个文件包含了应用于"数据集字典"的默认映射。

__all__ = ["DatasetMapper"]  # 指定模块的公开接口


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """
    # 一个可调用对象，接收Detectron2数据集格式的字典，并将其映射为模型使用的格式。
    # 这是用于将数据集字典映射到训练数据的默认可调用对象。
    # 你可能需要参考它来实现自己的自定义逻辑，比如不同的图像读取或转换方式。
    # 详情请参见文档中的data_loading教程。
    #
    # 该可调用对象当前执行以下操作：
    # 1. 从"file_name"读取图像
    # 2. 对图像和注释应用裁剪/几何变换
    # 3. 将数据和注释准备为Tensor和Instances类

    @configurable  # 使用configurable装饰器，使类可通过配置文件进行配置
    def __init__(
        self,
        is_train: bool,  # 是否为训练模式
        *,  # 强制使用关键字参数
        augmentations: List[Union[T.Augmentation, T.Transform]],  # 数据增强列表
        image_format: str,  # 图像格式
        use_instance_mask: bool = False,  # 是否使用实例分割掩码
        use_keypoint: bool = False,  # 是否使用关键点
        instance_mask_format: str = "polygon",  # 实例掩码格式
        keypoint_hflip_indices: Optional[np.ndarray] = None,  # 关键点水平翻转索引
        precomputed_proposal_topk: Optional[int] = None,  # 预计算建议框的top-k数量
        recompute_boxes: bool = False,  # 是否重新计算边界框
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        # 注意：此接口是实验性的。
        #
        # 参数：
        #     is_train: 是否用于训练或推理
        #     augmentations: 要应用的数据增强或确定性变换列表
        #     image_format: detection_utils.read_image函数支持的图像格式
        #     use_instance_mask: 是否处理实例分割注释（如果可用）
        #     use_keypoint: 是否处理关键点注释（如果可用）
        #     instance_mask_format: "polygon"或"bitmask"之一，将实例分割掩码处理为此格式
        #     keypoint_hflip_indices: 参见detection_utils.create_keypoint_hflip_indices函数
        #     precomputed_proposal_topk: 如果给定，将从dataset_dict加载预计算的建议框，并为每个图像保留前k个建议
        #     recompute_boxes: 是否通过从实例掩码注释计算紧密边界框来覆盖边界框注释
        if recompute_boxes:  # 如果需要重新计算边界框
            assert use_instance_mask, "recompute_boxes requires instance masks"  # 断言必须使用实例掩码
        # fmt: off  # 关闭格式化，保持代码对齐
        self.is_train               = is_train  # 设置是否为训练模式
        self.augmentations          = T.AugmentationList(augmentations)  # 创建数据增强列表
        self.image_format           = image_format  # 设置图像格式
        self.use_instance_mask      = use_instance_mask  # 设置是否使用实例掩码
        self.instance_mask_format   = instance_mask_format  # 设置实例掩码格式
        self.use_keypoint           = use_keypoint  # 设置是否使用关键点
        self.keypoint_hflip_indices = keypoint_hflip_indices  # 设置关键点水平翻转索引
        self.proposal_topk          = precomputed_proposal_topk  # 设置预计算建议框的top-k数量
        self.recompute_boxes        = recompute_boxes  # 设置是否重新计算边界框
        # fmt: on  # 恢复格式化
        logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
        mode = "training" if is_train else "inference"  # 根据is_train确定模式
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")  # 记录使用的数据增强

    @classmethod  # 类方法装饰器
    def from_config(cls, cfg, is_train: bool = True):  # 从配置创建实例的类方法
        augs = utils.build_augmentation(cfg, is_train)  # 根据配置构建数据增强
        if cfg.INPUT.CROP.ENABLED and is_train:  # 如果启用了裁剪且处于训练模式
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))  # 在增强列表开头插入随机裁剪
            recompute_boxes = cfg.MODEL.MASK_ON  # 如果启用了掩码，则需要重新计算边界框
        else:
            recompute_boxes = False  # 否则不重新计算边界框

        ret = {  # 返回初始化参数字典
            "is_train": is_train,  # 是否为训练模式
            "augmentations": augs,  # 数据增强列表
            "image_format": cfg.INPUT.FORMAT,  # 图像格式
            "use_instance_mask": cfg.MODEL.MASK_ON,  # 是否使用实例掩码
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,  # 实例掩码格式
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,  # 是否使用关键点
            "recompute_boxes": recompute_boxes,  # 是否重新计算边界框
        }

        if cfg.MODEL.KEYPOINT_ON:  # 如果启用了关键点
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)  # 创建关键点水平翻转索引

        if cfg.MODEL.LOAD_PROPOSALS:  # 如果需要加载预计算的建议框
            ret["precomputed_proposal_topk"] = (  # 设置预计算建议框的top-k数量
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN  # 训练时的top-k数量
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST  # 测试时的top-k数量
            )
        return ret  # 返回参数字典

    def _transform_annotations(self, dataset_dict, transforms, image_shape):  # 转换注释的私有方法
        # USER: Modify this if you want to keep them for some reason.
        # 用户：如果你出于某种原因想保留它们，可以修改这里
        for anno in dataset_dict["annotations"]:  # 遍历所有注释
            if not self.use_instance_mask:  # 如果不使用实例掩码
                anno.pop("segmentation", None)  # 移除分割信息
            if not self.use_keypoint:  # 如果不使用关键点
                anno.pop("keypoints", None)  # 移除关键点信息

        # USER: Implement additional transformations if you have other types of data
        # 用户：如果你有其他类型的数据，可以实现额外的转换
        annos = [  # 创建转换后的注释列表
            utils.transform_instance_annotations(  # 转换实例注释
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices  # 应用变换
            )
            for obj in dataset_dict.pop("annotations")  # 从字典中弹出并遍历注释
            if obj.get("iscrowd", 0) == 0  # 只处理非群体对象
        ]
        instances = utils.annotations_to_instances(  # 将注释转换为实例
            annos, image_shape, mask_format=self.instance_mask_format  # 使用指定的掩码格式
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        # 在应用如裁剪等变换后，边界框可能不再紧密包围对象。例如，想象一个三角形对象
        # [(0,0), (2,0), (0,2)]被一个框[(1,0),(2,2)](XYXY格式)裁剪。裁剪后的三角形的
        # 紧密边界框应该是[(1,0),(2,1)]，这与原始边界框和裁剪框的交集不相等。
        if self.recompute_boxes:  # 如果需要重新计算边界框
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # 从掩码获取紧密边界框
        dataset_dict["instances"] = utils.filter_empty_instances(instances)  # 过滤空实例并存储到字典中

    def __call__(self, dataset_dict):  # 实现可调用接口，用于处理单个数据集字典
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # 参数：
        #     dataset_dict (dict)：一张图像的元数据，采用Detectron2数据集格式。
        #
        # 返回：
        #     dict：detectron2内置模型接受的格式
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # 深拷贝数据集字典，因为下面的代码会修改它，避免影响原始数据
        # USER: Write your own image loading if it's not from a file
        # 用户：如果图像不是从文件加载的，请编写自己的图像加载代码
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)  # 从文件路径读取图像，使用指定的图像格式
        utils.check_image_size(dataset_dict, image)  # 检查图像尺寸是否符合要求

        # USER: Remove if you don't do semantic/panoptic segmentation.
        # 用户：如果不进行语义/全景分割，可以移除这部分代码
        if "sem_seg_file_name" in dataset_dict:  # 如果数据集包含语义分割标注文件
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)  # 读取语义分割标注图像，转换为单通道
        else:
            sem_seg_gt = None  # 没有语义分割标注

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)  # 创建数据增强的输入对象，包含图像和语义分割标注
        transforms = self.augmentations(aug_input)  # 应用数据增强变换
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg  # 获取增强后的图像和语义分割标注

        image_shape = image.shape[:2]  # h, w，获取图像的高度和宽度
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # PyTorch的数据加载器对torch.Tensor很高效（因为使用共享内存），
        # 但对大型通用数据结构效率不高（因为使用pickle和mp.Queue）。
        # 因此使用torch.Tensor很重要。
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))  # 将图像转换为PyTorch张量，并调整通道顺序为(C,H,W)
        if sem_seg_gt is not None:  # 如果有语义分割标注
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))  # 将语义分割标注转换为长整型张量

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        # 用户：如果不使用预计算的建议框，可以移除这部分代码。
        # 大多数用户不需要这个功能。
        if self.proposal_topk is not None:  # 如果指定了建议框的top-k数量
            utils.transform_proposals(  # 转换预计算的建议框
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:  # 如果是在测试/推理阶段
            # USER: Modify this if you want to keep them for some reason.
            # 用户：如果出于某种原因想保留这些信息，可以修改这里
            dataset_dict.pop("annotations", None)  # 移除标注信息
            dataset_dict.pop("sem_seg_file_name", None)  # 移除语义分割文件路径
            return dataset_dict  # 返回处理后的数据字典

        if "annotations" in dataset_dict:  # 如果数据集包含标注信息（训练阶段）
            self._transform_annotations(dataset_dict, transforms, image_shape)  # 转换标注信息，应用相同的数据增强变换

        return dataset_dict  # 返回处理后的数据字典
