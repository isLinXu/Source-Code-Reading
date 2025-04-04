# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# 导入所需的Python标准库和第三方库
import logging  # 导入日志模块
from typing import Dict, List  # 导入类型注解
import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块

# 导入detectron2的相关模块
from detectron2.config import configurable  # 导入配置相关的装饰器
from detectron2.structures import ImageList  # 导入图像列表数据结构

# 导入后处理相关函数
from ..postprocessing import detector_postprocess, sem_seg_postprocess  # 导入检测器和语义分割的后处理函数
from .build import META_ARCH_REGISTRY  # 导入模型注册器
from .rcnn import GeneralizedRCNN  # 导入通用RCNN基类
from .semantic_seg import build_sem_seg_head  # 导入语义分割头部构建函数

__all__ = ["PanopticFPN"]  # 指定该模块的公开接口


@META_ARCH_REGISTRY.register()  # 注册PanopticFPN模型到元架构注册表
class PanopticFPN(GeneralizedRCNN):  # 定义PanopticFPN类，继承自GeneralizedRCNN
    """
    Implement the paper :paper:`PanopticFPN`.
    实现PanopticFPN论文中的方法
    """

    @configurable  # 使用configurable装饰器，使该类可配置
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,  # 语义分割头部模块
        combine_overlap_thresh: float = 0.5,  # 实例合并的重叠阈值
        combine_stuff_area_thresh: float = 4096,  # 背景区域的面积阈值
        combine_instances_score_thresh: float = 0.5,  # 实例分割的置信度阈值
        **kwargs,  # 其他参数
    ):
        """
        NOTE: this interface is experimental.
        注意：这个接口是实验性的。

        Args:
            sem_seg_head: a module for the semantic segmentation head.
            语义分割头部模块，用于处理语义分割任务。
            combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
            当两个掩码的重叠程度超过此阈值时，将它们合并为一个实例。
            combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
            忽略小于此阈值的背景区域。
            combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold
            忽略置信度低于此阈值的实例。

        Other arguments are the same as :class:`GeneralizedRCNN`.
        其他参数与GeneralizedRCNN类相同。
        """
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.sem_seg_head = sem_seg_head  # 初始化语义分割头部
        # options when combining instance & semantic outputs
        # 设置实例分割和语义分割结果合并时的选项
        self.combine_overlap_thresh = combine_overlap_thresh  # 设置重叠阈值
        self.combine_stuff_area_thresh = combine_stuff_area_thresh  # 设置背景区域面积阈值
        self.combine_instances_score_thresh = combine_instances_score_thresh  # 设置实例置信度阈值

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update(
            {
                "combine_overlap_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH,
                "combine_stuff_area_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT,
                "combine_instances_score_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH,  # noqa
            }
        )
        ret["sem_seg_head"] = build_sem_seg_head(cfg, ret["backbone"].output_shape())
        logger = logging.getLogger(__name__)
        if not cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED:
            logger.warning(
                "PANOPTIC_FPN.COMBINED.ENABLED is no longer used. "
                " model.inference(do_postprocess=) should be used to toggle postprocessing."
            )
        if cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT != 1.0:
            w = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT
            logger.warning(
                "PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT should be replaced by weights on each ROI head."
            )

            def update_weight(x):
                if isinstance(x, dict):
                    return {k: v * w for k, v in x.items()}
                else:
                    return x * w

            roi_heads = ret["roi_heads"]
            roi_heads.box_predictor.loss_weight = update_weight(roi_heads.box_predictor.loss_weight)
            roi_heads.mask_head.loss_weight = update_weight(roi_heads.mask_head.loss_weight)
        return ret

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                DatasetMapper的批处理输出列表，每个元素包含一张图像的输入数据。

                For now, each item in the list is a dict that contains:
                目前，列表中的每个元素都是一个包含以下内容的字典：

                * "image": Tensor, image in (C, H, W) format.
                  图像张量，格式为(C, H, W)。
                * "instances": Instances
                  实例标注信息。
                * "sem_seg": semantic segmentation ground truth.
                  语义分割的真实标签。
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
                  原始字典中包含的其他信息，如：
                  "height", "width"（整数）：模型输出的分辨率，用于推理。
                  详见postprocess方法。

        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                每个字典包含一张图像的结果，字典包含以下键：

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                  实例分割结果，格式见GeneralizedRCNN.forward方法。
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                  语义分割结果，格式见SemanticSegmentor.forward方法。
                * "panoptic_seg": See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
                  全景分割结果，格式见combine_semantic_and_instance_outputs函数的返回值。
        """
        if not self.training:  # 如果不是训练模式，则执行推理
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)  # 预处理输入图像
        features = self.backbone(images.tensor)  # 通过骨干网络提取特征

        assert "sem_seg" in batched_inputs[0]  # 确保输入数据包含语义分割标签
        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]  # 将语义分割标签转移到指定设备
        gt_sem_seg = ImageList.from_tensors(  # 将语义分割标签转换为张量列表
            gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
        ).tensor
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)  # 计算语义分割结果和损失

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # 将实例标注转移到指定设备
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)  # 生成候选区域和计算损失
        detector_results, detector_losses = self.roi_heads(  # 计算检测结果和损失
            images, features, proposals, gt_instances
        )

        losses = sem_seg_losses  # 合并所有损失
        losses.update(proposal_losses)  # 添加候选区域生成的损失
        losses.update(detector_losses)  # 添加检测器的损失
        return losses  # 返回总损失

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        """
        Run inference on the given inputs.
        对给定的输入执行推理。

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            batched_inputs (list[dict]): 与forward方法中的输入格式相同
            do_postprocess (bool): whether to apply post-processing on the outputs.
            do_postprocess (bool): 是否对输出进行后处理

        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
            当do_postprocess=True时，返回格式见forward方法的文档。
            否则，返回包含原始检测器输出和原始语义分割输出的元组(list[Instances], list[Tensor])。
        """
        images = self.preprocess_image(batched_inputs)  # 预处理输入图像
        features = self.backbone(images.tensor)  # 通过骨干网络提取特征
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, None)  # 计算语义分割结果
        proposals, _ = self.proposal_generator(images, features, None)  # 生成候选区域
        detector_results, _ = self.roi_heads(images, features, proposals, None)  # 计算检测结果

        if do_postprocess:  # 如果需要进行后处理
            processed_results = []  # 存储处理后的结果
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):  # 遍历每张图像的结果
                height = input_per_image.get("height", image_size[0])  # 获取输出高度
                width = input_per_image.get("width", image_size[1])  # 获取输出宽度
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)  # 对语义分割结果进行后处理
                detector_r = detector_postprocess(detector_result, height, width)  # 对检测结果进行后处理

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})  # 添加处理后的结果

                panoptic_r = combine_semantic_and_instance_outputs(  # 合并语义分割和实例分割结果
                    detector_r,
                    sem_seg_r.argmax(dim=0),  # 获取每个像素的类别
                    self.combine_overlap_thresh,  # 重叠阈值
                    self.combine_stuff_area_thresh,  # 背景区域面积阈值
                    self.combine_instances_score_thresh,  # 实例置信度阈值
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r  # 添加全景分割结果
            return processed_results  # 返回处理后的结果
        else:
            return detector_results, sem_seg_results  # 返回原始结果


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_thresh,
    instances_score_thresh,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.
    实现一个简单的合并逻辑，参考panopticapi中的
    "combine_semantic_and_instance_predictions.py"来生成全景分割输出。

    Args:
        instance_results: output of :func:`detector_postprocess`.
        实例分割的后处理结果。
        semantic_results: an (H, W) tensor, each element is the contiguous semantic
            category id
        语义分割结果，形状为(H, W)的张量，每个元素是连续的语义类别ID。

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        panoptic_seg (Tensor): 形状为(height, width)的张量，其中的值是每个分割区域的ID。
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
        segments_info (list[dict]): 描述panoptic_seg中的每个分割区域。
            每个字典包含"id"、"category_id"、"isthing"等键。
    """
    # 创建与语义分割结果相同大小的全零张量，用于存储全景分割结果
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    # 根据置信度对实例结果进行排序
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0  # 当前分割区域的ID
    segments_info = []  # 存储分割区域的信息

    # 将实例掩码转换为布尔类型并移动到指定设备
    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    # 逐个添加实例，检查与现有实例的重叠情况
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()  # 获取实例的置信度
        if score < instances_score_thresh:  # 如果置信度低于阈值，则停止处理
            break
        mask = instance_masks[inst_id]  # H,W，获取实例掩码
        mask_area = mask.sum().item()  # 计算掩码面积

        if mask_area == 0:  # 如果掩码面积为0，则跳过
            continue

        # 计算与现有分割区域的重叠部分
        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()  # 计算重叠面积

        # 如果重叠面积占比过大，则跳过
        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        # 如果有重叠，只保留未分配的部分
        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        # 为当前实例分配新的分割ID
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,  # 分割区域ID
                "isthing": True,  # 标记为物体类
                "score": score,  # 置信度
                "category_id": instance_results.pred_classes[inst_id].item(),  # 类别ID
                "instance_id": inst_id.item(),  # 实例ID
            }
        )

    # Add semantic results to remaining empty areas
    # 将语义分割结果添加到剩余的空白区域
    semantic_labels = torch.unique(semantic_results).cpu().tolist()  # 获取所有唯一的语义标签
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue  # 跳过特殊的物体类标签0
        # 找出当前语义类别的未分配区域
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()  # 计算区域面积
        if mask_area < stuff_area_thresh:  # 如果面积小于阈值，则跳过
            continue

        # 为当前语义区域分配新的分割ID
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,  # 分割区域ID
                "isthing": False,  # 标记为背景类
                "category_id": semantic_label,  # 类别ID
                "area": mask_area,  # 区域面积
            }
        )

    return panoptic_seg, segments_info  # 返回全景分割结果和区域信息
