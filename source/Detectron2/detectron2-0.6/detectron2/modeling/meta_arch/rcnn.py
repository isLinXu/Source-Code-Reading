# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable  # 导入配置相关的模块
from detectron2.data.detection_utils import convert_image_to_rgb  # 导入图像格式转换工具
from detectron2.structures import ImageList, Instances  # 导入数据结构相关的类
from detectron2.utils.events import get_event_storage  # 导入事件存储相关的工具
from detectron2.utils.logger import log_first_n  # 导入日志记录工具

from ..backbone import Backbone, build_backbone  # 导入骨干网络相关的模块
from ..postprocessing import detector_postprocess  # 导入检测后处理模块
from ..proposal_generator import build_proposal_generator  # 导入区域建议生成器
from ..roi_heads import build_roi_heads  # 导入ROI头部模块
from .build import META_ARCH_REGISTRY  # 导入模型注册器

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]  # 指定可以被外部导入的类


@META_ARCH_REGISTRY.register()  # 注册GeneralizedRCNN类到模型注册器
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    通用R-CNN。包含以下三个组件的任何模型：
    1. 每张图像的特征提取（即骨干网络）
    2. 区域建议生成
    3. 每个区域的特征提取和预测
    """

    @configurable  # 标记该方法为可配置的
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        参数说明：
            backbone: 骨干网络模块，必须遵循detectron2的骨干网络接口
            proposal_generator: 使用骨干网络特征生成建议区域的模块
            roi_heads: 执行每个区域计算的ROI头部模块
            pixel_mean, pixel_std: 列表或元组，包含通道数量的元素，表示用于归一化输入图像的每个通道的均值和标准差
            input_format: 描述输入通道的含义，可视化时需要
            vis_period: 运行可视化的周期，设置为0则禁用
        """
        super().__init__()  # 调用父类的初始化方法
        self.backbone = backbone  # 设置骨干网络
        self.proposal_generator = proposal_generator  # 设置建议区域生成器
        self.roi_heads = roi_heads  # 设置ROI头部

        self.input_format = input_format  # 设置输入格式
        self.vis_period = vis_period  # 设置可视化周期
        if vis_period > 0:  # 如果启用可视化
            assert input_format is not None, "input_format is required for visualization!"  # 确保提供了输入格式

        # 注册像素均值和标准差为模型的缓冲区
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"  # 确保均值和标准差的形状相同

    @classmethod
    def from_config(cls, cfg):  # 从配置创建模型实例的类方法
        backbone = build_backbone(cfg)  # 构建骨干网络
        return {
            "backbone": backbone,  # 返回骨干网络
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),  # 构建并返回建议区域生成器
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),  # 构建并返回ROI头部
            "input_format": cfg.INPUT.FORMAT,  # 设置输入格式
            "vis_period": cfg.VIS_PERIOD,  # 设置可视化周期
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,  # 设置像素均值
            "pixel_std": cfg.MODEL.PIXEL_STD,  # 设置像素标准差
        }

    @property
    def device(self):  # 获取模型所在设备的属性方法
        return self.pixel_mean.device  # 返回像素均值张量所在的设备

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        一个用于可视化图像和建议区域的函数。它在原始图像上显示真实边界框和最多20个得分最高的预测目标建议区域。
        用户可以为不同的模型实现不同的可视化函数。

        参数：
            batched_inputs (list): 包含模型输入的列表
            proposals (list): 包含预测建议区域的列表，与batched_inputs长度相同
        """
        from detectron2.utils.visualizer import Visualizer  # 导入可视化工具

        storage = get_event_storage()  # 获取事件存储器
        max_vis_prop = 20  # 设置最大可视化建议区域数量

        for input, prop in zip(batched_inputs, proposals):  # 遍历每个输入和对应的建议区域
            img = input["image"]  # 获取输入图像
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)  # 将图像转换为RGB格式
            v_gt = Visualizer(img, None)  # 创建真实标注的可视化器
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)  # 叠加真实边界框
            anno_img = v_gt.get_image()  # 获取带有标注的图像
            box_size = min(len(prop.proposal_boxes), max_vis_prop)  # 确定要显示的建议区域数量
            v_pred = Visualizer(img, None)  # 创建预测结果的可视化器
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()  # 叠加预测的建议区域
            )
            prop_img = v_pred.get_image()  # 获取带有预测结果的图像
            vis_img = np.concatenate((anno_img, prop_img), axis=1)  # 水平拼接两张图像
            vis_img = vis_img.transpose(2, 0, 1)  # 调整图像维度顺序
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"  # 设置可视化图像的标题
            storage.put_image(vis_name, vis_img)  # 将可视化结果存储到事件存储器
            break  # only visualize one image in a batch  # 每个批次只可视化一张图像

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        参数：
            batched_inputs: 一个列表，包含:class:`DatasetMapper`的批处理输出。
                列表中的每个元素包含一张图像的输入数据。
                目前，列表中的每个元素是一个字典，包含：

                * image: 张量，格式为(C, H, W)的图像数据
                * instances (可选): 真实标注数据 :class:`Instances`
                * proposals (可选): :class:`Instances`，预计算的建议区域

                原始字典中还包含其他信息，例如：

                * "height", "width" (int): 模型的输出分辨率，用于推理。
                  详见:meth:`postprocess`。

        返回值：
            list[dict]：
                每个字典是一张输入图像的输出结果。
                字典包含一个键"instances"，其值是一个:class:`Instances`对象。
                :class:`Instances`对象包含以下键：
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:  # 如果不是训练模式
            return self.inference(batched_inputs)  # 执行推理

        images = self.preprocess_image(batched_inputs)  # 预处理输入图像
        if "instances" in batched_inputs[0]:  # 如果输入数据包含实例标注
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # 将真实标注数据转移到对应设备
        else:
            gt_instances = None  # 没有真实标注数据

        features = self.backbone(images.tensor)  # 通过骨干网络提取特征

        if self.proposal_generator is not None:  # 如果有建议区域生成器
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)  # 生成建议区域和计算损失
        else:  # 如果没有建议区域生成器
            assert "proposals" in batched_inputs[0]  # 确保输入数据包含预计算的建议区域
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]  # 使用预计算的建议区域
            proposal_losses = {}  # 空的建议区域损失

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)  # 通过ROI头部处理特征并计算检测损失
        if self.vis_period > 0:  # 如果启用了可视化
            storage = get_event_storage()  # 获取事件存储器
            if storage.iter % self.vis_period == 0:  # 如果达到可视化周期
                self.visualize_training(batched_inputs, proposals)  # 可视化训练过程

        losses = {}  # 初始化总损失字典
        losses.update(detector_losses)  # 更新检测器损失
        losses.update(proposal_losses)  # 更新建议区域损失
        return losses  # 返回总损失

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        在给定输入上运行推理。

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
                与forward方法中的输入格式相同，包含图像和其他信息的字典列表。
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
                如果不为None，则包含每个图像的`Instances`对象。每个`Instances`对象
                包含已知的边界框（pred_boxes）和类别（pred_classes）。
                此时推理将跳过边界框检测，仅预测其他每个ROI（感兴趣区域）的输出。
            do_postprocess (bool): whether to apply post-processing on the outputs.
                是否对输出进行后处理。

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
            当do_postprocess=True时，返回格式与forward方法相同。
            否则，返回包含原始网络输出的Instances列表。
        """
        assert not self.training  # 确保模型处于推理模式

        images = self.preprocess_image(batched_inputs)  # 预处理输入图像，包括归一化和批处理
        features = self.backbone(images.tensor)  # 通过骨干网络提取图像特征

        if detected_instances is None:  # 如果没有提供已检测的实例
            if self.proposal_generator is not None:  # 如果有建议区域生成器
                proposals, _ = self.proposal_generator(images, features, None)  # 生成候选区域建议
            else:  # 如果没有建议区域生成器
                assert "proposals" in batched_inputs[0]  # 确保输入数据中包含预计算的建议区域
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]  # 使用预计算的建议区域

            results, _ = self.roi_heads(images, features, proposals, None)  # 使用ROI头部进行目标检测
        else:  # 如果提供了已检测的实例
            detected_instances = [x.to(self.device) for x in detected_instances]  # 将检测实例转移到指定设备
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)  # 使用给定的边界框进行特征提取和预测

        if do_postprocess:  # 如果需要进行后处理
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."  # 确保不在TorchScript模式下运行
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)  # 对检测结果进行后处理
        else:
            return results  # 返回原始检测结果

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        对输入图像进行归一化、填充和批处理。
        """
        images = [x["image"].to(self.device) for x in batched_inputs]  # 将输入图像转移到指定设备
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]  # 使用预定义的均值和标准差进行图像归一化
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)  # 将图像转换为ImageList格式，处理不同尺寸的图像
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        将输出实例缩放到目标尺寸。
        """
        # note: private function; subject to changes
        # 注意：这是一个私有函数，可能会发生变化
        processed_results = []  # 存储处理后的结果
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):  # 遍历每张图像的检测结果
            height = input_per_image.get("height", image_size[0])  # 获取目标高度
            width = input_per_image.get("width", image_size[1])  # 获取目标宽度
            r = detector_postprocess(results_per_image, height, width)  # 对检测结果进行后处理，调整到目标尺寸
            processed_results.append({"instances": r})  # 将处理后的结果添加到列表中
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    一个仅预测目标候选框的元架构。
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        参数说明：
            backbone: 骨干网络模块，必须遵循detectron2的骨干网络接口规范
            proposal_generator: 使用骨干网络特征生成候选框的模块
            pixel_mean, pixel_std: 包含通道数量元素的列表或元组，表示用于归一化输入图像的每个通道的均值和标准差
        """
        super().__init__()  # 调用父类的初始化方法
        self.backbone = backbone  # 设置骨干网络
        self.proposal_generator = proposal_generator  # 设置候选框生成器
        # 注册像素均值和标准差为模型的缓冲区，用于图像归一化
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        # 从配置文件构建模型实例的类方法
        backbone = build_backbone(cfg)  # 构建骨干网络
        return {
            "backbone": backbone,  # 返回骨干网络
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),  # 构建并返回候选框生成器
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,  # 设置像素均值
            "pixel_std": cfg.MODEL.PIXEL_STD,  # 设置像素标准差
        }

    @property
    def device(self):
        # 获取模型所在设备的属性方法
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        参数：
            与:class:`GeneralizedRCNN.forward`相同

        返回值：
            list[dict]：
                列表中的每个字典对应一张输入图像的输出。
                字典包含一个键"proposals"，其值是一个:class:`Instances`对象，
                该对象包含"proposal_boxes"（候选框坐标）和"objectness_logits"（目标性得分）两个键。
        """
        # 将输入图像转移到指定设备并进行归一化
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # 将图像转换为ImageList格式，处理不同尺寸的图像
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        # 通过骨干网络提取特征
        features = self.backbone(images.tensor)

        # 处理实例标注数据
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            # 处理旧版本的标注格式，targets已更名为instances
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # 生成候选框和计算损失
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        # 在训练时，虽然候选框不会被使用，但我们仍然生成它们
        # 这使得仅使用RPN的模型速度降低约5%
        if self.training:
            return proposal_losses

        # 处理每张图像的预测结果
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            # 获取输出图像的高度和宽度
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # 对预测结果进行后处理，调整到目标尺寸
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
