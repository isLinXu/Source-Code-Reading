# Copyright (c) Facebook, Inc. and its affiliates.
# 版权所有 (c) Facebook, Inc. 及其附属公司。
import numpy as np
import pycocotools.mask as mask_util

from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
    _PanopticPrediction,
)

from .colormap import random_color


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.
    
    用于存储视频帧中检测到的对象数据，以便将颜色传递到未来帧中的对象。

    Attributes:
        label (int): 
        # 标签（整数类型）：表示对象的类别ID
        bbox (tuple[float]):
        # 边界框（浮点数元组）：表示对象的边界框坐标
        mask_rle (dict):
        # 掩码RLE（字典）：使用RLE（Run-Length Encoding，游程编码）格式存储掩码
        color (tuple[float]): RGB colors in range (0, 1)
        # 颜色（浮点数元组）：RGB颜色值，范围为(0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
        # 生存时间（整数）：实例的生存时间。例如，如果ttl=2，则实例颜色可以传递到接下来的两帧中的对象。
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]
    # __slots__用于优化内存使用，限制该类只能有这些属性

    def __init__(self, label, bbox, mask_rle, color, ttl):
        # 初始化方法，设置检测实例的各项属性
        self.label = label  # 类别标签
        self.bbox = bbox    # 边界框
        self.mask_rle = mask_rle  # 掩码的RLE编码
        self.color = color  # 实例的颜色
        self.ttl = ttl      # 生存时间


class VideoVisualizer:
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
            # 元数据目录：包含图像元数据信息
        """
        self.metadata = metadata  # 存储元数据
        self._old_instances = []  # 存储上一帧中的实例，用于跟踪
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        # 断言实例模式必须是支持的模式之一，目前只支持IMAGE和IMAGE_BW两种模式
        self._instance_mode = instance_mode  # 设置实例显示模式

    def draw_instance_predictions(self, frame, predictions):
        """
        Draw instance-level prediction results on an image.
        在图像上绘制实例级预测结果。

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            # 帧（ndarray）：形状为(H, W, C)的RGB图像，值范围为[0, 255]。
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
            # 预测结果（Instances）：实例检测/分割模型的输出。将使用以下字段进行绘制：
            # "pred_boxes"（预测框）, "pred_classes"（预测类别）, "scores"（得分）, "pred_masks"（预测掩码）。

        Returns:
            output (VisImage): image object with visualizations.
            # 输出（VisImage）：带有可视化内容的图像对象。
        """
        frame_visualizer = Visualizer(frame, self.metadata)  # 创建帧可视化器
        num_instances = len(predictions)  # 获取预测实例数量
        if num_instances == 0:
            return frame_visualizer.output  # 如果没有实例，直接返回原始帧

        # 从预测结果中提取各种信息
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None  # 提取预测框
        scores = predictions.scores if predictions.has("scores") else None  # 提取分数
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None  # 提取类别
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None  # 提取关键点
        colors = predictions.COLOR if predictions.has("COLOR") else [None] * len(predictions)  # 提取颜色，如果没有则创建None列表

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks  # 提取预测掩码
            # mask IOU is not yet enabled
            # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
            # assert len(masks_rles) == num_instances
            # 掩码IOU功能尚未启用
        else:
            masks = None

        # 为每个检测到的实例创建_DetectedInstance对象
        detected = [
            _DetectedInstance(classes[i], boxes[i], mask_rle=None, color=colors[i], ttl=8)
            for i in range(num_instances)
        ]
        if not predictions.has("COLOR"):
            colors = self._assign_colors(detected)  # 如果没有预先分配的颜色，则分配颜色

        # 创建文本标签（类别名称和得分）
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            # 如果是黑白图像模式，将背景转为灰度
            frame_visualizer.output.reset_image(
                frame_visualizer._create_grayscale_image(
                    (masks.any(dim=0) > 0).numpy() if masks is not None else None
                )
            )
            alpha = 0.3  # 设置黑白模式下的透明度
        else:
            alpha = 0.5  # 设置彩色模式下的透明度

        # 在帧上叠加实例（框、掩码、标签等）
        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes,  # boxes are a bit distracting
            # 如果有掩码，则不显示边界框（因为边界框有些分散注意力）
            masks=masks,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        return frame_visualizer.output  # 返回可视化后的图像

    def draw_sem_seg(self, frame, sem_seg, area_threshold=None):
        """
        Args:
            sem_seg (ndarray or Tensor): semantic segmentation of shape (H, W),
                each value is the integer label.
                # 语义分割（ndarray或Tensor）：形状为(H, W)的语义分割图，每个值都是整数标签。
            area_threshold (Optional[int]): only draw segmentations larger than the threshold
                # 面积阈值（可选整数）：只绘制大于阈值的分割区域
        """
        # don't need to do anything special
        # 不需要做任何特殊处理
        frame_visualizer = Visualizer(frame, self.metadata)  # 创建帧可视化器
        frame_visualizer.draw_sem_seg(sem_seg, area_threshold=None)  # 绘制语义分割
        return frame_visualizer.output  # 返回可视化后的图像

    def draw_panoptic_seg_predictions(
        self, frame, panoptic_seg, segments_info, area_threshold=None, alpha=0.5
    ):
        """
        绘制全景分割预测结果
        """
        frame_visualizer = Visualizer(frame, self.metadata)  # 创建帧可视化器
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)  # 创建全景分割预测对象

        if self._instance_mode == ColorMode.IMAGE_BW:
            # 如果是黑白图像模式，将背景转为灰度
            frame_visualizer.output.reset_image(
                frame_visualizer._create_grayscale_image(pred.non_empty_mask())
            )

        # draw mask for all semantic segments first i.e. "stuff"
        # 首先绘制所有语义分割的掩码，即"stuff"（非实例物体，如天空、道路等）
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]  # 获取类别索引
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]  # 尝试获取预定义颜色
            except AttributeError:
                mask_color = None  # 如果没有预定义颜色，则使用默认颜色

            # 绘制二值掩码
            frame_visualizer.draw_binary_mask(
                mask,
                color=mask_color,
                text=self.metadata.stuff_classes[category_idx],
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # 收集所有实例掩码
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return frame_visualizer.output  # 如果没有实例，直接返回
        # draw mask for all instances second
        # 其次绘制所有实例的掩码
        masks, sinfo = list(zip(*all_instances))  # 分离掩码和分割信息
        num_instances = len(masks)  # 获取实例数量
        # 将掩码编码为RLE格式
        masks_rles = mask_util.encode(
            np.asarray(np.asarray(masks).transpose(1, 2, 0), dtype=np.uint8, order="F")
        )
        assert len(masks_rles) == num_instances  # 确保RLE掩码数量与实例数量一致

        # 获取类别ID
        category_ids = [x["category_id"] for x in sinfo]
        # 为每个检测到的实例创建_DetectedInstance对象
        detected = [
            _DetectedInstance(category_ids[i], bbox=None, mask_rle=masks_rles[i], color=None, ttl=8)
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)  # 分配颜色
        labels = [self.metadata.thing_classes[k] for k in category_ids]  # 创建标签

        # 在帧上叠加实例
        frame_visualizer.overlay_instances(
            boxes=None,
            masks=masks,
            labels=labels,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return frame_visualizer.output  # 返回可视化后的图像

    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.
        
        简单的跟踪启发式算法，为相同的实例分配相同的颜色，
        会更新已跟踪实例的内部状态。

        Returns:
            list[tuple[float]]: list of colors.
            # 浮点数元组列表：颜色列表。
        """

        # Compute iou with either boxes or masks:
        # 使用边界框或掩码计算IOU（交并比）：
        is_crowd = np.zeros((len(instances),), dtype=np.bool)  # 创建crowd标志数组（全为False）
        if instances[0].bbox is None:
            assert instances[0].mask_rle is not None
            # use mask iou only when box iou is None
            # because box seems good enough
            # 仅当边界框为None时使用掩码IOU，因为边界框通常足够好
            rles_old = [x.mask_rle for x in self._old_instances]  # 获取旧实例的RLE掩码
            rles_new = [x.mask_rle for x in instances]  # 获取新实例的RLE掩码
            ious = mask_util.iou(rles_old, rles_new, is_crowd)  # 计算掩码IOU
            threshold = 0.5  # 设置掩码IOU的阈值
        else:
            boxes_old = [x.bbox for x in self._old_instances]  # 获取旧实例的边界框
            boxes_new = [x.bbox for x in instances]  # 获取新实例的边界框
            ious = mask_util.iou(boxes_old, boxes_new, is_crowd)  # 计算边界框IOU
            threshold = 0.6  # 设置边界框IOU的阈值
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")  # 如果没有IOU，创建全零数组

        # Only allow matching instances of the same label:
        # 只允许匹配相同标签的实例：
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0  # 如果标签不同，将IOU设为0

        # 找出每个旧实例匹配的最佳新实例
        matched_new_per_old = np.asarray(ious).argmax(axis=1)  # 每个旧实例匹配的新实例索引
        max_iou_per_old = np.asarray(ious).max(axis=1)  # 每个旧实例的最大IOU值

        # Try to find match for each old instance:
        # 尝试为每个旧实例找到匹配：
        extra_instances = []  # 存储未匹配的旧实例
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:  # 如果最大IOU超过阈值
                newidx = matched_new_per_old[idx]  # 获取匹配的新实例索引
                if instances[newidx].color is None:  # 如果新实例没有颜色
                    instances[newidx].color = inst.color  # 将旧实例的颜色传递给新实例
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            # 如果旧实例没有匹配到任何新实例，保留它以便在下一帧中使用，以防它只是被检测器漏检
            inst.ttl -= 1  # 减少生存时间
            if inst.ttl > 0:  # 如果生存时间仍大于0
                extra_instances.append(inst)  # 将其添加到额外实例列表中

        # Assign random color to newly-detected instances:
        # 为新检测到的实例分配随机颜色：
        for inst in instances:
            if inst.color is None:  # 如果实例没有颜色
                inst.color = random_color(rgb=True, maximum=1)  # 分配随机颜色
        self._old_instances = instances[:] + extra_instances  # 更新旧实例列表（当前实例+未匹配的旧实例）
        return [d.color for d in instances]  # 返回实例颜色列表
