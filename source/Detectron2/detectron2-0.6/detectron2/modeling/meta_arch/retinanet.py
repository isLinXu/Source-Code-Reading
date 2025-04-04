# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import CycleBatchNormList, ShapeSpec, batched_nms, cat, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..anchor_generator import build_anchor_generator
from ..backbone import Backbone, build_backbone
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss
from ..matcher import Matcher
from .build import META_ARCH_REGISTRY
from .dense_detector import DenseDetector, permute_to_N_HWA_K  # noqa

__all__ = ["RetinaNet"]


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class RetinaNet(DenseDetector):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    实现RetinaNet目标检测器，参考RetinaNet论文
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        head_in_features,
        anchor_generator,
        box2box_transform,
        anchor_matcher,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        """
        NOTE: this interface is experimental.
        注意：这个接口是实验性的

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            骨干网络模块，必须遵循detectron2的骨干网络接口
            
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            检测头模块，用于预测每个特征层级的分类logits和回归偏移量
            
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            输入到检测头的特征图名称列表
            
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            锚框生成器，从特征图列表中生成锚框，通常是AnchorGenerator的实例
            
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            边界框变换器，定义从锚框到实例框的变换
            
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            锚框匹配器，通过与真实框匹配来标记锚框
            
            num_classes (int): number of classes. Used to label background proposals.
            类别数量，用于标记背景建议框

            # Loss parameters:
            # 损失函数参数：
            focal_loss_alpha (float): focal_loss_alpha
            focal loss的alpha参数
            focal_loss_gamma (float): focal_loss_gamma
            focal loss的gamma参数
            smooth_l1_beta (float): smooth_l1_beta
            smooth L1损失的beta参数
            box_reg_loss_type (str): Options are "smooth_l1", "giou", "diou", "ciou"
            边界框回归损失类型，可选项包括smooth_l1、giou、diou、ciou

            # Inference parameters:
            # 推理参数：
            test_score_thresh (float): Inference cls score threshold, only anchors with
                score > INFERENCE_TH are considered for inference (to improve speed)
            推理时的分类分数阈值，只有分数大于此阈值的锚框才会被考虑（用于提高速度）
            
            test_topk_candidates (int): Select topk candidates before NMS
            NMS前选择的topk候选框数量
            
            test_nms_thresh (float): Overlap threshold used for non-maximum suppression
                (suppress boxes with IoU >= this threshold)
            非极大值抑制使用的IoU阈值（抑制IoU大于等于此阈值的框）
            
            max_detections_per_image (int):
                Maximum number of detections to return per image during inference
                (100 is based on the limit established for the COCO dataset).
            每张图像在推理时返回的最大检测框数量（100是基于COCO数据集设定的限制）

            pixel_mean, pixel_std: see :class:`DenseDetector`.
            像素均值和标准差：参见DenseDetector类
        """
        super().__init__(
            backbone, head, head_in_features, pixel_mean=pixel_mean, pixel_std=pixel_std
        )  # 调用父类的初始化方法
        self.num_classes = num_classes  # 设置类别数量

        # Anchors
        # 锚框相关组件
        self.anchor_generator = anchor_generator  # 设置锚框生成器
        self.box2box_transform = box2box_transform  # 设置边界框变换器
        self.anchor_matcher = anchor_matcher  # 设置锚框匹配器

        # Loss parameters:
        # 损失函数参数
        self.focal_loss_alpha = focal_loss_alpha  # focal loss的alpha参数
        self.focal_loss_gamma = focal_loss_gamma  # focal loss的gamma参数
        self.smooth_l1_beta = smooth_l1_beta  # smooth L1损失的beta参数
        self.box_reg_loss_type = box_reg_loss_type  # 边界框回归损失类型
        
        # Inference parameters:
        # 推理参数
        self.test_score_thresh = test_score_thresh  # 推理时的分类分数阈值
        self.test_topk_candidates = test_topk_candidates  # NMS前的候选框数量
        self.test_nms_thresh = test_nms_thresh  # NMS的IoU阈值
        self.max_detections_per_image = max_detections_per_image  # 每张图像的最大检测框数量
        
        # Vis parameters
        # 可视化参数
        self.vis_period = vis_period  # 可视化周期
        self.input_format = input_format  # 输入格式

    @classmethod
    def from_config(cls, cfg):
        """从配置文件构建RetinaNet模型"""
        backbone = build_backbone(cfg)  # 构建骨干网络
        backbone_shape = backbone.output_shape()  # 获取骨干网络的输出形状
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]  # 获取特征图形状
        head = RetinaNetHead(cfg, feature_shapes)  # 构建RetinaNet检测头
        anchor_generator = build_anchor_generator(cfg, feature_shapes)  # 构建锚框生成器
        
        # 返回模型配置字典
        return {
            "backbone": backbone,  # 骨干网络
            "head": head,  # 检测头
            "anchor_generator": anchor_generator,  # 锚框生成器
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS),  # 边界框变换器
            "anchor_matcher": Matcher(  # 锚框匹配器
                cfg.MODEL.RETINANET.IOU_THRESHOLDS,  # IoU阈值
                cfg.MODEL.RETINANET.IOU_LABELS,  # IoU标签
                allow_low_quality_matches=True,  # 允许低质量匹配
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,  # 像素均值
            "pixel_std": cfg.MODEL.PIXEL_STD,  # 像素标准差
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,  # 类别数量
            "head_in_features": cfg.MODEL.RETINANET.IN_FEATURES,  # 输入特征
            # Loss parameters:
            # 损失函数参数
            "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,  # focal loss的alpha参数
            "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,  # focal loss的gamma参数
            "smooth_l1_beta": cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA,  # smooth L1损失的beta参数
            "box_reg_loss_type": cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE,  # 边界框回归损失类型
            # Inference parameters:
            # 推理参数
            "test_score_thresh": cfg.MODEL.RETINANET.SCORE_THRESH_TEST,  # 推理时的分类分数阈值
            "test_topk_candidates": cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST,  # NMS前的候选框数量
            "test_nms_thresh": cfg.MODEL.RETINANET.NMS_THRESH_TEST,  # NMS的IoU阈值
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,  # 每张图像的最大检测框数量
            # Vis parameters
            # 可视化参数
            "vis_period": cfg.VIS_PERIOD,  # 可视化周期
            "input_format": cfg.INPUT.FORMAT,  # 输入格式
        }

    def forward_training(self, images, features, predictions, gt_instances):
        # Transpose the Hi*Wi*A dimension to the middle:
        # 将Hi*Wi*A维度转置到中间位置
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4]
        )
        anchors = self.anchor_generator(features)  # 生成锚框
        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)  # 为锚框分配标签和对应的真实框
        return self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)  # 计算损失

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            特征层级的锚框列表
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            gt_labels和gt_boxes：参见RetinaNet.label_anchors方法的输出
                它们的形状分别为(N, R)和(N, R, 4)，其中R是所有层级的锚框总数，即sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.
            pred_logits和pred_anchor_deltas：都是张量列表，列表中的每个元素对应一个层级
                形状为(N, Hi * Wi * Ai, K或4)，其中K是pred_logits中使用的类别数

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
            返回值：
                从损失名称到标量张量的映射，用于存储损失值
                仅在训练时使用，字典的键为："loss_cls"和"loss_box_reg"
        """
        num_images = len(gt_labels)  # 获取图像数量
        gt_labels = torch.stack(gt_labels)  # (N, R)，将标签堆叠为张量

        valid_mask = gt_labels >= 0  # 创建有效样本掩码
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)  # 创建正样本掩码
        num_pos_anchors = pos_mask.sum().item()  # 计算正样本数量
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)  # 记录每张图像的平均正样本数
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)  # 使用指数移动平均更新损失归一化因子

        # classification and regression loss
        # 分类和回归损失
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class，不计算背景类的损失
        loss_cls = sigmoid_focal_loss_jit(  # 计算Focal Loss分类损失
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = _dense_box_regression_loss(  # 计算边界框回归损失
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        return {  # 返回归一化后的损失
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_box_reg / normalizer,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
                每个特征层级的锚框列表，包含了该图像在特定特征层级上的所有锚框
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.
                N个Instances实例的列表，第i个Instances包含第i张输入图像的每个实例的真实标注

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            图像数量个张量的列表。第i个元素是一个标签向量，其长度是所有特征图上的锚框总数(sum(Hi * Wi * A))。
            标签值在{-1, 0, ..., K}范围内，-1表示忽略，K表示背景。

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground.
            图像数量个张量的列表。第i个元素是一个Rx4的张量，R是所有特征图上的锚框总数。
            这些值是每个锚框匹配到的真实框。对于未被标记为前景的锚框，这些值是未定义的。
        """

        anchors = Boxes.cat(anchors)  # Rx4  # 将所有特征层级的锚框合并为一个Rx4的张量

        gt_labels = []  # 存储每个图像的锚框标签
        matched_gt_boxes = []  # 存储每个图像中锚框匹配到的真实框
        for gt_per_image in gt_instances:  # 遍历每张图像的真实标注
            # 计算真实框和锚框之间的IoU矩阵
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            # 使用锚框匹配器为每个锚框分配最匹配的真实框索引和标签
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix  # 释放内存

            if len(gt_per_image) > 0:  # 如果图像中有真实框
                # 获取每个锚框匹配到的真实框坐标
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                # 获取每个锚框匹配到的真实框类别
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                # 标签为0的锚框被视为背景，设置其类别为num_classes
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                # 标签为-1的锚框被忽略
                gt_labels_i[anchor_labels == -1] = -1
            else:  # 如果图像中没有真实框
                # 所有锚框对应的真实框坐标设为0
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                # 所有锚框的类别标签设为背景(num_classes)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)  # 添加当前图像的锚框标签
            matched_gt_boxes.append(matched_gt_boxes_i)  # 添加当前图像的匹配真实框

        return gt_labels, matched_gt_boxes

    def forward_inference(
        self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]
    ):
        # 将预测结果转置为[num_classes, 4]格式
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4]
        )
        # 根据特征图生成锚框
        anchors = self.anchor_generator(features)

        results: List[Instances] = []  # 存储所有图像的检测结果
        for img_idx, image_size in enumerate(images.image_sizes):  # 遍历每张图像
            # 对每个特征层级的分类预测进行sigmoid激活
            scores_per_image = [x[img_idx].sigmoid_() for x in pred_logits]
            # 获取每个特征层级的边界框回归预测
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            # 对单张图像进行推理
            results_per_image = self.inference_single_image(
                anchors, scores_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)  # 添加当前图像的检测结果
        return results

    def inference_single_image(
        self,
        anchors: List[Boxes],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).
        单图像推理。通过对分数进行阈值处理和应用非极大值抑制(NMS)返回边界框检测结果。

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
                特征层级列表。每个元素包含一个Boxes对象，其中包含该特征层级的所有锚框。
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
                特征层级列表。每个元素包含大小为(H x W x A, K)的张量，表示分类预测分数。
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
                与box_cls形状相同，但K变为4，表示边界框回归预测。
            image_size (tuple(H, W)): a tuple of the image height and width.
                图像高度和宽度的元组。

        Returns:
            Same as `inference`, but for only one image.
            与inference相同，但只处理一张图像。
        """
        # 解码多层级的预测结果，包括应用分数阈值和选择topk候选框
        pred = self._decode_multi_level_predictions(
            anchors,
            box_cls,
            box_delta,
            self.test_score_thresh,  # 分数阈值
            self.test_topk_candidates,  # 每层保留的候选框数量
            image_size,  # 用于裁剪预测框
        )
        # 对每个类别分别进行NMS，去除重叠的检测框
        keep = batched_nms(  # per-class NMS
            pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh
        )
        # 返回每张图像保留的最大检测框数量
        return pred[keep[: self.max_detections_per_image]]


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    RetinaNet中用于目标分类和边界框回归的检测头。
    它有两个子网络分别用于这两个任务，具有相同的结构但参数不共享。
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        num_anchors,
        conv_dims: List[int],
        norm="",
        prior_prob=0.01,
    ):
        """
        NOTE: this interface is experimental.
        注意：这个接口是实验性的。

        Args:
            input_shape (List[ShapeSpec]): input shape
                输入特征图的形状列表
            num_classes (int): number of classes. Used to label background proposals.
                类别数量，用于标记背景建议框
            num_anchors (int): number of generated anchors
                生成的锚框数量
            conv_dims (List[int]): dimensions for each convolution layer
                每个卷积层的通道数列表
            norm (str or callable):
                Normalization for conv layers except for the two output layers.
                See :func:`detectron2.layers.get_norm` for supported types.
                除了两个输出层之外的卷积层的归一化方式。
                支持的类型请参见:func:`detectron2.layers.get_norm`
            prior_prob (float): Prior weight for computing bias
                用于计算偏置的先验权重
        """
        super().__init__()

        self._num_features = len(input_shape)  # 特征层级数量
        # 处理批归一化层的设置
        if norm == "BN" or norm == "SyncBN":
            logger.info(
                f"Using domain-specific {norm} in RetinaNetHead with len={self._num_features}."
            )
            # 根据norm类型选择BatchNorm2d或SyncBatchNorm
            bn_class = nn.BatchNorm2d if norm == "BN" else nn.SyncBatchNorm

            def norm(c):
                # 创建循环批归一化层列表，用于多特征层级
                return CycleBatchNormList(
                    length=self._num_features, bn_class=bn_class, num_features=c
                )

        else:
            norm_name = str(type(get_norm(norm, 1)))
            if "BN" in norm_name:
                # 警告共享BatchNorm可能不适合RetinaNetHead
                logger.warning(
                    f"Shared BatchNorm (type={norm_name}) may not work well in RetinaNetHead."
                )

        # 构建分类子网络和边界框回归子网络
        cls_subnet = []  # 分类子网络
        bbox_subnet = []  # 边界框回归子网络
        # 构建卷积层序列
        for in_channels, out_channels in zip(
            [input_shape[0].channels] + list(conv_dims), conv_dims
        ):
            # 添加分类子网络的卷积层
            cls_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:  # 添加归一化层
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())  # 添加ReLU激活函数
            # 添加边界框回归子网络的卷积层
            bbox_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:  # 添加归一化层
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())  # 添加ReLU激活函数

        # 将子网络列表转换为Sequential模块
        self.cls_subnet = nn.Sequential(*cls_subnet)  # 分类子网络
        self.bbox_subnet = nn.Sequential(*bbox_subnet)  # 边界框回归子网络
        # 分类得分预测层
        self.cls_score = nn.Conv2d(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        # 边界框偏移预测层
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        # 初始化网络参数
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    # 使用正态分布初始化卷积层权重
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    # 初始化偏置为0
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        # 使用先验概率初始化分类层偏置以提高稳定性
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        # 构建锚框生成器并获取每个特征层的锚框数量
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # 确保所有特征层使用相同数量的锚框
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"  # 目前不支持在不同层级使用不同数量的锚框
        num_anchors = num_anchors[0]  # 获取单个特征层的锚框数量

        # 返回RetinaNetHead所需的配置参数
        return {
            "input_shape": input_shape,  # 输入特征图的形状
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,  # 类别数量
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.RETINANET.NUM_CONVS,  # 卷积层通道数，与输入通道数相同
            "prior_prob": cfg.MODEL.RETINANET.PRIOR_PROB,  # 用于计算分类层偏置的先验概率
            "norm": cfg.MODEL.RETINANET.NORM,  # 归一化层类型
            "num_anchors": num_anchors,  # 每个位置的锚框数量
        }

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
                FPN特征图张量列表，按从高分辨率到低分辨率排序。列表中的每个张量对应不同的特征层级。

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
                每层特征图的分类预测张量，形状为(N, AxK, Hi, Wi)。
                张量在每个空间位置上预测A个锚框对K个目标类别的分类概率。

            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
                每层特征图的边界框回归张量，形状为(N, Ax4, Hi, Wi)。
                张量预测每个锚框的4维向量(dx,dy,dw,dh)回归值，这些值表示锚框与真实框之间的相对偏移。
        """
        # 确保输入特征图数量与预期相同
        assert len(features) == self._num_features
        logits = []  # 存储所有特征层的分类预测结果
        bbox_reg = []  # 存储所有特征层的边界框回归预测结果
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))  # 通过分类子网络和分类得分层得到分类预测
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))  # 通过边界框子网络和预测层得到边界框回归预测
        return logits, bbox_reg  # 返回所有特征层的分类和回归预测结果
