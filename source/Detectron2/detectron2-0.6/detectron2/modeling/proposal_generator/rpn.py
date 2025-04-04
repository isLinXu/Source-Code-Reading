# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

# 导入必要的模块和组件
from detectron2.config import configurable  # 导入配置相关的模块
from detectron2.layers import Conv2d, ShapeSpec, cat  # 导入基础网络层组件
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou  # 导入数据结构相关组件
from detectron2.utils.events import get_event_storage  # 导入事件存储工具
from detectron2.utils.memory import retry_if_cuda_oom  # 导入CUDA内存管理工具
from detectron2.utils.registry import Registry  # 导入注册器工具

# 导入RPN相关的组件
from ..anchor_generator import build_anchor_generator  # 导入锚框生成器
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss  # 导入边界框回归相关组件
from ..matcher import Matcher  # 导入匹配器
from ..sampling import subsample_labels  # 导入标签采样工具
from .build import PROPOSAL_GENERATOR_REGISTRY  # 导入建议框生成器注册表
from .proposal_utils import find_top_rpn_proposals  # 导入建议框筛选工具

# 创建RPN头部注册表
RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.
# RPN头部的注册表，用于接收特征图并对锚框执行目标分类和边界框回归

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
# 注册的对象将通过`obj(cfg, input_shape)`调用
# 该调用应返回一个`nn.Module`对象
"""


"""
Shape shorthand in this module:
# 本模块中使用的形状简写：

    N: number of images in the minibatch
    # N: 小批量中的图像数量
    L: number of feature maps per image on which RPN is run
    # L: 每张图像上RPN运行的特征图数量
    A: number of cell anchors (must be the same for all feature maps)
    # A: 每个单元的锚框数量（对所有特征图必须相同）
    Hi, Wi: height and width of the i-th feature map
    # Hi, Wi: 第i个特征图的高度和宽度
    B: size of the box parameterization
    # B: 边界框参数化的大小

Naming convention:
# 命名约定：

    objectness: refers to the binary classification of an anchor as object vs. not object.
    # objectness: 指锚框是否包含目标的二分类

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.
    # deltas: 指用于参数化边界框转换的4维(dx, dy, dw, dh)偏移量
    # (参见 :class:`box_regression.Box2BoxTransform`)，或旋转框的5维偏移量

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).
    # pred_objectness_logits: 预测的目标性得分，范围在[-inf, +inf]之间；
    # 使用sigmoid(pred_objectness_logits)来估计目标的概率P(object)

    gt_labels: ground-truth binary classification labels for objectness
    # gt_labels: 目标性的真实二分类标签

    pred_anchor_deltas: predicted box2box transform deltas
    # pred_anchor_deltas: 预测的边界框转换偏移量

    gt_anchor_deltas: ground-truth box2box transform deltas
    # gt_anchor_deltas: 真实的边界框转换偏移量
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    # 构建由`cfg.MODEL.RPN.HEAD_NAME`定义的RPN头部
    """
    name = cfg.MODEL.RPN.HEAD_NAME  # 获取RPN头部的名称
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)  # 从注册表中获取并实例化对应的RPN头部


@RPN_HEAD_REGISTRY.register()  # 注册StandardRPNHead类到RPN头部注册表
class StandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    # Faster R-CNN论文中描述的标准RPN分类和回归头部
    # 使用3x3卷积产生共享的隐藏状态，然后通过两个1x1卷积分别预测：
    # 1. 每个锚框的目标性得分
    # 2. 用于将每个锚框变形为目标建议框的边界框偏移量
    """

    @configurable  # 标记该方法为可配置的
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)
    ):
        """
        NOTE: this interface is experimental.
        # 注意：这个接口是实验性的

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            # in_channels (int): 输入特征通道数。当使用多个输入特征时，
            # 它们必须具有相同的通道数

            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            # num_anchors (int): 特征图上每个空间位置预测的锚框数量
            # 每个特征图的总锚框数量将是`num_anchors * H * W`

            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
            # box_dim (int): 边界框的维度，也是每个锚框需要预测的回归值数量
            # 轴对齐的框有4个维度，旋转框有5个维度

            conv_dims (list[int]): a list of integers representing the output channels
                of N conv layers. Set it to -1 to use the same number of output channels
                as input channels.
            # conv_dims (list[int]): 表示N个卷积层输出通道数的整数列表
            # 设置为-1时使用与输入通道数相同的输出通道数
        """
        super().__init__()  # 调用父类的初始化方法
        cur_channels = in_channels  # 初始化当前通道数为输入通道数
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        # 保持旧的变量名和结构以保持向后兼容性
        # 否则旧的检查点将无法加载
        if len(conv_dims) == 1:  # 如果只有一个卷积层
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]  # 确定输出通道数
            # 3x3 conv for the hidden representation
            # 使用3x3卷积生成隐藏表示
            self.conv = self._get_rpn_conv(cur_channels, out_channels)  # 创建RPN卷积层
            cur_channels = out_channels  # 更新当前通道数
        else:  # 如果有多个卷积层
            self.conv = nn.Sequential()  # 创建顺序容器
            for k, conv_dim in enumerate(conv_dims):  # 遍历每个卷积维度
                out_channels = cur_channels if conv_dim == -1 else conv_dim  # 确定输出通道数
                if out_channels <= 0:  # 检查输出通道数的有效性
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)  # 创建RPN卷积层
                self.conv.add_module(f"conv{k}", conv)  # 将卷积层添加到顺序容器中
                cur_channels = out_channels  # 更新当前通道数
        # 1x1 conv for predicting objectness logits
        # 使用1x1卷积预测目标性得分
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        # 使用1x1卷积预测边界框转换偏移量
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        # Keeping the order of weights initialization same for backwards compatiblility.
        # 保持权重初始化的顺序以保持向后兼容性
        for layer in self.modules():  # 遍历所有模块
            if isinstance(layer, nn.Conv2d):  # 如果是卷积层
                nn.init.normal_(layer.weight, std=0.01)  # 使用正态分布初始化权重
                nn.init.constant_(layer.bias, 0)  # 将偏置初始化为0

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        # 标准RPN在所有层级之间共享：
        in_channels = [s.channels for s in input_shape]  # 获取所有输入特征的通道数
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"  # 确保所有层级具有相同的通道数
        in_channels = in_channels[0]  # 获取通道数

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        # RPN头部应该与锚框生成器使用相同的输入
        # 注意：假设创建锚框生成器不会产生不必要的副作用
        anchor_generator = build_anchor_generator(cfg, input_shape)  # 构建锚框生成器
        num_anchors = anchor_generator.num_anchors  # 获取锚框数量
        box_dim = anchor_generator.box_dim  # 获取边界框维度
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"  # 确保每个空间位置的锚框数量相同
        return {
            "in_channels": in_channels,  # 输入通道数
            "num_anchors": num_anchors[0],  # 每个位置的锚框数量
            "box_dim": box_dim,  # 边界框维度
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,  # 卷积层的输出通道数配置
        }

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps
            # features (list[Tensor]): 特征图列表

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            # list[Tensor]: 包含L个元素的列表
            # 第i个元素是形状为(N, A, Hi, Wi)的张量，表示所有锚框的预测目标性得分
            # 其中A是每个单元的锚框数量
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            # list[Tensor]: 包含L个元素的列表
            # 第i个元素是形状为(N, A*box_dim, Hi, Wi)的张量
            # 表示用于将锚框转换为建议框的预测偏移量
        """
        pred_objectness_logits = []  # 存储预测的目标性得分
        pred_anchor_deltas = []  # 存储预测的锚框偏移量
        for x in features:  # 遍历每个特征图
            t = self.conv(x)  # 通过共享的卷积层处理特征
            pred_objectness_logits.append(self.objectness_logits(t))  # 预测目标性得分
            pred_anchor_deltas.append(self.anchor_deltas(t))  # 预测边界框偏移量
        return pred_objectness_logits, pred_anchor_deltas  # 返回预测结果


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    区域建议网络，由Faster R-CNN论文引入。这是一个用于生成目标检测候选框的网络模块。
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
    ):
        """
        NOTE: this interface is experimental.
        注意：这是一个实验性接口。

        Args:
            in_features (list[str]): list of names of input features to use
            # 要使用的输入特征名称列表
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            # 一个模块，用于从每个层级的特征列表中预测逻辑值和回归偏移量
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            # 从特征列表生成锚框的模块，通常是AnchorGenerator类的实例
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            # 通过将锚框与真实标注框匹配来标记锚框的匹配器
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            # 定义从锚框到实例框的变换
            batch_size_per_image (int): number of anchors per image to sample for training
            # 每张图像用于训练的锚框采样数量
            positive_fraction (float): fraction of foreground anchors to sample for training
            # 用于训练的前景锚框采样比例
            pre_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select before NMS, in
                training and testing.
            # (训练，测试)时NMS前选择的top k个建议框数量
            post_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select after NMS, in
                training and testing.
            # (训练，测试)时NMS后选择的top k个建议框数量
            nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
            # 用于去除重复预测建议框的NMS阈值
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            # 移除任意边长小于该阈值的建议框（单位：输入图像像素）
            anchor_boundary_thresh (float): legacy option
            # 锚框边界阈值（遗留选项）
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                    "loss_rpn_cls" - applied to classification loss
                    "loss_rpn_loc" - applied to box regression loss
            # 损失权重，可以是单个浮点数（同时权重所有RPN损失）或字典（单独权重）。
            # 有效的字典键包括：
            #     "loss_rpn_cls" - 应用于分类损失
            #     "loss_rpn_loc" - 应用于框回归损失
            box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
            # 使用的框回归损失类型。支持的损失："smooth_l1"、"giou"
            smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
                use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
            # smooth L1回归损失的beta参数。默认使用L1损失。
            # 仅在box_reg_loss_type为"smooth_l1"时使用
        """
        super().__init__()  # 调用父类初始化方法
        self.in_features = in_features  # 存储输入特征名称列表
        self.rpn_head = head  # 存储RPN头部模块，用于预测目标性和边界框回归
        self.anchor_generator = anchor_generator  # 存储锚框生成器
        self.anchor_matcher = anchor_matcher  # 存储锚框匹配器
        self.box2box_transform = box2box_transform  # 存储边界框转换器
        self.batch_size_per_image = batch_size_per_image  # 每张图像的锚框采样数量
        self.positive_fraction = positive_fraction  # 正样本（前景）的采样比例
        # Map from self.training state to train/test settings
        # 根据训练状态映射训练/测试设置
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}  # NMS前保留的候选框数量
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}  # NMS后保留的候选框数量
        self.nms_thresh = nms_thresh  # NMS阈值
        self.min_box_size = float(min_box_size)  # 最小边界框尺寸
        self.anchor_boundary_thresh = anchor_boundary_thresh  # 锚框边界阈值
        if isinstance(loss_weight, float):  # 如果损失权重是浮点数
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}  # 转换为字典形式
        self.loss_weight = loss_weight  # 存储损失权重
        self.box_reg_loss_type = box_reg_loss_type  # 边界框回归损失类型
        self.smooth_l1_beta = smooth_l1_beta  # smooth L1损失的beta参数

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # 从配置文件构建RPN实例的类方法
        in_features = cfg.MODEL.RPN.IN_FEATURES  # 获取输入特征名称列表
        ret = {
            "in_features": in_features,  # 设置输入特征
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,  # 设置最小边界框尺寸
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,  # 设置NMS阈值
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,  # 设置每图像的批量大小
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,  # 设置正样本比例
            "loss_weight": {  # 设置损失权重
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,  # 分类损失权重
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,  # 定位损失权重
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,  # 设置锚框边界阈值
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),  # 创建边界框转换器
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,  # 设置边界框回归损失类型
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,  # 设置smooth L1损失的beta参数
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.
        随机采样正负样本的子集，并将未包含在采样中的所有元素的标签向量重写为忽略值(-1)。

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
            # 标签张量：一个包含-1、0、1的向量。将被就地修改并返回。
            # -1表示忽略，0表示负样本（背景），1表示正样本（前景）
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            # 每个特征图的锚框列表
            gt_instances: the ground-truth instances for each image.
            # 每张图像的真实标注实例

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
                # 图像数量大小的张量列表。第i个元素是一个标签向量，其长度是所有特征图上的锚框总数 R = sum(Hi * Wi * A)。
                # 标签值在{-1, 0, 1}中，含义为：-1 = 忽略；0 = 负类（背景）；1 = 正类（前景）。
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
                # 第i个元素是一个Rx4的张量。这些值是每个锚框匹配的真实框。
                # 对于标签不为1的锚框，其对应的值是未定义的。
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.
        返回一组RPN预测及其相关真实标注的损失。

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            # 每个特征图的锚框列表，每个形状为(Hi*Wi*A, B)，其中B是框的维度（4或5）
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            # L个元素的列表。第i个元素是形状为(N, Hi*Wi*A)的张量，表示所有锚框的预测目标性得分
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            # 真实标签列表：label_and_sample_anchors方法的输出
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            # L个元素的列表。第i个元素是形状为(N, Hi*Wi*A, 4或5)的张量，表示用于将锚框转换为建议框的预测"偏移量
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            # 长度为N的输入图像列表
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            # 输入数据，是从特征图名称到张量的映射。
            # 轴0表示输入数据中的图像数量N；
            # 轴1-3是通道数、高度和宽度，这些在不同特征图之间可能不同（例如，使用特征金字塔时）
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
            # 长度为N的Instances列表（可选）。每个Instances存储对应图像的真实标注实例。

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            # 建议框列表：包含"proposal_boxes"（建议框坐标）和"objectness_logits"（目标性得分）字段
            loss: dict[Tensor] or None
            # 损失字典或None
        """
        features = [features[f] for f in self.in_features]  # 获取指定的输入特征
        anchors = self.anchor_generator(features)  # 根据特征生成锚框

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)  # 通过RPN头部预测目标性得分和锚框偏移量
        # Transpose the Hi*Wi*A dimension to the middle:
        # 将Hi*Wi*A维度转置到中间：
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            # 将形状从(N, A, Hi, Wi)转换为(N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            # 将形状从(N, A*B, Hi, Wi)转换为(N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:  # 如果是训练模式
            assert gt_instances is not None, "RPN requires gt_instances in training!"  # 确保训练时有真实标注
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)  # 对锚框进行标注和采样
            losses = self.losses(  # 计算损失
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:  # 如果是测试模式
            losses = {}  # 不计算损失
        proposals = self.predict_proposals(  # 生成建议框
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses  # 返回建议框和损失

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.
        将所有预测的框回归偏移量解码为建议框。通过应用NMS和移除过小的框来找到最佳建议框。

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
            # 建议框列表：包含N个Instances。第i个Instances存储了图像i的post_nms_topk个目标建议框，
            # 按其目标性得分降序排序。
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes' coordinates that
        # are also network responses.
        # 在与ROI头部联合训练时，建议框被视为固定的。
        # 这种方法忽略了关于建议框坐标（也是网络响应）的导数。
        with torch.no_grad():  # 不计算梯度
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)  # 解码建议框
            return find_top_rpn_proposals(  # 找到最佳RPN建议框
                pred_proposals,  # 预测的建议框
                pred_objectness_logits,  # 预测的目标性得分
                image_sizes,  # 图像尺寸
                self.nms_thresh,  # NMS阈值
                self.pre_nms_topk[self.training],  # NMS前保留的建议框数量
                self.post_nms_topk[self.training],  # NMS后保留的建议框数量
                self.min_box_size,  # 最小框尺寸
                self.training,  # 是否处于训练模式
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        通过应用预测的锚框偏移量将锚框转换为建议框。

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
            # 建议框列表：包含L个张量。第i个张量的形状为(N, Hi*Wi*A, B)
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]  # 获取批次大小
        proposals = []  # 初始化建议框列表
        # For each feature map
        # 对每个特征图处理
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)  # 获取框的维度（4或5）
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)  # 重塑预测的锚框偏移量
            # Expand anchors to shape (N*Hi*Wi*A, B)
            # 将锚框扩展到形状(N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)  # 应用预测的偏移量转换锚框
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            # 添加形状为(N, Hi*Wi*A, B)的特征图建议框
            proposals.append(proposals_i.view(N, -1, B))
        return proposals  # 返回所有特征图的建议框
