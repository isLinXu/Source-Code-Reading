# Copyright (c) Facebook, Inc. and its affiliates.
# 版权所有 (c) Facebook, Inc. 及其附属公司。
from .config import CfgNode as CN  # 从config模块导入CfgNode类，并重命名为CN

# NOTE: given the new config system
# (https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html),
# we will stop adding new functionalities to default CfgNode.
# 注意：鉴于新的配置系统
# (https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html),
# 我们将不再为默认CfgNode添加新功能。

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# 关于训练/测试特定参数的约定
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# 每当一个参数既可用于训练又可用于测试时，
# 对应的名称将以_TRAIN作为训练参数的后缀，
# 或以_TEST作为测试特定参数的后缀。
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST
# 例如，训练期间的图像数量将是IMAGES_PER_BATCH_TRAIN，
# 而测试的图像数量将是IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# 配置定义
# -----------------------------------------------------------------------------

_C = CN()  # 创建一个配置节点实例

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
# 版本号，用于在发生任何更改时从旧配置升级到新配置。
# 建议在配置文件中保留VERSION。
_C.VERSION = 2  # 当前配置版本为2

_C.MODEL = CN()  # 创建模型配置节点
_C.MODEL.LOAD_PROPOSALS = False  # 是否加载预先计算的区域建议
_C.MODEL.MASK_ON = False  # 是否启用实例分割功能
_C.MODEL.KEYPOINT_ON = False  # 是否启用关键点检测功能
_C.MODEL.DEVICE = "cuda"  # 运行模型的设备，默认为CUDA
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"  # 模型的元架构

# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
# 加载到模型的检查点文件的路径（文件路径，或像detectron2://.., https://..这样的URL）。
# 可以在模型库中找到可用的模型。
_C.MODEL.WEIGHTS = ""  # 预训练模型权重路径

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
# 用于图像归一化的值（BGR顺序，因为INPUT.FORMAT默认为BGR）。
# 要在不同通道数的图像上训练，只需设置不同的均值和标准差。
# 默认值是ImageNet的平均像素值：[103.53, 116.28, 123.675]
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]  # 像素均值
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
# 当使用Detectron1中的预训练模型或任何MSRA模型时，
# 标准差已被吸收到其conv1权重中，所以标准差需要设置为1。
# 否则，可以使用[57.375, 57.120, 58.395]（ImageNet标准差）
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]  # 像素标准差


# -----------------------------------------------------------------------------
# INPUT
# 输入
# -----------------------------------------------------------------------------
_C.INPUT = CN()  # 创建输入配置节点
# By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
# Please refer to ResizeShortestEdge for detailed definition.
# 默认情况下，{MIN,MAX}_SIZE选项用于transforms.ResizeShortestEdge。
# 有关详细定义，请参阅ResizeShortestEdge。
# Size of the smallest side of the image during training
# 训练期间图像最短边的尺寸
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # 训练图像最短边的尺寸
# Sample size of smallest side by choice or random selection from range give by
# INPUT.MIN_SIZE_TRAIN
# 通过选择或从INPUT.MIN_SIZE_TRAIN给出的范围内随机选择最小边的采样大小
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"  # 训练时最小尺寸的采样方式
# Maximum size of the side of the image during training
# 训练期间图像边的最大尺寸
_C.INPUT.MAX_SIZE_TRAIN = 1333  # 训练图像的最大边长
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
# 测试期间图像最短边的尺寸。设置为零以在测试中禁用调整大小。
_C.INPUT.MIN_SIZE_TEST = 800  # 测试图像最短边的尺寸
# Maximum size of the side of the image during testing
# 测试期间图像边的最大尺寸
_C.INPUT.MAX_SIZE_TEST = 1333  # 测试图像的最大边长
# Mode for flipping images used in data augmentation during training
# choose one of ["horizontal, "vertical", "none"]
# 训练期间数据增强中用于翻转图像的模式
# 选择["horizontal", "vertical", "none"]之一
_C.INPUT.RANDOM_FLIP = "horizontal"  # 随机翻转模式

# `True` if cropping is used for data augmentation during training
# 如果在训练期间使用裁剪进行数据增强，则为`True`
_C.INPUT.CROP = CN({"ENABLED": False})  # 裁剪配置
# Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
# 裁剪类型。请参阅`detectron2.data.transforms.RandomCrop`的文档进行解释。
_C.INPUT.CROP.TYPE = "relative_range"  # 裁剪类型
# Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
# pixels if CROP.TYPE is "absolute"
# 如果CROP.TYPE是"relative"或"relative_range"，则裁剪大小在(0, 1]范围内，
# 如果CROP.TYPE是"absolute"，则以像素数为单位
_C.INPUT.CROP.SIZE = [0.9, 0.9]  # 裁剪尺寸


# Whether the model needs RGB, YUV, HSV etc.
# Should be one of the modes defined here, as we use PIL to read the image:
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# with BGR being the one exception. One can set image format to BGR, we will
# internally use RGB for conversion and flip the channels over
# 模型是否需要RGB、YUV、HSV等。
# 应该是这里定义的模式之一，因为我们使用PIL读取图像：
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# BGR是一个例外。可以将图像格式设置为BGR，我们将
# 在内部使用RGB进行转换并翻转通道
_C.INPUT.FORMAT = "BGR"  # 输入图像格式
# The ground truth mask format that the model will use.
# Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
# 模型将使用的真实掩码格式。
# Mask R-CNN支持"polygon"或"bitmask"作为真实掩码。
_C.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"  # 掩码格式


# -----------------------------------------------------------------------------
# Dataset
# 数据集
# -----------------------------------------------------------------------------
_C.DATASETS = CN()  # 创建数据集配置节点
# List of the dataset names for training. Must be registered in DatasetCatalog
# Samples from these datasets will be merged and used as one dataset.
# 训练数据集名称列表。必须在DatasetCatalog中注册
# 来自这些数据集的样本将被合并并用作一个数据集。
_C.DATASETS.TRAIN = ()  # 训练数据集列表
# List of the pre-computed proposal files for training, which must be consistent
# with datasets listed in DATASETS.TRAIN.
# 训练的预计算建议文件列表，必须与DATASETS.TRAIN中列出的数据集一致。
_C.DATASETS.PROPOSAL_FILES_TRAIN = ()  # 训练建议文件列表
# Number of top scoring precomputed proposals to keep for training
# 保留用于训练的顶级预计算建议的数量
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000  # 训练时保留的顶级建议数量
# List of the dataset names for testing. Must be registered in DatasetCatalog
# 测试数据集名称列表。必须在DatasetCatalog中注册
_C.DATASETS.TEST = ()  # 测试数据集列表
# List of the pre-computed proposal files for test, which must be consistent
# with datasets listed in DATASETS.TEST.
# 测试的预计算建议文件列表，必须与DATASETS.TEST中列出的数据集一致。
_C.DATASETS.PROPOSAL_FILES_TEST = ()  # 测试建议文件列表
# Number of top scoring precomputed proposals to keep for test
# 保留用于测试的顶级预计算建议的数量
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000  # 测试时保留的顶级建议数量

# -----------------------------------------------------------------------------
# DataLoader
# 数据加载器
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()  # 创建数据加载器配置节点
# Number of data loading threads
# 数据加载线程数
_C.DATALOADER.NUM_WORKERS = 4  # 工作线程数
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
# 如果为True，每个批次应只包含纵横比兼容的图像。
# 这将使竖直图像分组在一起，横向图像不会与竖直图像一起批处理。
_C.DATALOADER.ASPECT_RATIO_GROUPING = True  # 是否按纵横比分组
# Options: TrainingSampler, RepeatFactorTrainingSampler
# 选项：TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"  # 训练采样器类型
# Repeat threshold for RepeatFactorTrainingSampler
# RepeatFactorTrainingSampler的重复阈值
_C.DATALOADER.REPEAT_THRESHOLD = 0.0  # 重复采样阈值
# Tf True, when working on datasets that have instance annotations, the
# training dataloader will filter out images without associated annotations
# 如果为True，在处理具有实例注释的数据集时，
# 训练数据加载器将过滤掉没有相关注释的图像
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True  # 是否过滤无注释图像

# ---------------------------------------------------------------------------- #
# Backbone options
# 骨干网络选项
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()  # 创建骨干网络配置节点

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"  # 骨干网络构建函数名称
# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
# 冻结前几个阶段，使它们不被训练。
# ResNet中有5个阶段。第一个是卷积，后面的
# 阶段分别是残差块组。
_C.MODEL.BACKBONE.FREEZE_AT = 2  # 冻结的阶段数


# ---------------------------------------------------------------------------- #
# FPN options
# 特征金字塔网络选项
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()  # 创建FPN配置节点
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
# FPN使用的输入特征图的名称
# 它们必须具有连续的2的幂次步长
# 例如，["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.IN_FEATURES = []  # FPN输入特征
_C.MODEL.FPN.OUT_CHANNELS = 256  # FPN输出通道数

# Options: "" (no norm), "GN"
# 选项：""（无归一化），"GN"
_C.MODEL.FPN.NORM = ""  # FPN归一化类型

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
# FPN顶部和侧向特征融合的类型。可以是"sum"或"avg"
_C.MODEL.FPN.FUSE_TYPE = "sum"  # FPN融合类型


# ---------------------------------------------------------------------------- #
# Proposal generator options
# 提议生成器选项
# ---------------------------------------------------------------------------- #
_C.MODEL.PROPOSAL_GENERATOR = CN()  # 创建提议生成器配置节点
# Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
# 当前提议生成器包括"RPN", "RRPN"和"PrecomputedProposals"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"  # 提议生成器名称
# Proposal height and width both need to be greater than MIN_SIZE
# (a the scale used during training or inference)
# 提议高度和宽度都需要大于MIN_SIZE
# （训练或推理期间使用的比例）
_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0  # 提议最小尺寸


# ---------------------------------------------------------------------------- #
# Anchor generator options
# 锚点生成器选项
# ---------------------------------------------------------------------------- #
_C.MODEL.ANCHOR_GENERATOR = CN()  # 创建锚点生成器配置节点
# The generator can be any name in the ANCHOR_GENERATOR registry
# 生成器可以是ANCHOR_GENERATOR注册表中的任何名称
_C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"  # 锚点生成器名称
# Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
# Format: list[list[float]]. SIZES[i] specifies the list of sizes to use for
# IN_FEATURES[i]; len(SIZES) must be equal to len(IN_FEATURES) or 1.
# When len(SIZES) == 1, SIZES[0] is used for all IN_FEATURES.
# 以网络输入为参考的锚点大小（以像素为单位）。
# 格式：list[list[float]]。SIZES[i]指定用于
# IN_FEATURES[i]的尺寸列表；len(SIZES)必须等于len(IN_FEATURES)或1。
# 当len(SIZES) == 1时，SIZES[0]用于所有IN_FEATURES。
_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]  # 锚点大小
# Anchor aspect ratios. For each area given in `SIZES`, anchors with different aspect
# ratios are generated by an anchor generator.
# Format: list[list[float]]. ASPECT_RATIOS[i] specifies the list of aspect ratios (H/W)
# to use for IN_FEATURES[i]; len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
# or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0] is used
# for all IN_FEATURES.
# 锚点纵横比。对于给定的`SIZES`中的每个区域，
# 由锚点生成器生成的锚点具有不同的纵横比。
# 格式：list[list[float]]。ASPECT_RATIOS[i]指定用于
# IN_FEATURES[i]的纵横比列表；len(ASPECT_RATIOS) == len(IN_FEATURES)必须为真，
# 或len(ASPECT_RATIOS) == 1为真且纵横比列表ASPECT_RATIOS[0]用于
# 所有IN_FEATURES。
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]  # 锚点纵横比
# Anchor angles.
# list[list[float]], the angle in degrees, for each input feature map.
# ANGLES[i] specifies the list of angles for IN_FEATURES[i].
# 锚点角度。
# list[list[float]]，以度为单位的角，每个输入特征图。
# ANGLES[i]指定IN_FEATURES[i]的角度列表。
_C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]  # 锚点角度
# Relative offset between the center of the first anchor and the top-left corner of the image
# Value has to be in [0, 1). Recommend to use 0.5, which means half stride.
# 锚点与图像左上角的中心之间的相对偏移
# 值必须在[0, 1)中。建议使用0.5，这意味着半步长。
# 该值不期望影响模型精度。
_C.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0  # 锚点偏移

# ---------------------------------------------------------------------------- #
# RPN options
# RPN选项
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()  # 创建RPN配置节点
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY
# 名称由RPN_HEAD_REGISTRY使用

# Names of the input feature maps to be used by RPN
# e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
# RPN使用的输入特征图名称
# 例如，["p2", "p3", "p4", "p5", "p6"]用于FPN
_C.MODEL.RPN.IN_FEATURES = ["res4"]  # RPN输入特征
# Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
# 通过BOUNDARY_THRESH像素从图像中删除RPN锚点
# 设置为-1或大值，例如100000，以禁用修剪锚点
_C.MODEL.RPN.BOUNDARY_THRESH = -1  # 边界阈值
# IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example: 1)
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example: 0)
# Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
# are ignored (-1)
# 锚点与地面真实框之间的IOU重叠比率[BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# 最小重叠要求锚点与地面真实框对(anchor, gt box)对为正例(IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example: 1)
# 最大重叠允许锚点与地面真实框对(anchor, gt box)对为负例(IoU < BG_IOU_THRESHOLD
# ==> negative RPN example: 0)
# 重叠在(BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)之间的锚点
# 被忽略(-1)
_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]  # RPN IOU阈值
_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]  # RPN IOU标签
# Number of regions per image used to train RPN
# 每张图像用于训练RPN的区域数
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256  # RPN每张图像的区域数
# Target fraction of foreground (positive) examples per RPN minibatch
# RPN每批次的正例（正例）目标比例
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5  # RPN正例比例
# Options are: "smooth_l1", "giou", "diou", "ciou"
_C.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"  # RPN边界框回归损失类型
_C.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0  # RPN边界框回归损失权重
# Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
# RPN锚点回归目标的权重
_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)  # RPN边界框回归权重
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
# L1到L2损失的过渡点。设置为0.0以使损失简单L1。
_C.MODEL.RPN.SMOOTH_L1_BETA = 0.0  # RPN平滑L1 beta
_C.MODEL.RPN.LOSS_WEIGHT = 1.0  # RPN损失权重
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
# 在应用NMS之前保留前k个得分RPN提议
# 当使用FPN时，这是*每个FPN级别*（不是总数）
_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000  # RPN训练前NMS前k个提议
_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000  # RPN测试前NMS前k个提议
# Number of top scoring RPN proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See the "find_top_rpn_proposals" function for details.
# 在应用NMS之后保留前k个得分RPN提议
# 当使用FPN时，此限制将应用于每个级别，然后再应用于所有级别的提议的并集
# 注意：当使用FPN时，此配置的含义与Detectron1不同。
# 这意味着在Detectron1中是每批前k，但在每张图像前k这里。
# 有关详细信息，请参见"find_top_rpn_proposals"函数：
_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000  # RPN训练后NMS前k个提议
_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000  # RPN测试后NMS前k个提议
# NMS threshold used on RPN proposals
# RPN提议使用的NMS阈值
_C.MODEL.RPN.NMS_THRESH = 0.7  # RPN提议NMS阈值
# Set this to -1 to use the same number of output channels as input channels.
# 设置为-1以使用与输入通道相同的输出通道数。
_C.MODEL.RPN.CONV_DIMS = [-1]  # RPN卷积维度

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ROI头部选项
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()  # 创建ROI头部配置节点
_C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"  # ROI头部名称
# Number of foreground classes
# 前景类数量
_C.MODEL.ROI_HEADS.NUM_CLASSES = 80  # 前景类数量
# Names of the input feature maps to be used by ROI heads
# Currently all heads (box, mask, ...) use the same input feature map list
# e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
# 当前所有头部（框、掩码、...）使用相同的输入特征图列表
# 例如，["p2", "p3", "p4", "p5"]是常用作FPN
_C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]  # ROI输入特征
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
# ROI重叠比率[IOU_THRESHOLD]
# 如果RoI被视为背景（如果<IOU_THRESHOLD）
# 如果RoI被视为前景（如果>=IOU_THRESHOLD）
_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]  # ROI IOU阈值
_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]  # ROI标签
# RoI minibatch size *per image* (number of regions of interest [ROIs]) during training
# Total number of RoIs per training minibatch =
#   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 16 = 8192
# 训练期间每张图像的RoI minibatch大小*每张图像*（感兴趣区域[ROIs]的数量）
# 训练minibatch中的总RoI数量=
#   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
# 例如，常见配置为：512 * 16 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # ROI每张图像的批大小
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
# 目标RoI minibatch中标记为前景（即类>0）的比例
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25  # ROI正例比例

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# inference.
# 最小分数阈值（假设分数在[0, 1]范围内）；
# 选择一个值以平衡获得高召回率而不产生太多低精度
# 检测，这会减慢推理后处理步骤（如NMS）
# 默认阈值为0.0会增加AP约0.2-0.3，但会显著减慢
# 推理。
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # 测试最小分数阈值
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
# 用于非最大抑制（抑制IoU >=此阈值的框）
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # 测试NMS阈值
# If True, augment proposals with ground-truth boxes before sampling proposals to
# train ROI heads.
# 如果为True，在采样提议以训练ROI头部之前，增强提议与地面真实框
_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True  # 提议附加GT

# ---------------------------------------------------------------------------- #
# Box Head
# 框头
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()  # 创建框头配置节点
# C4 don't use head name option
# Options for non-C4 models: FastRCNNConvFCHead,
# 非C4模型选项：FastRCNNConvFCHead,
_C.MODEL.ROI_BOX_HEAD.NAME = ""  # 框头名称选项
# Options are: "smooth_l1", "giou", "diou", "ciou"
# 选项："smooth_l1", "giou", "diou", "ciou"
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"  # 框回归损失类型
# The final scaling coefficient on the box regression loss, used to balance the magnitude of its
# gradients with other losses in the model. See also `MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT`.
# 框回归损失的最终缩放系数，用于平衡其
# 与其他损失在模型中的梯度幅度。另见`MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT`。
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0  # 框回归损失权重
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# 框回归目标的默认权重
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # 框回归权重
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
# L1到L2损失的过渡点。设置为0.0以使损失简单L1。
_C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0  # 平滑L1 beta
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14  # 池化器分辨率
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0  # 池化器采样比例
# Type of pooling operation applied to the incoming feature map for each RoI
# 应用于每个RoI的池化操作
_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"  # 池化器类型

_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0  # 框头FC层数量
# Hidden layer dimension for FC layers in the RoI box head
# RoI框头中FC层的隐藏层维度
_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024  # 框头FC维度
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0  # 框头卷积层数量
# Channel dimension for Conv layers in the RoI box head
# RoI框头中卷积层的通道维度
_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256  # 框头卷积维度
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
# 卷积层的归一化方法。
# 选项：""（无归一化），"GN"，"SyncBN"。
_C.MODEL.ROI_BOX_HEAD.NORM = ""  # 卷积归一化方法
# Whether to use class agnostic for bbox regression
# 是否使用类不可知框回归
_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False  # 框回归类不可知
# If true, RoI heads use bounding boxes predicted by the box head rather than proposal boxes.
# 如果为真，RoI头部使用框头预测的边界框，而不是提议框。
_C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False  # 在预测框上训练

# ---------------------------------------------------------------------------- #
# Cascaded Box Head
# 级联框头
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_CASCADE_HEAD = CN()  # 创建级联框头配置节点
# The number of cascade stages is implicitly defined by the length of the following two configs.
# 级联阶段数由以下两个配置的长度隐式定义。
_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)  # 级联框回归权重
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)  # 级联IOU


# ---------------------------------------------------------------------------- #
# Mask Head
# 掩码头
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD = CN()  # 创建掩码头配置节点
_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"  # 掩码头名称
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14  # 池化器分辨率
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0  # 池化器采样比例
_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  # 掩码头卷积层数量
_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256  # 掩码头卷积维度
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
# 卷积层的归一化方法。
# 选项：""（无归一化），"GN"，"SyncBN"。
_C.MODEL.ROI_MASK_HEAD.NORM = ""  # 卷积归一化方法
# Whether to use class agnostic for mask prediction
# 是否使用类不可知掩码预测
_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False  # 掩码不可知
# Type of pooling operation applied to the incoming feature map for each RoI
# 应用于每个RoI的池化操作
_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"  # 池化器类型


# ---------------------------------------------------------------------------- #
# Keypoint Head
# 关键点头
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD = CN()  # 创建关键点头配置节点
_C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"  # 关键点头名称
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14  # 池化器分辨率
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0  # 池化器采样比例
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))  # 关键点头卷积维度
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  # 17是COCO中的关键点数量

# Images with too few (or no) keypoints are excluded from training.
# 如果图像中关键点太少（或没有），则从训练中排除。
_C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1  # 每张图像的最小关键点数量
# Normalize by the total number of visible keypoints in the minibatch if True.
# Otherwise, normalize by the total number of keypoints that could ever exist
# in the minibatch.
# The keypoint softmax loss is only calculated on visible keypoints.
# Since the number of visible keypoints can vary significantly between
# minibatches, this has the effect of up-weighting the importance of
# minibatches with few visible keypoints. (Imagine the extreme case of
# only one visible keypoint versus N: in the case of N, each one
# contributes 1/N to the gradient compared to the single keypoint
# determining the gradient direction). Instead, we can normalize the
# loss by the total number of keypoints, if it were the case that all
# keypoints were visible in a full minibatch. (Returning to the example,
# this means that the one visible keypoint contributes as much as each
# of the N keypoints.)
# 如果为真，则按每批中可见关键点的总数进行归一化。
# 否则，按每批中可能存在的关键点的总数进行归一化。
# 关键点softmax损失仅在可见关键点上计算。
# 由于每批中可见关键点的数量可以显著变化，这会影响每批中关键点的数量
# 最小批次的数量。（想象极端情况，只有一张图像，
# 在N的情况下，每个关键点
# 确定梯度方向）。相反，我们可以按每批中关键点的总数进行归一化，
# 如果所有关键点都在完整的批中可见，则意味着每个关键点
# 贡献与每批中的N个关键点一样多。（回到例子，
# 这意味着一个可见的关键点贡献与每批中的每个关键点一样多。）
_C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True  # 按可见关键点归一化损失
# Multi-task loss weight to use for keypoints
# Recommended values:
#   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
#   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
# 关键点损失权重
# 推荐值：
#   - 如果NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS为True，请使用1.0
#   - 如果NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS为False，请使用4.0
_C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0  # 关键点损失权重
# Type of pooling operation applied to the incoming feature map for each RoI
# 应用于每个RoI的池化操作
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"  # 池化器类型

# ---------------------------------------------------------------------------- #
# Semantic Segmentation Head
# 语义分割头
# ---------------------------------------------------------------------------- #
_C.MODEL.SEM_SEG_HEAD = CN()  # 创建语义分割头配置节点
_C.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"  # 语义分割头名称
_C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # 语义分割头输入特征
# Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# the correposnding pixel.
# 语义分割地面真实标签中被忽略的标签，即不计算相应像素的损失。
_C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255  # 语义分割忽略值
# Number of classes in the semantic segmentation head
# 语义分割头中的类数量
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54  # 语义分割头类数量
# Number of channels in the 3x3 convs inside semantic-FPN heads.
# 语义-FPN头部中3x3卷积中的通道数量。
_C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128  # 语义分割头卷积通道数
# Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
# 语义-FPN头部的输出缩放到COMMON_STRIDE步长。
_C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4  # 语义分割头公共步长
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
# 卷积层的归一化方法。选项：""（无归一化），"GN"。
_C.MODEL.SEM_SEG_HEAD.NORM = "GN"  # 语义分割头归一化方法
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0  # 语义分割头损失权重

_C.MODEL.PANOPTIC_FPN = CN()  # 创建PANOPTIC_FPN配置节点
# Scaling of all losses from instance detection / segmentation head.
# 实例检测/分割头损失的比例。
_C.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0  # 实例损失权重

# options when combining instance & semantic segmentation outputs
# 实例与语义分割输出组合选项
_C.MODEL.PANOPTIC_FPN.COMBINE = CN({"ENABLED": True})  # "COMBINE.ENABLED" is deprecated & not used
# "COMBINE.ENABLED"已弃用且未使用
_C.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5  # 重叠阈值
_C.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096  # 物品区域限制
_C.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5  # 实例置信阈值


# ---------------------------------------------------------------------------- #
# RetinaNet Head
# RetinaNet头
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()  # 创建RETINANET配置节点

# This is the number of foreground classes.
# 这是前景类的数量。
_C.MODEL.RETINANET.NUM_CLASSES = 80  # 前景类数量

_C.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]  # RETINANET输入特征

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
# 注意：这包括cls和bbox塔中的卷积
# 注意：这不包括最后一个conv用于logits
_C.MODEL.RETINANET.NUM_CONVS = 4  # RETINANET卷积数量

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
# 锚点与地面真实框之间的IOU重叠比率[bg, fg]用于标记锚点。
# 锚点< bg为负（0）
# 锚点>= bg且< fg为负（-1）
# 锚点>= fg为正（1）
_C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]  # RETINANET IOU阈值
_C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]  # RETINANET IOU标签

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
# 在训练开始时为罕见情况（即前景）设置logits层的偏置。
# 这有助于在类不平衡的情况下稳定训练。
_C.MODEL.RETINANET.PRIOR_PROB = 0.01  # RETINANET前景概率

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
# 推理cls分数阈值，仅考虑分数>INFERENCE_TH的锚点进行推理（以提高速度）
_C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05  # RETINANET测试分数阈值
# Select topk candidates before NMS
# 在NMS之前选择topk候选
_C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000  # RETINANET测试前k个候选
_C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5  # RETINANET测试NMS阈值

# Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
# RETINANET锚点回归目标的权重
_C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)  # RETINANET边界框回归权重

# Loss parameters
# 损失参数
_C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0  # RETINANET焦损失伽玛
_C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25  # RETINANET焦损失阿尔法
_C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1  # RETINANET平滑L1损失beta
# Options are: "smooth_l1", "giou", "diou", "ciou"
# 选项："smooth_l1", "giou", "diou", "ciou"
_C.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"  # RETINANET边界框回归损失类型

# One of BN, SyncBN, FrozenBN, GN
# Only supports GN until unshared norm is implemented
# 一个BN, SyncBN, FrozenBN, GN
# 仅支持GN直到未共享归一化实现
_C.MODEL.RETINANET.NORM = ""  # RETINANET归一化


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# 注意：ResNet = {ResNet, ResNeXt}
# 注意：ResNet中的部分可以用于骨干和头部
# 这些选项适用于两者
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()  # 创建RESNETS配置节点

_C.MODEL.RESNETS.DEPTH = 50  # RESNETS深度
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4用于C4骨干，res2..5用于FPN骨干
# 输出功能

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
# 使用的组数；1 ==> ResNet；> 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1  # RESNETS组数

# Options: FrozenBN, GN, "SyncBN", "BN"
# 选项：FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.RESNETS.NORM = "FrozenBN"  # RESNETS归一化

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
# 每个组的基准宽度。
# 缩放此参数将缩放所有瓶颈层。
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64  # RESNETS每组宽度

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
# 将步长为2的卷积放在1x1滤波器上
# 仅对原始MSRA ResNet使用True；对C2和Torch模型使用False
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True  # 是否在1x1卷积中使用步长

# Apply dilation in stage "res5"
# 在"res5"阶段应用膨胀
_C.MODEL.RESNETS.RES5_DILATION = 1  # res5阶段的膨胀率

# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# For R18 and R34, this needs to be set to 64
# res2的输出宽度。缩放此参数将缩放ResNet中所有1x1卷积的宽度
# 对于R18和R34，这需要设置为64
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256  # res2阶段的输出通道数
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64  # stem阶段的输出通道数

# Apply Deformable Convolution in stages
# Specify if apply deform_conv on Res2, Res3, Res4, Res5
# 在各阶段应用可变形卷积
# 指定是否在Res2、Res3、Res4、Res5上应用可变形卷积
_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]  # 每个阶段是否使用可变形卷积
# Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
# Use False for DeformableV1.
# 使用True以使用调制可变形卷积（DeformableV2，https://arxiv.org/abs/1811.11168）；
# 使用False则使用DeformableV1。
_C.MODEL.RESNETS.DEFORM_MODULATED = False  # 是否使用调制可变形卷积
# Number of groups in deformable conv.
# 可变形卷积中的组数
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1  # 可变形卷积的组数


# ---------------------------------------------------------------------------- #
# Solver
# 求解器
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()  # 创建求解器配置节点

# Options: WarmupMultiStepLR, WarmupCosineLR.
# See detectron2/solver/build.py for definition.
# 选项：WarmupMultiStepLR，WarmupCosineLR。
# 见detectron2/solver/build.py中的定义。
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"  # 学习率调度器名称

_C.SOLVER.MAX_ITER = 40000  # 最大迭代次数

_C.SOLVER.BASE_LR = 0.001  # 基础学习率

_C.SOLVER.MOMENTUM = 0.9  # 动量

_C.SOLVER.NESTEROV = False  # 是否使用Nesterov动量

_C.SOLVER.WEIGHT_DECAY = 0.0001  # 权重衰减
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
# 应用于归一化层参数的权重衰减
# （通常是仿射变换）
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0  # 归一化层的权重衰减

_C.SOLVER.GAMMA = 0.1  # 学习率衰减因子
# The iteration number to decrease learning rate by GAMMA.
# 通过GAMMA减少学习率的迭代次数。
_C.SOLVER.STEPS = (30000,)  # 学习率衰减步骤

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000  # 热身因子
_C.SOLVER.WARMUP_ITERS = 1000  # 热身迭代次数
_C.SOLVER.WARMUP_METHOD = "linear"  # 热身方法

# Save a checkpoint after every this number of iterations
# 每隔这么多次迭代后保存一个检查点
_C.SOLVER.CHECKPOINT_PERIOD = 5000  # 检查点保存周期

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration). If we use 16 GPUs
# and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
# 所有机器上每批的图像数。这也是每步（即每次迭代）的
# 训练图像数。如果我们使用16个GPU
# 且IMS_PER_BATCH = 32，每个GPU将看到每批2张图像。
# 如果设置了REFERENCE_WORLD_SIZE，可能会自动调整。
_C.SOLVER.IMS_PER_BATCH = 16  # 每批图像数

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
# 此配置旨在训练的参考工作者（GPU）数量。
# 当设置为0时不起作用。
# 使用非零值时，DefaultTrainer将用它来计算所需的
# 每个工作者的批大小，然后缩放其他相关配置（总批大小，
# 学习率等）以匹配每个工作者的批大小。
# 有关详细信息，请参见`DefaultTrainer.auto_scale_workers`的文档：
_C.SOLVER.REFERENCE_WORLD_SIZE = 0  # 参考世界大小

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
# Detectron v1（和之前的检测代码）对偏置使用了2倍高的LR和0 WD。
# 这不是很有用（至少对于最近的模型）。你应该避免
# 更改这些，它们的存在只是为了在需要时重现Detectron v1训练。
_C.SOLVER.BIAS_LR_FACTOR = 1.0  # 偏置学习率因子
_C.SOLVER.WEIGHT_DECAY_BIAS = None  # None means following WEIGHT_DECAY  # 偏置权重衰减，None表示跟随WEIGHT_DECAY

# Gradient clipping
# 梯度裁剪
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})  # 梯度裁剪配置
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
# 梯度裁剪的类型，目前支持两个值：
# - "value"：每个梯度元素的绝对值被裁剪
# - "norm"：每个参数的梯度范数被裁剪，因此
#   影响参数中的所有元素
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"  # 裁剪类型
# Maximum absolute value used for clipping gradients
# 用于裁剪梯度的最大绝对值
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0  # 裁剪值
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
# 用于"norm"梯度裁剪类型的L-p范数的浮点数p；
# 对于L-inf，请指定.inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0  # 范数类型

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
# 启用训练的自动混合精度
# 注意，这不会改变模型的推理行为。
# 要在推理中使用AMP，请在autocast()下运行推理
_C.SOLVER.AMP = CN({"ENABLED": False})  # 自动混合精度配置

# ---------------------------------------------------------------------------- #
# Specific test options
# 特定测试选项
# ---------------------------------------------------------------------------- #
_C.TEST = CN()  # 创建测试配置节点
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
# 用于端到端测试以验证预期的准确性。
# 每个项目是[任务，指标，值，容差]
# 例如：[['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []  # 预期结果
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
# 在训练期间评估模型的周期（以步数为单位）。
# 设置为0以禁用。
_C.TEST.EVAL_PERIOD = 0  # 评估周期
# The sigmas used to calculate keypoint OKS. See http://cocodataset.org/#keypoints-eval
# When empty, it will use the defaults in COCO.
# Otherwise it should be a list[float] with the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
# 用于计算关键点OKS的sigma值。参见http://cocodataset.org/#keypoints-eval
# 当为空时，将使用COCO中的默认值。
# 否则应该是与ROI_KEYPOINT_HEAD.NUM_KEYPOINTS长度相同的list[float]。
_C.TEST.KEYPOINT_OKS_SIGMAS = []  # 关键点OKS sigma值
# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
# 推理期间每张图像返回的最大检测数（100是
# 基于为COCO数据集建立的限制）。
_C.TEST.DETECTIONS_PER_IMAGE = 100  # 每图像的检测数量

_C.TEST.AUG = CN({"ENABLED": False})  # 测试时增强配置
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)  # 测试增强的最小尺寸
_C.TEST.AUG.MAX_SIZE = 4000  # 测试增强的最大尺寸
_C.TEST.AUG.FLIP = True  # 测试是否进行翻转增强

_C.TEST.PRECISE_BN = CN({"ENABLED": False})  # 精确批归一化配置
_C.TEST.PRECISE_BN.NUM_ITER = 200  # 精确批归一化的迭代次数

# ---------------------------------------------------------------------------- #
# Misc options
# 杂项选项
# ---------------------------------------------------------------------------- #
# Directory where output files are written
# 输出文件的写入目录
_C.OUTPUT_DIR = "./output"  # 输出目录
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
# 将种子设置为负数以完全随机化一切。
# 将种子设置为正数以使用固定种子。注意，固定种子增加了
# 可重复性，但不能保证完全确定性的行为。
# 禁用所有并行性进一步增加可重复性。
_C.SEED = -1  # 随机种子
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
# 对不同的cudnn算法进行基准测试。
# 如果输入图像的大小差异很大，此选项将有大量开销
# 大约10k次迭代。它通常会损害总时间，但对某些模型有益。
# 如果输入图像大小相同或相似，基准测试通常是有帮助的。
_C.CUDNN_BENCHMARK = False  # 是否使用CUDNN基准测试
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
# 训练时小批量可视化的周期（以步为单位）。
# 设置为0以禁用。
_C.VIS_PERIOD = 0  # 可视化周期

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
# 全局配置用于快速黑客目的。
# 你可以在命令行或配置文件中设置它们，
# 并通过以下方式访问：
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
# 不要向其中提交任何配置。
_C.GLOBAL = CN()  # 全局配置节点
_C.GLOBAL.HACK = 1.0  # 全局黑客值
