# Copyright (c) Facebook, Inc. and its affiliates.
# 导入所需的Python标准库和第三方库
import collections  # 用于处理集合数据类型
import math  # 用于数学计算
from typing import List  # 用于类型注解
import torch  # PyTorch深度学习框架
from torch import nn  # PyTorch神经网络模块

# 导入detectron2相关模块
from detectron2.config import configurable  # 用于配置系统
from detectron2.layers import ShapeSpec  # 用于指定特征图的形状
from detectron2.structures import Boxes, RotatedBoxes  # 用于表示边界框和旋转边界框
from detectron2.utils.registry import Registry  # 用于注册模块

# 创建锚框生成器的注册表
ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
ANCHOR_GENERATOR_REGISTRY.__doc__ = """
Registry for modules that creates object detection anchors for feature maps.
用于创建目标检测特征图锚框的模块注册表。

The registered object will be called with `obj(cfg, input_shape)`.
注册的对象将通过`obj(cfg, input_shape)`调用。
"""


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    类似于nn.ParameterList，但用于缓冲区而不是参数
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            # 使用非持久性缓冲区，这样数值不会保存在检查点中
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        # 返回缓冲区的数量
        return len(self._buffers)

    def __iter__(self):
        # 返回缓冲区值的迭代器
        return iter(self._buffers.values())


def _create_grid_offsets(size: List[int], stride: int, offset: float, device: torch.device):
    """创建网格偏移量
    Args:
        size: 特征图的高度和宽度
        stride: 特征图的步长
        offset: 锚点相对于网格的偏移量
        device: 计算设备
    Returns:
        shift_x, shift_y: 网格点的x和y坐标偏移量
    """
    grid_height, grid_width = size  # 解析特征图的高度和宽度
    # 计算x方向的偏移量
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device
    )
    # 计算y方向的偏移量
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )

    # 使用meshgrid生成网格坐标
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    # 将坐标展平为一维张量
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.
    如果指定了一个尺寸（或宽高比），并且有多个特征图，我们将这个单一尺寸（或宽高比）的锚框
    广播到所有特征图上。

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.
    如果params是list[float]，或者是长度为1的list[list[float]]，将其重复num_features次。

    Returns:
        list[list[float]]: param for each feature
        为每个特征层返回参数列表
    """
    # 确保参数是序列类型
    assert isinstance(
        params, collections.abc.Sequence
    ), f"{name} in anchor generator has to be a list! Got {params}."
    # 确保参数非空
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]
        # 如果参数是浮点数列表，复制到所有特征层
        return [params] * num_features
    if len(params) == 1:
        # 如果参数列表长度为1，复制到所有特征层
        return list(params) * num_features
    # 确保参数数量与特征层数量匹配
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params


@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    按照Faster R-CNN论文中描述的标准方式计算锚框。
    """

    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    每个锚框的维度。
    """

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        """
        This interface is experimental.
        此接口为实验性质。

        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
                如果sizes是list[list[float]]，sizes[i]是用于第i个特征图的锚框尺寸列表
                （即锚框面积的平方根）。
                如果sizes是list[float]，则这些尺寸用于所有特征图。
                锚框尺寸以输入图像的绝对长度为单位；当输入图像尺寸改变时，它们不会动态缩放。

            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
                锚框使用的宽高比列表（即高度/宽度）。与sizes使用相同的"广播"规则。

            strides (list[int]): stride of each input feature.
                每个输入特征图的步长。

            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
                第一个锚框中心与图像左上角之间的相对偏移量。值必须在[0, 1)范围内。
                建议使用0.5，表示半个步长。
        """
        super().__init__()

        self.strides = strides  # 设置特征图的步长
        self.num_features = len(self.strides)  # 特征图的数量
        # 广播尺寸和宽高比参数到所有特征层
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        # 计算每个特征层的基础锚框
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = offset  # 设置锚框中心的偏移量
        assert 0.0 <= self.offset < 1.0, self.offset  # 确保偏移量在有效范围内

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        """从配置文件构建锚框生成器的参数"""
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,  # 锚框尺寸
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,  # 锚框宽高比
            "strides": [x.stride for x in input_shape],  # 特征图步长
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,  # 锚框偏移量
        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        """计算每个特征层的基础锚框"""
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        num_anchors的别名。
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_anchors` on every feature map is the same.
            返回一个整数列表，每个整数表示特征图上每个像素位置的锚框数量。
            例如，如果在每个像素位置使用3种宽高比和5种尺寸的锚框，则锚框数量为15。
            （参见配置中的ANCHOR_GENERATOR.SIZES和ANCHOR_GENERATOR.ASPECT_RATIOS）

            在标准RPN模型中，每个特征图上的锚框数量相同。
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
            返回一个张量列表，每个张量对应一个特征图，形状为(位置数量 x 每个位置的锚框数量) x 4
        """
        anchors = []  # 存储所有特征图的锚框
        # buffers() not supported by torchscript. use named_buffers() instead
        # buffers()不被torchscript支持，改用named_buffers()
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            # 为每个特征图创建网格偏移
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            # 将x和y偏移堆叠成边界框格式(x1, y1, x2, y2)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # 将偏移应用到基础锚框上，生成特征图上所有位置的锚框
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).
        生成一个张量，存储规范的锚框，这些锚框具有不同的尺寸和宽高比，都以(0, 0)为中心。
        我们可以通过平移和平铺这些张量来构建完整特征图的锚框集合（参见`meth:_grid_anchors`）。

        Args:
            sizes (tuple[float]): 锚框的尺寸列表
            aspect_ratios (tuple[float]]): 锚框的宽高比列表

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
            返回形状为(尺寸数量 * 宽高比数量, 4)的张量，以XYXY格式存储锚框。
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227
        # 这与原始Faster R-CNN代码或Detectron中定义的锚框生成器不同。它们产生相同的AP，
        # 但旧版本以一种不太自然的方式定义单元锚框，使用相对于特征网格的偏移和量化，
        # 导致不同宽高比的锚框大小略有不同。
        # 参见 https://github.com/facebookresearch/Detectron/issues/227

        anchors = []  # 存储生成的锚框
        for size in sizes:  # 遍历所有尺寸
            area = size ** 2.0  # 计算锚框面积
            for aspect_ratio in aspect_ratios:  # 遍历所有宽高比
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                # 通过面积和宽高比计算宽度和高度
                w = math.sqrt(area / aspect_ratio)  # 计算宽度
                h = aspect_ratio * w  # 计算高度
                # 计算锚框的左上角和右下角坐标
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.
            特征图列表，用于生成锚框的骨干网络特征图。

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
            返回一个Boxes列表，包含每个特征图的所有锚框
                （即在特征图所有位置重复的单元锚框）。
                每个特征图的锚框数量为Hi x Wi x num_cell_anchors，
                其中Hi、Wi是特征图分辨率除以锚框步长的结果。
        """
        # 获取每个特征图的空间尺寸
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        # 为所有特征图生成锚框
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        # 将锚框转换为Boxes对象
        return [Boxes(x) for x in anchors_over_all_feature_maps]


@ANCHOR_GENERATOR_REGISTRY.register()  # 注册旋转锚框生成器到注册表
class RotatedAnchorGenerator(nn.Module):  # 继承自nn.Module的旋转锚框生成器类
    """
    Compute rotated anchors used by Rotated RPN (RRPN), described in
    "Arbitrary-Oriented Scene Text Detection via Rotation Proposals".
    计算旋转区域建议网络(RRPN)使用的旋转锚框，在论文"基于旋转建议的任意方向场景文本检测"中描述。
    """

    box_dim: int = 5  # 设置锚框维度为5（x, y, w, h, angle）
    """
    the dimension of each anchor box.
    每个锚框的维度。
    """

    @configurable  # 使类可配置的装饰器
    def __init__(self, *, sizes, aspect_ratios, strides, angles, offset=0.5):
        """
        This interface is experimental.
        此接口为实验性质。

        Args:
            sizes (list[list[float]] or list[float]):
                If sizes is list[list[float]], sizes[i] is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If sizes is list[float], the sizes are used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
                如果sizes是list[list[float]]，sizes[i]是用于第i个特征图的锚框尺寸列表（即锚框面积的平方根）。
                如果sizes是list[float]，则这些尺寸用于所有特征图。
                锚框尺寸以输入图像的绝对长度为单位，当输入图像尺寸改变时不会动态缩放。

            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
                锚框使用的宽高比列表（即高度/宽度）。与sizes使用相同的"广播"规则。

            strides (list[int]): stride of each input feature.
                每个输入特征图的步长。

            angles (list[list[float]] or list[float]): list of angles (in degrees CCW)
                to use for anchors. Same "broadcast" rule for `sizes` applies.
                锚框使用的角度列表（逆时针角度）。与sizes使用相同的"广播"规则。

            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
                第一个锚框中心与图像左上角之间的相对偏移量。值必须在[0, 1)范围内。
                建议使用0.5，表示半个步长。
        """
        super().__init__()  # 调用父类初始化方法

        self.strides = strides  # 存储特征图步长
        self.num_features = len(self.strides)  # 计算特征图数量
        # 对尺寸、宽高比和角度参数进行广播，使其与特征图数量匹配
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        angles = _broadcast_params(angles, self.num_features, "angles")
        # 计算基础锚框
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios, angles)

        self.offset = offset  # 存储锚框中心偏移量
        assert 0.0 <= self.offset < 1.0, self.offset  # 确保偏移量在有效范围内

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        """
        从配置文件构建旋转锚框生成器的参数
        """
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,  # 锚框尺寸
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,  # 锚框宽高比
            "strides": [x.stride for x in input_shape],  # 特征图步长
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,  # 锚框偏移量
            "angles": cfg.MODEL.ANCHOR_GENERATOR.ANGLES,  # 锚框旋转角度
        }

    def _calculate_anchors(self, sizes, aspect_ratios, angles):
        """
        计算每个特征层的基础锚框
        """
        cell_anchors = [
            self.generate_cell_anchors(size, aspect_ratio, angle).float()
            for size, aspect_ratio, angle in zip(sizes, aspect_ratios, angles)
        ]
        return BufferList(cell_anchors)  # 返回锚框缓冲列表

    @property
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        num_anchors的别名。
        """
        return self.num_anchors

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios, 2 sizes and 5 angles, the number of anchors is 30.
                (See also ANCHOR_GENERATOR.SIZES, ANCHOR_GENERATOR.ASPECT_RATIOS
                and ANCHOR_GENERATOR.ANGLES in config)

                In standard RRPN models, `num_anchors` on every feature map is the same.
            返回一个整数列表，每个整数表示特征图上每个像素位置的锚框数量。
            例如，如果在每个像素位置使用3种宽高比、2种尺寸和5种角度的锚框，则锚框数量为30。
            （参见配置中的ANCHOR_GENERATOR.SIZES、ANCHOR_GENERATOR.ASPECT_RATIOS
            和ANCHOR_GENERATOR.ANGLES）

            在标准RRPN模型中，每个特征图上的锚框数量相同。
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]  # 返回每个特征图的锚框数量

    def _grid_anchors(self, grid_sizes):
        """
        生成特征图网格上的所有锚框
        """
        anchors = []  # 存储所有特征图的锚框
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            # 为每个特征图创建网格偏移
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            zeros = torch.zeros_like(shift_x)  # 创建与shift_x相同形状的零张量
            # 堆叠偏移量和零张量，形成5维向量(x, y, 0, 0, 0)
            shifts = torch.stack((shift_x, shift_y, zeros, zeros, zeros), dim=1)

            # 将偏移应用到基础锚框上，生成特征图上所有位置的锚框
            anchors.append((shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)).reshape(-1, 5))

        return anchors

    def generate_cell_anchors(
        self,
        sizes=(32, 64, 128, 256, 512),  # 默认锚框尺寸
        aspect_ratios=(0.5, 1, 2),  # 默认宽高比
        angles=(-90, -60, -30, 0, 30, 60, 90),  # 默认旋转角度
    ):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes, aspect_ratios, angles centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).
        生成一个张量，存储规范的锚框，这些锚框具有不同的尺寸、宽高比和角度，都以(0, 0)为中心。
        我们可以通过平移和平铺这些张量来构建完整特征图的锚框集合（参见`meth:_grid_anchors`）。

        Args:
            sizes (tuple[float]): 锚框尺寸列表
            aspect_ratios (tuple[float]]): 锚框宽高比列表
            angles (tuple[float]]): 锚框旋转角度列表

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios) * len(angles), 5)
                storing anchor boxes in (x_ctr, y_ctr, w, h, angle) format.
            返回形状为(尺寸数量 * 宽高比数量 * 角度数量, 5)的张量，
            以(x_ctr, y_ctr, w, h, angle)格式存储锚框。
        """
        anchors = []  # 存储生成的锚框
        for size in sizes:  # 遍历所有尺寸
            area = size ** 2.0  # 计算锚框面积
            for aspect_ratio in aspect_ratios:  # 遍历所有宽高比
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                # 通过面积和宽高比计算宽度和高度
                w = math.sqrt(area / aspect_ratio)  # 计算宽度
                h = aspect_ratio * w  # 计算高度
                # 为每个角度生成一个锚框，格式为(x_ctr, y_ctr, w, h, angle)
                anchors.extend([0, 0, w, h, a] for a in angles)

        return torch.tensor(anchors)  # 将锚框列表转换为张量

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.
            特征图列表，用于生成锚框的骨干网络特征图。

        Returns:
            list[RotatedBoxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
            返回一个RotatedBoxes列表，包含每个特征图的所有锚框
                （即在特征图所有位置重复的单元锚框）。
                每个特征图的锚框数量为Hi x Wi x num_cell_anchors，
                其中Hi、Wi是特征图分辨率除以锚框步长的结果。
        """
        # 获取每个特征图的空间尺寸
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        # 为所有特征图生成锚框
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        # 将锚框转换为RotatedBoxes对象并返回
        return [RotatedBoxes(x) for x in anchors_over_all_feature_maps]


def build_anchor_generator(cfg, input_shape):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    根据配置文件中的`cfg.MODEL.ANCHOR_GENERATOR.NAME`构建一个锚框生成器。

    Args:
        cfg: 配置对象，包含了锚框生成器的配置信息，特别是MODEL.ANCHOR_GENERATOR部分
        input_shape: 输入特征图的形状信息，用于确定锚框生成器的参数

    Returns:
        返回一个已注册的锚框生成器实例，该实例根据配置文件中指定的类型创建
    """
    # 从配置文件中获取指定的锚框生成器名称
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    # 通过注册表获取对应的锚框生成器类，并使用配置和输入形状信息实例化该生成器
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)
