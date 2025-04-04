# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
实现多个有用的数据增强类。
"""
import numpy as np
import sys
from typing import Tuple
import torch
from fvcore.transforms.transform import (
    BlendTransform,    # 混合变换
    CropTransform,     # 裁剪变换
    HFlipTransform,    # 水平翻转变换
    NoOpTransform,     # 无操作变换
    PadTransform,      # 填充变换
    Transform,         # 基础变换类
    TransformList,     # 变换列表
    VFlipTransform,    # 垂直翻转变换
)
from PIL import Image

from .augmentation import Augmentation, _transform_to_aug
from .transform import ExtentTransform, ResizeTransform, RotationTransform

__all__ = [
    "FixedSizeCrop",              # 固定大小裁剪
    "RandomApply",                # 随机应用
    "RandomBrightness",          # 随机亮度
    "RandomContrast",            # 随机对比度
    "RandomCrop",                # 随机裁剪
    "RandomExtent",              # 随机范围
    "RandomFlip",                # 随机翻转
    "RandomSaturation",          # 随机饱和度
    "RandomLighting",            # 随机光照
    "RandomRotation",            # 随机旋转
    "Resize",                    # 调整大小
    "ResizeScale",               # 缩放调整
    "ResizeShortestEdge",        # 调整最短边
    "RandomCrop_CategoryAreaConstraint",  # 带类别面积约束的随机裁剪
]


class RandomApply(Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    以给定的概率随机应用一个数据增强操作。
    """

    def __init__(self, tfm_or_aug, prob=0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
                要应用的变换或增强操作，可以是`Transform`或`Augmentation`实例。
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
                应用包装变换的概率，取值在0.0到1.0之间。
        """
        super().__init__()
        # 将输入的变换或增强转换为增强实例
        self.aug = _transform_to_aug(tfm_or_aug)
        # 确保概率值在有效范围内
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob

    def get_transform(self, *args):
        # 根据概率决定是否应用变换
        do = self._rand_range() < self.prob
        if do:
            # 如果应用变换，返回增强实例的变换
            return self.aug.get_transform(*args)
        else:
            # 如果不应用变换，返回无操作变换
            return NoOpTransform()

    def __call__(self, aug_input):
        # 直接调用时的行为与get_transform类似
        do = self._rand_range() < self.prob
        if do:
            # 应用增强操作
            return self.aug(aug_input)
        else:
            # 返回无操作变换
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    以给定的概率对图像进行水平或垂直翻转。
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
                翻转的概率。
            horizontal (boolean): whether to apply horizontal flipping
                是否进行水平翻转。
            vertical (boolean): whether to apply vertical flipping
                是否进行垂直翻转。
        """
        super().__init__()

        # 不允许同时进行水平和垂直翻转
        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        # 至少要启用一种翻转方式
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, image):
        # 获取图像尺寸
        h, w = image.shape[:2]
        # 根据概率决定是否进行翻转
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                # 返回水平翻转变换
                return HFlipTransform(w)
            elif self.vertical:
                # 返回垂直翻转变换
                return VFlipTransform(h)
        else:
            # 不进行翻转时返回无操作变换
            return NoOpTransform()


class Resize(Augmentation):
    """Resize image to a fixed target size
    将图像调整为固定的目标大小
    """

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
                目标尺寸，可以是(高度,宽度)元组或单个整数(此时表示正方形)
            interp: PIL interpolation method
                PIL图像插值方法
        """
        # 如果shape是单个整数，转换为正方形尺寸元组
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        # 返回调整大小的变换
        # 参数依次为：原始高度、原始宽度、目标高度、目标宽度、插值方法
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    调整图像大小，同时保持纵横比不变。
    尝试将较短边缩放到给定的`short_edge_length`，
    只要较长边不超过`max_size`。
    如果达到`max_size`，则进行下采样使较长边不超过max_size。
    """

    @torch.jit.unused
    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
                如果``sample_style=="range"``，表示采样最短边长度的[最小值,最大值]区间。
                如果``sample_style=="choice"``，表示可供选择的最短边长度列表。
            max_size (int): maximum allowed longest edge length.
                允许的最大边长。
            sample_style (str): either "range" or "choice".
                采样方式，可以是"range"或"choice"。
        """
        super().__init__()
        # 验证采样方式的有效性
        assert sample_style in ["range", "choice"], sample_style

        # 设置是否为范围采样模式
        self.is_range = sample_style == "range"
        # 如果输入的是单个整数，转换为相同的区间
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        # 范围模式下验证输入长度
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    @torch.jit.unused
    def get_transform(self, image):
        # 获取图像尺寸
        h, w = image.shape[:2]
        # 根据采样方式选择目标大小
        if self.is_range:
            # 范围模式：在区间内随机采样
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            # 选择模式：从列表中随机选择
            size = np.random.choice(self.short_edge_length)
        # 如果目标大小为0，返回无操作变换
        if size == 0:
            return NoOpTransform()

        # 计算保持纵横比的新尺寸
        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        # 返回调整大小的变换
        return ResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        根据输入尺寸和目标短边长度计算输出尺寸。
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        # 计算缩放比例，使短边达到目标大小
        scale = size / min(h, w)
        # 根据图像的宽高比例计算新的尺寸
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        # 如果任一边超过最大尺寸，进行等比例缩放
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        # 四舍五入到整数
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ResizeScale(Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self._init(locals())

    def _get_resize(self, image: np.ndarray, scale: float) -> Transform:
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        return ResizeTransform(
            input_size[0], input_size[1], output_size[0], output_size[1], self.interp
        )

    def get_transform(self, image: np.ndarray) -> Transform:
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        return self._get_resize(image, random_scale)


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class FixedSizeCrop(Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(self, crop_size: Tuple[int], pad: bool = True, pad_value: float = 128.0):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value.
        """
        super().__init__()
        self._init(locals())

    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        return CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )

    def _get_pad(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        return PadTransform(
            0, 0, pad_size[1], pad_size[0], original_size[1], original_size[0], self.pad_value
        )

    def get_transform(self, image: np.ndarray) -> TransformList:
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return TransformList(transforms)


class RandomCrop(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
                裁剪类型，可以是以下四种之一："relative_range"、"relative"、"absolute"、"absolute_range"
            crop_size (tuple[float, float]): two floats, explained below.
                两个浮点数，具体含义如下所述

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
          从尺寸为(H, W)的输入图像中裁剪出(H * crop_size[0], W * crop_size[1])大小的区域。
          crop_size的值应该在(0, 1]范围内
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
          从[crop_size[0], 1]和[crop_size[1], 1]中均匀采样两个值，
          然后像"relative"类型一样使用这些值
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
          从输入图像中裁剪出(crop_size[0], crop_size[1])大小的区域。
          crop_size必须小于输入图像尺寸
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
          对于尺寸为(H, W)的输入，在[crop_size[0], min(H, crop_size[1])]范围内均匀采样H_crop，
          在[crop_size[0], min(W, crop_size[1])]范围内均匀采样W_crop。
          然后裁剪出(H_crop, W_crop)大小的区域。
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
                图像尺寸：高度，宽度

        Returns:
            crop_size (tuple): height, width in absolute pixels
                裁剪尺寸：以绝对像素表示的高度和宽度
        """
        h, w = image_size  # 获取图像的高度和宽度
        if self.crop_type == "relative":
            # 相对裁剪：按比例计算裁剪尺寸
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)  # 四舍五入到整数
        elif self.crop_type == "relative_range":
            # 相对范围裁剪：在给定范围内随机生成裁剪比例
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)  # 四舍五入到整数
        elif self.crop_type == "absolute":
            # 绝对裁剪：直接使用给定的裁剪尺寸，但不超过原图尺寸
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            # 绝对范围裁剪：在给定范围内随机生成裁剪尺寸
            assert self.crop_size[0] <= self.crop_size[1]  # 确保范围有效
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomCrop_CategoryAreaConstraint(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    类似于RandomCrop，但会寻找一个裁剪窗口，使得语义分割真值中没有单个类别占据超过single_category_max_area的比例，
    这可以避免训练不稳定。该函数最多尝试10次来寻找这样的有效裁剪窗口。
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        single_category_max_area: float = 1.0,
        ignored_category: int = None,
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
                与RandomCrop类中的参数相同
            single_category_max_area: the maximum allowed area ratio of a
                category. Set to 1.0 to disable
                单个类别允许的最大面积比例。设置为1.0表示禁用此限制
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
                允许在语义分割真值中超过面积比例的类别。通常设置为训练中被忽略的类别
        """
        self.crop_aug = RandomCrop(crop_type, crop_size)
        self._init(locals())

    def get_transform(self, image, sem_seg):
        if self.single_category_max_area >= 1.0:
            # 如果最大面积比例大于等于1.0，直接使用普通的随机裁剪
            return self.crop_aug.get_transform(image)
        else:
            h, w = sem_seg.shape  # 获取语义分割图的尺寸
            for _ in range(10):  # 最多尝试10次
                # 获取裁剪尺寸
                crop_size = self.crop_aug.get_crop_size((h, w))
                # 随机生成裁剪起点
                y0 = np.random.randint(h - crop_size[0] + 1)
                x0 = np.random.randint(w - crop_size[1] + 1)
                # 获取裁剪区域的语义分割
                sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                # 统计每个类别的像素数量
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    # 如果有忽略的类别，不计入统计
                    cnt = cnt[labels != self.ignored_category]
                # 检查是否满足面积比例约束：至少有两个类别，且最大类别的像素数不超过总像素数的指定比例
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.single_category_max_area:
                    break
            # 创建裁剪变换
            crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
            return crop_tfm


class RandomExtent(Augmentation):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    通过从源图像中裁剪随机的"子矩形"来输出图像。

    子矩形可以参数化为包含源图像外的像素，这种情况下这些像素将被设置为零（即黑色）。
    输出图像的大小将随着随机子矩形的大小而变化。
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
                输出图像的尺寸（高度，宽度）
            scale_range (l, h): Range of input-to-output size scaling factor
                输入到输出尺寸的缩放因子范围
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
                裁剪子矩形的偏移范围。矩形通过[w/2 * Uniform(-x, x), h/2 * Uniform(-y, y)]进行偏移，
                其中(w, h)是输入图像的（宽度，高度）。将每个分量设置为零可以在图像中心进行裁剪。
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img_h, img_w = image.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    随机变换图像对比度。

    对比度强度在(intensity_min, intensity_max)范围内均匀采样。
    - 强度 < 1 将降低对比度
    - 强度 = 1 将保持输入图像不变
    - 强度 > 1 将增加对比度
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
                最小增强强度
            intensity_max (float): Maximum augmentation
                最大增强强度
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=image.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    随机变换图像亮度。

    亮度强度在(intensity_min, intensity_max)范围内均匀采样。
    - 强度 < 1 将降低亮度
    - 强度 = 1 将保持输入图像不变
    - 强度 > 1 将增加亮度
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
                最小增强强度
            intensity_max (float): Maximum augmentation
                最大增强强度
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(Augmentation):
    """
    Randomly transforms saturation of an RGB image.
    Input images are assumed to have 'RGB' channel order.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    随机变换RGB图像的饱和度。
    输入图像假定具有'RGB'通道顺序。

    饱和度强度在(intensity_min, intensity_max)范围内均匀采样。
    - 强度 < 1 将降低饱和度（使图像更接近灰度图）
    - 强度 = 1 将保持输入图像不变
    - 强度 > 1 将增加饱和度
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        assert image.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = image.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(Augmentation):
    """
    The "lighting" augmentation described in AlexNet, using fixed PCA over ImageNet.
    Input images are assumed to have 'RGB' channel order.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    AlexNet中描述的"光照"增强，使用ImageNet上的固定PCA。
    输入图像假定具有'RGB'通道顺序。

    颜色抖动的程度通过正态分布随机采样，
    标准差由scale参数给定。
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
                主成分权重的标准差
        """
        super().__init__()
        self._init(locals())
        # 设置PCA的特征向量和特征值（在ImageNet上预计算得到）
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, image):
        # 确保输入是RGB图像
        assert image.shape[-1] == 3, "RandomLighting only works on RGB images"
        # 生成随机权重
        weights = np.random.normal(scale=self.scale, size=3)
        # 返回颜色变换
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )
