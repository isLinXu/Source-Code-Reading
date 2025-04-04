# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""

# 导入必要的库
import numpy as np  # 导入numpy用于数值计算
import torch  # 导入PyTorch深度学习框架
import torch.nn.functional as F  # 导入PyTorch的函数式接口
from fvcore.transforms.transform import (  # 从fvcore导入基础变换类
    CropTransform,  # 裁剪变换
    HFlipTransform,  # 水平翻转变换
    NoOpTransform,  # 无操作变换
    Transform,  # 基础变换类
    TransformList,  # 变换列表类
)
from PIL import Image  # 导入PIL库用于图像处理

try:
    import cv2  # noqa  # 尝试导入OpenCV库
except ImportError:
    # OpenCV is an optional dependency at the moment
    # OpenCV是一个可选的依赖项
    pass

# 定义模块的公共接口
__all__ = [
    "ExtentTransform",  # 区域提取变换
    "ResizeTransform",  # 尺寸调整变换
    "RotationTransform",  # 旋转变换
    "ColorTransform",  # 颜色变换
    "PILColorTransform",  # PIL颜色变换
]


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.
    从源图像中提取一个子区域并将其缩放到输出大小。

    The fill color is used to map pixels from the source rect that fall outside
    the source image.
    填充颜色用于映射源矩形中落在源图像外部的像素。

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    参考PIL文档中的ExtentTransform说明
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            源矩形的坐标，包含左上角(x0,y0)和右下角(x1,y1)的坐标
            output_size (h, w): dst image size
            目标图像的大小，以(高度,宽度)表示
            interp: PIL interpolation methods
            PIL图像插值方法
            fill: Fill color used when src_rect extends outside image
            当源矩形超出图像范围时使用的填充颜色
        """
        super().__init__()  # 调用父类的初始化方法
        self._set_attributes(locals())  # 设置实例属性

    def apply_image(self, img, interp=None):
        # 获取输出图像的高度和宽度
        h, w = self.output_size
        # 处理单通道图像
        if len(img.shape) > 2 and img.shape[2] == 1:
            # 如果是单通道图像，使用L模式（灰度图）
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            # 将numpy数组转换为PIL图像
            pil_image = Image.fromarray(img)
        # 应用区域变换
        pil_image = pil_image.transform(
            size=(w, h),  # 设置输出大小
            method=Image.EXTENT,  # 使用EXTENT方法进行变换
            data=self.src_rect,  # 设置源矩形区域
            resample=interp if interp else self.interp,  # 设置插值方法
            fill=self.fill,  # 设置填充颜色
        )
        # 将PIL图像转换回numpy数组
        ret = np.asarray(pil_image)
        # 如果输入是单通道图像，恢复其维度
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
        return ret

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        # 将图像中心从源坐标转换到输出坐标，然后将新原点映射到输出图像的角落
        h, w = self.output_size  # 获取输出图像的尺寸
        x0, y0, x1, y1 = self.src_rect  # 获取源矩形的坐标
        new_coords = coords.astype(np.float32)  # 将坐标转换为float32类型
        # 将坐标原点移动到源矩形的中心
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        # 根据源矩形和目标尺寸的比例进行缩放
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        # 将坐标原点移动到输出图像的中心
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        # 对分割图进行变换，使用最近邻插值以保持标签值
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    将图像调整到目标大小。
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            原始图像的高度和宽度
            new_h, new_w (int): new image size
            新图像的高度和宽度
            interp: PIL interpolation methods, defaults to bilinear.
            PIL插值方法，默认为双线性插值
        """
        # TODO decide on PIL vs opencv
        # TODO 决定使用PIL还是OpenCV
        super().__init__()  # 调用父类的初始化方法
        if interp is None:
            interp = Image.BILINEAR  # 默认使用双线性插值
        self._set_attributes(locals())  # 设置实例属性

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        # 对分割图进行变换，使用最近邻插值以保持标签值
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            原始图像的高度和宽度
            angle (float): degrees for rotation
            旋转的角度（以度为单位）
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            选择是否应该调整图像大小以适应整个旋转后的图像（默认值），或者简单地裁剪
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            旋转中心的坐标（宽度，高度），如果为None，则将使用图像中心作为旋转中心
            当expand=True时center无效，因为它只影响平移
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
            cv2插值方法，默认为cv2.INTER_LINEAR
        """
        super().__init__()  # 调用父类的初始化方法
        # 计算图像中心点
        image_center = np.array((w / 2, h / 2))
        # 如果未指定旋转中心，使用图像中心
        if center is None:
            center = image_center
        # 如果未指定插值方法，使用线性插值
        if interp is None:
            interp = cv2.INTER_LINEAR
        # 计算旋转角度的正弦和余弦值的绝对值
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            # 计算旋转后图像的新边界尺寸
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            # 不扩展时保持原始尺寸
            bound_w, bound_h = w, h

        self._set_attributes(locals())  # 设置实例属性
        # 创建用于坐标变换的旋转矩阵
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        # 由于OpenCV的问题#11784，需要为图像创建一个偏移的旋转矩阵
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        输入的img应该是一个numpy数组，格式为 高度 * 宽度 * 通道数
        """
        # 如果图像为空或旋转角度为360的整数倍，直接返回原图
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        # 检查图像尺寸是否符合预期
        assert img.shape[:2] == (self.h, self.w)
        # 确定使用的插值方法
        interp = interp if interp is not None else self.interp
        # 使用OpenCV的warpAffine函数进行仿射变换（旋转）
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        coords应该是一个N * 2的数组，包含N对(x, y)坐标点
        """
        # 将坐标转换为float类型的numpy数组
        coords = np.asarray(coords, dtype=float)
        # 如果坐标为空或旋转角度为360的整数倍，直接返回原坐标
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        # 使用OpenCV的transform函数对坐标进行变换
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])


class ColorTransform(Transform):
    """
    Generic wrapper for any photometric transforms.
    任何光度变换的通用包装器。
    These transformations should only affect the color space and
        not the coordinate space of the image (e.g. annotation
        coordinates such as bounding boxes should not be changed)
    这些变换应该只影响颜色空间而不影响图像的坐标空间
    （例如，边界框等标注坐标不应该被改变）
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in an ndarray and returns an ndarray.
            op (可调用对象): 要应用于图像的操作，
                接收一个ndarray并返回一个ndarray。
        """
        # 检查op参数是否可调用
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__()  # 调用父类的初始化方法
        self._set_attributes(locals())  # 设置实例属性

    def apply_image(self, img):
        # 对图像应用颜色变换操作
        return self.op(img)

    def apply_coords(self, coords):
        # 颜色变换不影响坐标，直接返回原坐标
        return coords

    def inverse(self):
        # 返回一个无操作变换作为逆变换
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        # 颜色变换不影响分割图，直接返回原分割图
        return segmentation


class PILColorTransform(ColorTransform):
    """
    Generic wrapper for PIL Photometric image transforms,
        which affect the color space and not the coordinate
        space of the image
    PIL光度图像变换的通用包装器，
        这些变换会影响图像的颜色空间
        但不会影响图像的坐标空间
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in a PIL Image and returns a transformed
                PIL Image.
                For reference on possible operations see:
                - https://pillow.readthedocs.io/en/stable/
            op (可调用对象): 要应用于图像的操作，
                接收一个PIL图像并返回一个转换后的PIL图像。
                有关可能的操作，请参考：
                - https://pillow.readthedocs.io/en/stable/
        """
        # 检查op参数是否可调用
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__(op)  # 调用父类的初始化方法

    def apply_image(self, img):
        # 将numpy数组转换为PIL图像
        img = Image.fromarray(img)
        # 调用父类的apply_image方法并将结果转回numpy数组
        return np.asarray(super().apply_image(img))


def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.
    对旋转框应用水平翻转变换。

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
        rotated_boxes (ndarray): Nx5浮点数数组，格式为
            (中心x坐标, 中心y坐标, 宽度, 高度, 角度(度)) 
            使用绝对坐标。
    """
    # Transform x_center
    # 变换中心点的x坐标（水平翻转）
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    # 变换角度（取负值）
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.
    对旋转框应用缩放变换。有关这些（近似）公式的推导细节，
    请参考:meth:`RotatedBoxes.scale`。

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
        rotated_boxes (ndarray): Nx5浮点数数组，格式为
            (中心x坐标, 中心y坐标, 宽度, 高度, 角度(度))
            使用绝对坐标。
    """
    # 计算x和y方向的缩放因子
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    # 缩放中心点坐标
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    # 将角度转换为弧度
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    # 计算角度的正弦和余弦值
    c = np.cos(theta)
    s = np.sin(theta)
    # 根据缩放因子和角度计算新的宽度
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    # 计算新的高度
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    # 计算新的角度并转换回度数
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


# 注册旋转框的水平翻转变换处理函数
HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
# 注册旋转框的缩放变换处理函数
ResizeTransform.register_type("rotated_box", Resize_rotated_box)

# not necessary any more with latest fvcore
# 使用最新的fvcore后不再需要
NoOpTransform.register_type("rotated_box", lambda t, x: x)
