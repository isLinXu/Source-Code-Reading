# 导入基础图像处理器
from transformers import CLIPImageProcessor
# 导入批处理特征和尺寸处理工具
from transformers.image_processing_utils import BatchFeature, get_size_dict
# 导入尺寸调整计算工具
from transformers.image_transforms import get_resize_output_image_size

# PyTorch相关模块
import torch
import torch.nn.functional as F

# 数值计算库
import numpy as np

class VideoFramesProcessor(CLIPImageProcessor):
    """视频帧处理类（继承自CLIP图像处理器）"""

    def __init__(self, **kwargs):
        """初始化方法，继承父类配置"""
        super().__init__(**kwargs)

    def preprocess(self, images, **kwargs):
        """视频帧预处理主方法
        参数：
            images: 输入视频帧（numpy数组或可迭代对象）
            **kwargs: 处理参数（覆盖类默认参数）
        返回：
            BatchFeature: 包含处理后的张量数据
        """
        # 处理非numpy数组输入（走父类流程）
        if not isinstance(images, np.ndarray):
            return super().preprocess(images=images, **kwargs)
        
        # 获取处理参数（优先使用传入参数，其次使用类默认参数）
        do_resize = kwargs.get('do_resize', self.do_resize)  # 是否调整尺寸
        size = kwargs.get('size', self.size)  # 目标尺寸配置
        size = get_size_dict(size, param_name="size", default_to_square=False)  # 转换为字典格式
        do_center_crop = kwargs.get('do_center_crop', self.do_center_crop)  # 是否中心裁剪
        crop_size = kwargs.get('crop_size', self.crop_size)  # 裁剪尺寸
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)  # 转换字典格式
        do_rescale = kwargs.get('do_rescale', self.do_rescale)  # 是否数值缩放
        rescale_factor = kwargs.get('rescale_factor', self.rescale_factor)  # 缩放系数
        do_normalize = kwargs.get('do_normalize', self.do_normalize)  # 是否标准化
        image_mean = kwargs.get('image_mean', self.image_mean)  # 均值参数
        image_std = kwargs.get('image_std', self.image_std)  # 标准差参数
        return_tensors = kwargs.get('return_tensors', None)  # 返回张量类型

        # 定义尺寸调整函数
        def resize(images, output_size):
            """双三次插值调整尺寸"""
            images = images.permute((0, 3, 1, 2))  # 调整维度顺序为 (B, C, H, W)
            images = F.interpolate(images, size=output_size, mode='bicubic')  # 双三次插值
            images = images.permute((0, 2, 3, 1))  # 恢复维度顺序为 (B, H, W, C)
            return images

        # 定义中心裁剪函数
        def center_crop(images, crop_size):
            """中心裁剪实现"""
            crop_width, crop_height = crop_size["width"], crop_size["height"]  # 获取裁剪尺寸
            img_width, img_height = images.shape[1:3]  # 获取原图尺寸
            x = (img_width - crop_width) // 2  # 计算水平起始点
            y = (img_height - crop_height) // 2  # 计算垂直起始点
            images = images[:, x:x+crop_width, y:y+crop_height]  # 执行裁剪
            return images
        
        # 定义数值缩放函数
        def rescale(images, rescale_factor):
            """像素值缩放（通常用于归一化到0-1）"""
            images = images * rescale_factor
            return images
        
        # 定义标准化函数
        def normalize(images, mean, std):
            """标准化处理（减均值除标准差）"""
            mean = torch.tensor(mean)  # 转换为张量
            std = torch.tensor(std)
            images = (images - mean) / std  # 标准化计算
            return images

        # 将numpy数组转换为浮点张量
        images = torch.from_numpy(images).float()

        # 执行尺寸调整
        if do_resize:
            # 计算调整后的目标尺寸
            output_size = get_resize_output_image_size(
                images[0], 
                size=size["shortest_edge"], 
                default_to_square=False
            )
            images = resize(images, output_size)
        
        # 执行中心裁剪
        if do_center_crop:
            images = center_crop(images, crop_size)
        
        # 执行数值缩放
        if do_rescale:
            images = rescale(images, rescale_factor)
        
        # 执行标准化
        if do_normalize:
            images = normalize(images, image_mean, image_std)

        # 调整维度顺序为 (B, C, H, W)
        images = images.permute((0, 3, 1, 2))
        # 构建返回数据
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
