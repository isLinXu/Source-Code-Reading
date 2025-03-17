import math
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import is_pillow_available, is_pyav_available, is_transformers_version_greater_than


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.45.0"):
    from transformers.models.mllama.processing_mllama import (
        convert_sparse_cross_attention_mask_to_dense,
        get_cross_attention_token_mask,
    )


if TYPE_CHECKING:
    from av.stream import Stream
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, ImageObject]
    VideoInput = str


def _get_paligemma_token_type_ids(
    imglens: Sequence[int],  # 每个样本的图像序列长度列表
    seqlens: Sequence[int],  # 每个样本的总序列长度列表
    processor: "ProcessorMixin"  # 包含图像处理配置的处理器
) -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.
    生成用于计算损失的Paligemma token类型ID

    Returns:
        batch_token_type_ids: shape (batch_size, sequence_length)
        返回形状为(batch_size, sequence_length)的token类型ID列表
    """
    batch_token_type_ids = []  # 初始化批次token类型ID容器
    for imglen, seqlen in zip(imglens, seqlens):  # 遍历每个样本的图像长度和总长度
        image_seqlen = imglen * getattr(processor, "image_seqlen")  # 计算图像部分总token数
        # 构建单个样本的token类型ID：
        # 图像部分用0填充，文本部分用1填充
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids  # 返回整个批次的token类型ID


# 基础插件类
class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token  # 图像占位符token
        self.video_token = video_token  # 视频占位符token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        验证模型是否支持当前输入模态
        """
        if len(images) != 0 and self.image_token is None:  # 有图像输入但未配置图像token
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:  # 有视频输入但未配置视频token
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        单张图像预处理流程
        """
        image_resolution: int = kwargs.get("image_resolution")  # 获取目标分辨率
        if (image.width * image.height) > image_resolution:  # 计算是否需要缩放
            # 计算缩放因子保持宽高比
            resize_factor = math.sqrt(image_resolution / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)  # 使用最近邻插值缩放

        if image.mode != "RGB":  # 确保图像为RGB模式
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        根据帧率计算视频采样帧数
        """
        video_fps: float = kwargs.get("video_fps")  # 目标帧率
        video_maxlen: int = kwargs.get("video_maxlen")  # 最大采样帧数
        total_frames = video_stream.frames  # 视频总帧数
        # 计算理论采样帧数（时长*帧率）
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        # 取理论值、最大限制和总帧数的最小值
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)  # 向下取整

    # 图像标准化处理
    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        标准化图像输入，包括读取和预处理，防止错误
        """
        results = []
        for image in images:
            # 处理不同输入类型的图像
            if isinstance(image, str):  # 文件路径
                image = Image.open(image)
            elif isinstance(image, bytes):  # 字节流
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):  # 包含路径或字节的字典
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))  # 执行图像预处理
        return results

    # 视频标准化处理
    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        标准化视频输入，包括读取、调整大小和转换格式
        """
        results = []
        for video in videos:
            container = av.open(video, "r")  # 打开视频容器
            video_stream = next(stream for stream in container.streams if stream.type == "video")  # 获取视频流
            total_frames = video_stream.frames  # 总帧数
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)  # 计算采样帧数
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)  # 生成采样索引
            
            frames: List["ImageObject"] = []
            container.seek(0)  # 重置到视频开头
            # 逐帧解码并采样
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())  # 转换为PIL图像

            frames = self._regularize_images(frames, **kwargs)  # 标准化图像帧
            results.append(frames)
        return results

    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],  # 原始消息列表
        images: Sequence["ImageInput"],      # 图像输入列表
        videos: Sequence["VideoInput"],      # 视频输入列表
        processor: Optional["ProcessorMixin"],  # 图像/视频处理器
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        在tokenization前预处理多模态输入消息
        """
        self._validate_input(images, videos)  # 验证输入模态支持
        return messages  # 返回原始消息（需子类实现具体处理）

    def process_token_ids(
        self,
        input_ids: List[int],                # token化后的输入ID列表
        labels: Optional[List[int]],         # 对应的标签ID列表
        images: Sequence["ImageInput"],      # 图像输入列表
        videos: Sequence["VideoInput"],      # 视频输入列表
        tokenizer: "PreTrainedTokenizer",    # tokenizer实例
        processor: Optional["ProcessorMixin"],  # 图像/视频处理器
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        在tokenization后处理token ID序列
        """
        self._validate_input(images, videos)  # 验证输入模态支持
        return input_ids, labels  # 返回原始token ID和标签（需子类实现具体处理）

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],      # 图像输入列表
        videos: Sequence["VideoInput"],      # 视频输入列表
        imglens: Sequence[int],              # 每个样本的图像数量
        vidlens: Sequence[int],              # 每个样本的视频数量
        batch_ids: Sequence[List[int]],      # 批次token ID列表
        processor: Optional["ProcessorMixin"],  # 图像/视频处理器
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.
        构建批次化的多模态模型输入
        """
        self._validate_input(images, videos)  # 验证输入模态支持
        return {}  # 返回空字典（需子类实现具体处理）


# LLaVA 插件实现
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen")  # 从处理器获取每个图像的token数
        messages = deepcopy(messages)  # 深拷贝避免修改原始数据
        
        # 遍历消息替换图像占位符
        for message in messages:
            content = message["content"]
            # 替换每个IMAGE_PLACEHOLDER为多个{{image}}占位符
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1  # 统计图像token数量
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)  # 替换为多个临时token
            # 将临时token替换为实际图像token
            message["content"] = content.replace("{{image}}", self.image_token)

        # 验证图像数量与占位符数量一致
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符数量({num_image_tokens})不匹配")
        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)  # 调用基类方法获取多模态输入


# LLaVA-Next 插件增强实现
class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)  # 获取预处理后的多模态输入
        
        # 动态计算图像特征序列长度
        if "image_sizes" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])  # 原始图像尺寸迭代器
        if "pixel_values" in mm_inputs:
            # 获取处理后的图像尺寸
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_size = next(image_sizes)  # 获取下一个原始图像尺寸
                orig_height, orig_width = image_size
                # 计算特征图序列长度
                image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                # 调整默认策略下的序列长度
                if getattr(processor, "vision_feature_select_strategy") == "default":
                    image_seqlen -= 1

                num_image_tokens += 1
                # 动态替换占位符
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
            # 替换为最终图像token
            message["content"] = content.replace("{{image}}", self.image_token)

        # 验证图像数量匹配
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符数量({num_image_tokens})不匹配")
        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


# LLaVA-Next 视频处理插件
class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        
        # 处理图像部分
        if "pixel_values" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])
            # 获取处理后的图像尺寸
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))
            for message in messages:
                content = message["content"]
                while IMAGE_PLACEHOLDER in content:
                    image_size = next(image_sizes)
                    orig_height, orig_width = image_size
                    # 计算特征序列长度
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if getattr(processor, "vision_feature_select_strategy") == "default":
                        image_seqlen -= 1  # 调整默认策略

                    num_image_tokens += 1
                    content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                message["content"] = content.replace("{{image}}", self.image_token)

        # 处理视频部分
        if "pixel_values_videos" in mm_inputs:
            pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
            height, width = get_image_size(pixel_values_video[0])
            num_frames = pixel_values_video.shape[0]  # 获取总帧数
            # 计算每帧特征数（基于patch大小）
            image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
            # 计算视频总特征数（平均池化后）
            video_seqlen = image_seqlen // 4 * num_frames
            
            for message in messages:
                content = message["content"]
                while VIDEO_PLACEHOLDER in content:
                    num_video_tokens += 1
                    # 替换视频占位符
                    content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)
                message["content"] = content.replace("{{video}}", self.video_token)

        # 验证输入数量匹配
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符({num_image_tokens})不匹配")
        if len(videos) != num_video_tokens:
            raise ValueError(f"视频数量({len(videos)})与占位符({num_video_tokens})不匹配")
        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        # 移除所有图像占位符
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "", 1)  # 直接移除占位符
            message["content"] = content

        # 验证图像数量匹配
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符({num_image_tokens})不匹配")
        return messages

    @override
    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(images, videos)
        num_images = len(images)
        # 计算图像token总长度
        image_seqlen = num_images * getattr(processor, "image_seqlen")
        # 获取图像token ID
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        # 在输入序列前添加图像token
        input_ids = [image_token_id] * image_seqlen + input_ids
        # 对应标签部分填充忽略索引
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seqlen + labels
        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        seqlens = [len(ids) for ids in batch_ids]  # 获取每个样本的序列长度
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        # 添加token类型ID（区分图像/文本）
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


# Pixtral 插件实现
class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],  # 原始消息列表
        images: Sequence["ImageInput"],      # 图像输入列表
        videos: Sequence["VideoInput"],      # 视频输入列表
        processor: Optional["ProcessorMixin"],  # 图像处理器
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)  # 验证输入合法性
        # 获取处理器配置参数
        patch_size = getattr(processor, "patch_size")  # 图像分块大小
        image_token = getattr(processor, "image_token")  # 图像token
        image_break_token = getattr(processor, "image_break_token")  # 分块分隔符
        image_end_token = getattr(processor, "image_end_token")  # 图像结束符

        num_image_tokens = 0  # 图像token计数器
        messages = deepcopy(messages)  # 深拷贝消息
        mm_inputs = self._get_mm_inputs(images, videos, processor)  # 获取预处理后的多模态输入
        image_input_sizes = mm_inputs.get("image_sizes", None)  # 获取原始图像尺寸列表

        # 遍历消息处理图像占位符
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if image_input_sizes is None:
                    raise ValueError("无法获取图像尺寸信息")

                # 获取当前图像的原始尺寸
                image_size = image_input_sizes[0][num_image_tokens]
                height, width = image_size
                # 计算高度和宽度方向的分块数量
                num_height_tokens = height // patch_size
                num_width_tokens = width // patch_size
                # 构建分块token结构（每行末尾添加分隔符）
                replace_tokens = [[image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                # 展平二维结构为一维列表
                replace_tokens = [item for sublist in replace_tokens for item in sublist]
                # 最后一个token替换为结束符
                replace_tokens[-1] = image_end_token
                # 将token列表转换为字符串
                replace_str = "".join(replace_tokens)
                # 替换占位符
                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)
                num_image_tokens += 1

            message["content"] = content

        # 验证图像数量匹配
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符({num_image_tokens})不匹配")
        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if mm_inputs.get("pixel_values"):
            mm_inputs["pixel_values"] = mm_inputs["pixel_values"][0]

        mm_inputs.pop("image_sizes", None)
        return mm_inputs


# Qwen2vl 插件实现
class Qwen2vlPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        # 执行基类预处理
        image = super()._preprocess_image(image, **kwargs)
        # 确保最小尺寸不低于28像素
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)
        # 处理极端长宽比图像（宽高比>200）
        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)
        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)
        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        # 获取基类计算的采样帧数
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        # 确保采样帧数为偶数
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)  # 输入验证
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2  # 合并块数
        mm_inputs = self._get_mm_inputs(images, videos, processor)  # 获取预处理输入
        image_grid_thw = mm_inputs.get("image_grid_thw", [])  # 图像网格尺寸
        video_grid_thw = mm_inputs.get("video_grid_thw", [])  # 视频网格尺寸

        num_image_tokens, num_video_tokens = 0, 0  # token计数器
        messages = deepcopy(messages)  # 深拷贝消息
        # 处理图像和视频占位符
        for message in messages:
            content = message["content"]
            # 处理图像占位符
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"图像数量不足，无法匹配{IMAGE_PLACEHOLDER}占位符数量")
                # 计算当前图像token数（总网格数/合并块数）
                token_count = image_grid_thw[num_image_tokens].prod() // merge_length
                # 替换为带特殊标记的token序列
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|vision_start|>{self.image_token * token_count}<|vision_end|>",
                    1,
                )
                num_image_tokens += 1
            # 处理视频占位符
            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"视频数量不足，无法匹配{VIDEO_PLACEHOLDER}占位符数量")
                # 计算当前视频token数
                token_count = video_grid_thw[num_video_tokens].prod() // merge_length
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"<|vision_start|>{self.video_token * token_count}<|vision_end|>",
                    1,
                )
                num_video_tokens += 1
            message["content"] = content

        # 验证输入数量匹配
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符({num_image_tokens})不匹配")
        if len(videos) != num_video_tokens:
            raise ValueError(f"视频数量({len(videos)})与占位符({num_video_tokens})不匹配")
        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


# VideoLlava 插件实现
class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)  # 输入验证
        num_image_tokens, num_video_tokens = 0, 0  # token计数器
        messages = deepcopy(messages)  # 深拷贝消息
        mm_inputs = self._get_mm_inputs(images, videos, processor)  # 获取预处理输入
        num_frames = 0  # 视频帧数
        has_images = "pixel_values_images" in mm_inputs  # 是否包含图像
        has_videos = "pixel_values_videos" in mm_inputs  # 是否包含视频

        if has_images or has_videos:
            # 获取特征图尺寸
            if has_images:
                height, width = get_image_size(to_numpy_array(mm_inputs.get("pixel_values_images")[0]))
                num_frames = 1  # 图像视为单帧
            if has_videos:
                pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(pixel_values_video[0])
                num_frames = pixel_values_video.shape[0]  # 获取视频总帧数

            # 计算特征序列长度
            image_seqlen = (height // processor.patch_size) * (width // processor.patch_size) + 1
            video_seqlen = image_seqlen * num_frames
            # 调整默认策略下的序列长度
            if getattr(processor, "vision_feature_select_strategy") == "default":
                image_seqlen -= 1

            # 替换消息中的占位符
            for message in messages:
                content = message["content"]
                # 替换图像占位符
                while IMAGE_PLACEHOLDER in content:
                    num_image_tokens += 1
                    content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                # 替换视频占位符
                while VIDEO_PLACEHOLDER in content:
                    num_video_tokens += 1
                    content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)
                # 替换临时token为实际token
                content = content.replace("{{image}}", self.image_token)
                message["content"] = content.replace("{{video}}", self.video_token)

        # 验证输入数量匹配
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符({num_image_tokens})不匹配")
        if len(videos) != num_video_tokens:
            raise ValueError(f"视频数量({len(videos)})与占位符({num_video_tokens})不匹配")
        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


# Mllama 插件实现
class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)  # 输入验证
        num_image_tokens = 0  # 图像token计数器
        messages = deepcopy(messages)  # 深拷贝消息
        # 统计并替换图像占位符
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)  # 统计占位符数量
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)  # 直接替换为图像token

        # 验证图像数量匹配
        if len(images) != num_image_tokens:
            raise ValueError(f"图像数量({len(images)})与占位符({num_image_tokens})不匹配")
        return messages

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs for mllama because its image processor only accepts List[List[ImageInput]].
        为MLLama处理视觉输入，其图像处理器需要二维列表输入
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        # 标准化图像输入
        images = self._regularize_images(images, image_resolution=getattr(processor, "image_resolution", 512 * 512))
        # 将图像列表转换为二维列表（每个样本一个列表）
        return image_processor([[image] for image in images], return_tensors="pt")  # 返回PyTorch张量

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        if len(images) != len(batch_ids):
            raise ValueError("Mllama only supports one image per sample.")

        mm_inputs = self._get_mm_inputs(images, videos, processor)
        num_tiles = mm_inputs.pop("num_tiles")
        image_token_id = getattr(processor, "image_token_id")
        max_image_tiles = getattr(processor.image_processor, "max_image_tiles")
        cross_attention_token_mask = [
            get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
        ]
        mm_inputs["cross_attention_mask"] = convert_sparse_cross_attention_mask_to_dense(
            cross_attention_token_mask,
            num_tiles=num_tiles,
            max_num_tiles=max_image_tiles,
            length=max(len(input_ids) for input_ids in batch_ids),
        )
        return mm_inputs


# 插件注册表
PLUGINS = {
    "llava": LlavaPlugin,
    "llava-next": LlavaNextPlugin,
    "llava-next-video": LlavaNextVideoPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2-vl": Qwen2vlPlugin,
    "video-llava": VideoLlavaPlugin,
    "mllama": MllamaPlugin,
}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return plugin_class(image_token, video_token)
