# 导入PyTorch基础模块
import torch
import torch.nn as nn

# 导入HuggingFace的CLIP相关组件
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
# 导入自定义的视频帧处理器
from ..processor.video_processor import VideoFramesProcessor

# CLIP视觉编码器主类
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        # 初始化加载状态标记
        self.is_loaded = False

        # 存储视觉模型名称
        self.vision_tower_name = vision_tower
        # 从参数获取特征选择层
        self.select_layer = args.mm_vision_select_layer
        # 获取特征选择方式（默认'patch'）
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # 是否优化视觉塔参数（默认False）
        self.is_optimize = getattr(args, 'optimize_vision_tower', False)
        
        # 立即加载模型的条件判断
        if not delay_load:
            self.load_model()
        # 即使延迟加载但需要解冻的情况
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # 仅加载配置用于延迟初始化
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    # 模型加载方法
    def load_model(self):
        # 初始化视频帧处理器（包含图像预处理）
        self.image_processor = VideoFramesProcessor.from_pretrained(self.vision_tower_name)
        # 加载CLIP视觉模型
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        # 冻结模型参数（默认不更新）
        self.vision_tower.requires_grad_(False)

        # 更新加载状态标记
        self.is_loaded = True

    # 特征选择方法
    def feature_select(self, image_forward_outs):
        # 从指定层获取隐藏状态
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # 根据选择模式处理特征
        if self.select_feature == 'patch':
            # 去除CLS token（只保留图像块特征）
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            # 保留CLS token和图像块特征
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # 图像前向处理方法
    def image_forward(self, images):
        # 处理图像列表的情况
        if type(images) is list:
            image_features = []
            for image in images:
                # 单张图像处理（添加批次维度）
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), 
                    output_hidden_states=True
                )
                # 特征选择并转换数据类型
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # 批量图像处理
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), 
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        return image_features

    # 主前向传播方法
    def forward(self, images):
        # 根据优化标志决定是否计算梯度
        if not self.is_optimize:
            with torch.no_grad():  # 禁用梯度计算
                image_features = self.image_forward(images)
        else:
            image_features = self.image_forward(images)

        return image_features

    # 虚拟特征属性（占位用）
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # 以下属性代理到vision_tower
    @property
    def dtype(self):
        return self.vision_tower.dtype  # 获取模型数据类型

    @property
    def device(self):
        return self.vision_tower.device  # 获取模型所在设备

    @property
    def config(self):
        # 根据加载状态返回配置
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size  # 获取隐藏层维度

    @property
    def num_patches(self):
        # 计算图像块数量：(图像尺寸/块尺寸)^2
        return (self.config.image_size // self.config.patch_size) ** 2