import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import logging
import deepspeed
from pathlib import Path
from open_clip.factory import load_state_dict, get_model_config
from open_clip.model import CLIPVisionCfg, CLIPTextCfg, _build_vision_tower, convert_to_custom_text_state_dict, resize_pos_embed
from typing import Dict, Optional
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled


class OpenCLIPVisionTower(nn.Module):
    """OpenCLIP视觉编码器主类"""
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        # 初始化加载状态标记
        self.is_loaded = False
        # 视觉模型路径
        self.vision_tower_name = vision_tower
        # 加载配置文件
        self.vision_config = json.load(open(os.path.join(vision_tower,'open_clip_config.json'), 'r'))
        # 优化标志（从参数获取，默认False）
        self.is_optimize = getattr(args, 'optimize_vision_tower_aux', False)
        # 是否使用drop path（从参数获取，默认True）
        self.is_droppath = getattr(args, 'drop_path', True)

        # 立即加载模型的条件判断
        if not delay_load:
            self.load_model()

    def load_model(self):
        """模型加载方法"""
        # 模型权重路径
        ckpt_path = os.path.join(self.vision_tower_name, 'open_clip_pytorch_model.bin')
        # 根据模型名称识别类型
        if 'convnext' in self.vision_tower_name:
            if 'large' in self.vision_tower_name and 'd-320' in self.vision_tower_name:
                self.model_type = 'convnext_large_d_320'
                self.model_channel = [192, 384, 768, 1536]  # 各阶段通道数
            elif 'base' in self.vision_tower_name and 'w-320' in self.vision_tower_name:
                self.model_type = 'convnext_base_w_320'
                self.model_channel = [128, 256, 512, 1024]
            elif 'xxlarge' in self.vision_tower_name:
                self.model_type = 'convnext_xxlarge'
                self.model_channel = [384, 768, 1536, 3072]

        # 初始化CLIP模型
        clip_model = CLIP(**get_model_config(self.model_type), drop_path=self.is_droppath)
        # 调整视觉主干结构
        clip_model.visual.trunk.norm_pre = None  # 移除预归一化层
        clip_model.visual.trunk.head = None      # 移除头部层
        clip_model.visual.head = None            # 移除最终头部
        print(f'Loading pretrained weights ({self.model_type}).')
        # 加载预训练权重
        load_checkpoint(clip_model, ckpt_path, strict=False)

        self.is_loaded = True  # 更新加载状态
        # 分解视觉主干为stem和stages
        self.vision_stem = clip_model.visual.trunk.stem    # 输入处理模块
        self.vision_stages = clip_model.visual.trunk.stages  # 多阶段处理模块
        # 冻结参数
        self.vision_stem.requires_grad_(False)
        self.vision_stages.requires_grad_(False)
    
    def forward(self, images):
        """主前向传播方法"""
        if type(images) is list:
            # 处理图像列表（逐帧处理）
            image_features = []
            for image in images:
                image_feature = self.backbone(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_feature)
        else:
            # 批量处理
            image_features = self.backbone(images.to(device=self.device, dtype=self.dtype))
        return image_features

    def backbone(self, images):
        """特征提取主干网络"""
        # 根据优化标志决定是否计算梯度
        if not self.is_optimize:
            with torch.no_grad():
                results = self.basic_forward(images)
        else:
            results = self.basic_forward(images)

        # 多尺度特征融合
        target_size = (results['stage_0'].shape[-2], results['stage_0'].shape[-1])
        result_cat = []
        for _stage in results:
            if _stage == 'stage_0':
                result_cat.append(results[_stage].contiguous())
            else:
                # 双线性插值对齐特征尺寸
                result_cat.append(F.interpolate(
                    results[_stage].float().contiguous(), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).to(dtype=results[_stage].dtype))
        # 通道维度拼接
        result_cat = torch.cat(result_cat, dim=1)
        return result_cat.contiguous()

    def basic_forward(self, images):
        """基础前向传播流程"""
        results = {}    
        x = self.vision_stem(images)  # 初始特征提取
        # 逐阶段处理
        for _idx in range(len(self.vision_stages)):
            x = self.vision_stages[_idx](x)
            results[f'stage_{_idx}'] = x  # 记录各阶段输出
        return results

    @property
    def dummy_feature(self):
        """虚拟特征（占位用）"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """模型数据类型"""
        return self.vision_stem[0].weight.dtype  # 从第一个层的权重获取

    @property
    def device(self):
        """模型所在设备"""
        return self.vision_stem[0].weight.device  # 从第一个层的权重获取

    @property
    def config(self):
        """模型配置"""
        return self.vision_config

    @property
    def hidden_size(self):
        """隐藏层维度（各阶段通道数之和）"""
        return sum(self.model_channel)

# 修改自open_clip的权重加载函数（支持Zero3阶段）
def load_checkpoint(model, checkpoint_path, strict=True):
    """加载模型检查点"""
    # 处理numpy格式权重
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        from open_clip.big_vision import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}

    # 加载状态字典
    state_dict = load_state_dict(checkpoint_path)
    # 转换旧格式状态字典
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # 处理位置ID（兼容新版本transformers）
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    # 调整位置编码尺寸
    resize_pos_embed(state_dict, model)
    
    # Zero3分布式训练处理
    if is_deepspeed_zero3_enabled():
        error_msgs = []
        # 自定义加载函数
        def load(module: nn.Module, state_dict, prefix=""):
            # 处理当前模块的参数
            local_metadata = {}
            args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
            # 仅处理当前前缀的参数
            if len([key for key in state_dict if key.startswith(prefix)]) > 0:
                if is_deepspeed_zero3_enabled():
                    # 收集分布在不同GPU上的参数
                    named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                    params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                    if len(params_to_gather) > 0:
                        # 使用deepspeed的上下文管理器加载参数
                        with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                module._load_from_state_dict(*args)
            # 递归处理子模块
            for name, child in module._modules.items():
                if child is not None:
                    load(child, state_dict, prefix + name + ".")
        
        load(model, state_dict)
        incompatible_keys = []
    else:
        # 普通加载方式
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        logging.info(f"不兼容的键: {incompatible_keys.missing_keys}")
    return incompatible_keys

class CLIP(nn.Module):
    """CLIP模型主类"""
    output_dict: torch.jit.Final[bool]  # 输出格式标记

    def __init__(
            self,
            embed_dim: int,          # 嵌入维度
            vision_cfg: CLIPVisionCfg,  # 视觉配置
            text_cfg: CLIPTextCfg,   # 文本配置
            quick_gelu: bool = False,  # 快速GELU激活
            cast_dtype: Optional[torch.dtype] = None,  # 转换数据类型
            output_dict: bool = False,  # 是否输出字典格式
            drop_path: bool = False,  # 是否使用drop path
    ):
        super().__init__()
        self.output_dict = output_dict

        # 配置drop path参数
        if not drop_path:
            print('训练时不使用drop path.')
            vision_cfg['timm_drop_path'] = 0.0  # 禁用drop path

        # 构建视觉模块
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
