#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.

    YOLOv6模型的主体结构，包含三个主要部分：
    1. 主干网络(backbone): 默认使用EfficientRep
    2. 特征融合颈部(neck): 默认使用Rep-PAN
    3. 检测头部(head): 默认使用高效解耦头
    '''
    # 用于标记是否处于模型导出模式
    export = False

    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False, distill_ns=False):
        """初始化YOLOv6模型
        
        Args:
            config: 模型配置对象，包含网络结构的详细设置
            channels (int): 输入图像的通道数，默认为3（RGB图像）
            num_classes (int): 目标类别数量，如果为None则使用配置文件中的设置
            fuse_ab (bool): 是否融合注意力和backbone特征，默认False
            distill_ns (bool): 是否使用知识蒸馏的教师模型，默认False
        """
        super().__init__()
        
        # 构建网络的三个主要组件
        num_layers = config.model.head.num_layers  # 获取检测头的层数
        self.backbone, self.neck, self.detect = build_network(
            config, channels, num_classes, num_layers, 
            fuse_ab=fuse_ab, distill_ns=distill_ns
        )

        # 初始化检测头
        self.stride = self.detect.stride  # 获取检测头的步长
        self.detect.initialize_biases()   # 初始化检测头的偏置参数

        # 初始化整个网络的权重
        initialize_weights(self)

    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入图像张量，形状为[batch_size, channels, height, width]
            
        Returns:
            如果是导出模式：
                返回检测头的输出
            否则：
                返回检测头输出和特征图列表的元组
        """
        # 检查是否处于ONNX导出模式
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        
        # 1. 通过主干网络提取特征
        x = self.backbone(x)
        
        # 2. 通过特征融合颈部处理特征
        x = self.neck(x)
        
        # 3. 在训练模式下保存特征图
        if not export_mode:
            featmaps = []
            featmaps.extend(x)
            
        # 4. 通过检测头进行预测
        x = self.detect(x)
        
        # 5. 根据模式返回不同的结果
        return x if export_mode is True else [x, featmaps]

    def _apply(self, fn):
        """应用函数到模型的参数
        
        这是PyTorch的内部方法，用于将特定操作（如.cuda()或.float()）应用到模型参数。
        这里重写该方法是为了确保检测头的特殊参数（stride和grid）也能正确地被处理。
        
        Args:
            fn: 要应用的函数，例如将张量移动到GPU或改变数据类型的函数
            
        Returns:
            更新后的模型实例
        """
        # 首先调用父类的_apply方法处理基本参数
        self = super()._apply(fn)
        
        # 对检测头的特殊参数应用函数
        self.detect.stride = fn(self.detect.stride)  # 处理步长张量
        self.detect.grid = list(map(fn, self.detect.grid))  # 处理网格列表
        
        return self


def make_divisible(x, divisor):
    """将数值向上调整为除数的整数倍
    
    Args:
        x: 需要调整的数值
        divisor: 除数（通常为8，用于硬件优化）
    
    Returns:
        调整后的数值，确保能被除数整除
    """
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    """构建YOLOv6网络结构
    
    该函数负责构建完整的YOLOv6网络，包括主干网络(backbone)、特征融合颈部(neck)和检测头(head)
    
    Args:
        config: 模型配置对象，包含网络结构的详细参数
        channels: 输入图像的通道数
        num_classes: 目标类别数量
        num_layers: 检测头的层数
        fuse_ab: 是否使用特征融合注意力机制，默认False
        distill_ns: 是否使用知识蒸馏，默认False
        
    Returns:
        tuple: (backbone, neck, head) 网络的三个主要组件
    """
    # 获取深度和宽度的缩放系数
    depth_mul = config.model.depth_multiple  # 深度乘子，用于调整网络层数
    width_mul = config.model.width_multiple  # 宽度乘子，用于调整通道数

    # 获取backbone的配置参数
    num_repeat_backbone = config.model.backbone.num_repeats  # 主干网络重复次数
    channels_list_backbone = config.model.backbone.out_channels  # 主干网络输出通道数
    fuse_P2 = config.model.backbone.get('fuse_P2')  # 是否融合P2特征层
    cspsppf = config.model.backbone.get('cspsppf')  # 是否使用CSP-SPPF模块

    # 获取neck的配置参数
    num_repeat_neck = config.model.neck.num_repeats  # neck层重复次数
    channels_list_neck = config.model.neck.out_channels  # neck层输出通道数

    # 获取检测头的配置参数
    use_dfl = config.model.head.use_dfl  # 是否使用分布式焦点损失
    reg_max = config.model.head.reg_max  # 回归最大值

    # 计算实际的重复次数（考虑深度缩放）
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    # 计算实际的通道数（考虑宽度缩放，并确保是8的倍数）
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    # 获取基础构建块和网络类型
    block = get_block(config.training_mode)  # 获取基础构建块类型
    BACKBONE = eval(config.model.backbone.type)  # 获取主干网络类型
    NECK = eval(config.model.neck.type)  # 获取特征融合颈部类型

    # 根据backbone类型构建不同的网络结构
    if 'CSP' in config.model.backbone.type:  # 使用CSP结构的backbone
        # 获取stage块类型，默认为BepC3
        if "stage_block_type" in config.model.backbone:
            stage_block_type = config.model.backbone.stage_block_type
        else:
            stage_block_type = "BepC3"  # 默认使用BepC3块

        # 构建带CSP结构的backbone
        backbone = BACKBONE(
            in_channels=channels,  # 输入通道数
            channels_list=channels_list,  # 各层通道数列表
            num_repeats=num_repeat,  # 重复次数列表
            block=block,  # 基础构建块
            csp_e=config.model.backbone.csp_e,  # CSP扩展比例
            fuse_P2=fuse_P2,  # 是否融合P2特征
            cspsppf=cspsppf,  # 是否使用CSP-SPPF
            stage_block_type=stage_block_type  # stage块类型
        )

        # 构建带CSP结构的neck
        neck = NECK(
            channels_list=channels_list,  # 通道数列表
            num_repeats=num_repeat,  # 重复次数列表
            block=block,  # 基础构建块
            csp_e=config.model.neck.csp_e,  # CSP扩展比例
            stage_block_type=stage_block_type  # stage块类型
        )
    else:  # 使用普通结构的backbone
        # 构建普通backbone
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        # 构建普通neck
        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    # 根据不同模式构建检测头
    if distill_ns:  # 知识蒸馏模式
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:  # 特征融合模式
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init  # 获取锚框初始化参数
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:  # 标准模式
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head  # 返回构建的三个主要组件


def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False):
    """构建完整的YOLOv6模型
    
    Args:
        cfg: 模型配置对象
        num_classes: 目标类别数量
        device: 运行设备（CPU/GPU）
        fuse_ab: 是否使用特征融合，默认False
        distill_ns: 是否使用知识蒸馏，默认False
    
    Returns:
        Model: 构建好的YOLOv6模型实例
    """
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model
