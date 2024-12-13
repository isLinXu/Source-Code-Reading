#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# 导入必要的Python包和模块
import math  # 数学运算相关函数
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from yolov6.layers.common import *  # 导入通用层定义
from yolov6.utils.torch_utils import initialize_weights  # 导入权重初始化函数
from yolov6.models.reppan import *  # 导入RepPAN网络结构
from yolov6.models.efficientrep import *  # 导入EfficientRep主干网络
from yolov6.utils.events import LOGGER  # 导入日志记录器
from yolov6.models.heads.effidehead_lite import Detect, build_effidehead_layer  # 导入轻量级检测头

class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    
    YOLOv6轻量级模型，包含三个主要部分：
    1. 主干网络(backbone): 默认使用EfficientRep
    2. 特征融合颈部(neck): 默认使用Rep-PAN
    3. 检测头部(head): 默认使用轻量级解耦头
    '''
    
    export = False  # 用于标记是否处于模型导出模式

    def __init__(self, config, channels=3, num_classes=None):
        """初始化YOLOv6-lite模型
        
        Args:
            config: 模型配置对象
            channels: 输入图像的通道数，默认为3（RGB图像）
            num_classes: 目标类别数量
        """
        super().__init__()
        
        # 构建网络的三个主要组件：主干网络、颈部网络和检测头
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes)

        # 初始化检测头
        self.stride = self.detect.stride  # 获取检测头的步长
        self.detect.initialize_biases()   # 初始化检测头的偏置参数

        # 初始化整个网络的权重
        initialize_weights(self)

    def forward(self, x):
        """前向传播函数
        
        Args:
            x: 输入图像张量，形状为[batch_size, channels, height, width]
            
        Returns:
            如果是导出模式：
                返回检测头的输出
            否则：
                返回检测头输出和特征图列表的元组
        """
        # 检查是否处于模型导出模式
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        
        # 1. 通过主干网络提取特征
        x = self.backbone(x)
        
        # 2. 通过特征融合颈部处理特征
        x = self.neck(x)
        
        # 3. 在非导出模式下保存特征图
        if not export_mode:
            featmaps = []
            featmaps.extend(x)
            
        # 4. 通过检测头进行预测
        x = self.detect(x)
        
        # 5. 根据模式返回不同的结果
        return x if export_mode or self.export is True else [x, featmaps]

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

def build_network(config, in_channels, num_classes):
    """构建YOLOv6网络结构
    
    Args:
        config: 模型配置对象，包含网络结构的详细参数
        in_channels: 输入图像的通道数
        num_classes: 目标类别数量
    
    Returns:
        tuple: (backbone, neck, head) 网络的三个主要组件
    """
    width_mul = config.model.width_multiple  # 获取宽度缩放因子

    # 获取主干网络的配置参数
    num_repeat_backbone = config.model.backbone.num_repeats  # 主干网络重复次数
    out_channels_backbone = config.model.backbone.out_channels  # 主干网络输出通道数
    scale_size_backbone = config.model.backbone.scale_size  # 主干网络缩放比例
    in_channels_neck = config.model.neck.in_channels  # 颈部网络输入通道数
    unified_channels_neck = config.model.neck.unified_channels  # 颈部网络统一通道数
    in_channels_head = config.model.head.in_channels  # 检测头输入通道数
    num_layers = config.model.head.num_layers  # 检测头层数

    BACKBONE = eval(config.model.backbone.type)  # 获取主干网络类型
    NECK = eval(config.model.neck.type)  # 获取颈部网络类型

    # 根据宽度缩放因子调整主干网络的输出通道数
    out_channels_backbone = [make_divisible(i * width_mul) for i in out_channels_backbone]
    # 根据缩放比例计算主干网络中间通道数
    mid_channels_backbone = [make_divisible(int(i * scale_size_backbone), divisor=8) for i in out_channels_backbone]
    # 根据宽度缩放因子调整颈部网络的输入通道数
    in_channels_neck = [make_divisible(i * width_mul) for i in in_channels_neck]

    # 构建主干网络
    backbone = BACKBONE(in_channels,
                        mid_channels_backbone,
                        out_channels_backbone,
                        num_repeat=num_repeat_backbone)
    # 构建颈部网络
    neck = NECK(in_channels_neck, unified_channels_neck)
    # 构建检测头层
    head_layers = build_effidehead_layer(in_channels_head, 1, num_classes, num_layers)
    # 创建检测头
    head = Detect(num_classes, num_layers, head_layers=head_layers)

    return backbone, neck, head  # 返回构建的三个主要组件


def build_model(cfg, num_classes, device):
    """构建完整的YOLOv6模型
    
    Args:
        cfg: 模型配置对象
        num_classes: 目标类别数量
        device: 运行设备（CPU/GPU）
    
    Returns:
        Model: 构建好的YOLOv6模型实例
    """
    model = Model(cfg, channels=3, num_classes=num_classes).to(device)  # 创建模型实例并移动到指定设备
    return model  # 返回模型实例


def make_divisible(v, divisor=16):
    """将数值调整为指定除数的最接近的整数倍
    
    Args:
        v: 需要调整的数值
        divisor: 除数，默认为16
    
    Returns:
        调整后的数值，确保能被除数整除
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)  # 向上调整为最接近的整数倍
    if new_v < 0.9 * v:  # 如果调整后的值小于原值的90%
        new_v += divisor  # 则再加上一个除数
    return new_v  # 返回调整后的数值
