# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""

import argparse  # 导入argparse模块，用于解析命令行参数
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数
from copy import deepcopy  # 从copy模块导入deepcopy函数，用于深拷贝对象
from pathlib import Path  # 从pathlib模块导入Path类，用于处理文件路径

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # YOLOv5根目录
if str(ROOT) not in sys.path:  # 如果根目录不在系统路径中
    sys.path.append(str(ROOT))  # 将根目录添加到系统路径中
# ROOT = ROOT.relative_to(Path.cwd())  # relative（相对路径） - 注释掉的代码

import numpy as np  # 导入numpy库，通常用于数值计算
import tensorflow as tf  # 导入TensorFlow库
import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch导入nn模块，用于构建神经网络
from tensorflow import keras  # 从TensorFlow导入Keras模块，用于构建深度学习模型

from models.common import (  # 从common模块导入常用模型组件
    C3,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3x,
    Concat,
    Conv,
    CrossConv,
    DWConv,
    DWConvTranspose2d,
    Focus,
    autopad,
)
from models.experimental import MixConv2d, attempt_load  # 从experimental模块导入MixConv2d和attempt_load
from models.yolo import Detect, Segment  # 从yolo模块导入Detect和Segment类
from utils.activations import SiLU  # 从utils模块导入SiLU激活函数
from utils.general import LOGGER, make_divisible, print_args  # 从utils模块导入日志记录器、make_divisible函数和print_args函数


class TFBN(keras.layers.Layer):  # TensorFlow BatchNormalization包装类
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):  # 初始化函数，可选参数w用于预训练权重
        """Initializes a TensorFlow BatchNormalization layer with optional pretrained weights.
        初始化一个TensorFlow BatchNormalization层，支持可选的预训练权重。
        """
        super().__init__()  # 调用父类构造函数
        self.bn = keras.layers.BatchNormalization(  # 创建BatchNormalization层
            beta_initializer=keras.initializers.Constant(w.bias.numpy()),  # 初始化beta
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()),  # 初始化gamma
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),  # 初始化移动平均
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()),  # 初始化移动方差
            epsilon=w.eps,  # 设置epsilon
        )

    def call(self, inputs):  # 前向传播函数
        """Applies batch normalization to the inputs.
        对输入应用批量归一化。
        """
        return self.bn(inputs)  # 返回归一化后的结果


class TFPad(keras.layers.Layer):  # 填充输入的层类
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):  # 初始化函数，pad为填充大小
        """
        Initializes a padding layer for spatial dimensions 1 and 2 with specified padding, supporting both int and tuple
        inputs.
        为空间维度1和2初始化填充层，支持整数和元组输入。
        """
        super().__init__()  # 调用父类构造函数
        if isinstance(pad, int):  # 如果pad是整数
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])  # 创建填充常量
        else:  # 如果pad是元组/列表
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])  # 创建填充常量

    def call(self, inputs):  # 前向传播函数
        """Pads input tensor with zeros using specified padding, suitable for int and tuple pad dimensions.
        使用指定的填充对输入张量进行零填充，适用于整数和元组填充维度。
        """
        return tf.pad(inputs, self.pad, mode="constant", constant_values=0)  # 返回填充后的张量


class TFConv(keras.layers.Layer):  # 标准卷积层类
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):  # 初始化函数
        """
        Initializes a standard convolution layer with optional batch normalization and activation; supports only
        group=1.
        初始化一个标准卷积层，支持可选的批量归一化和激活，仅支持group=1。
        """
        super().__init__()  # 调用父类构造函数
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"  # 确保group为1
        # TensorFlow卷积填充与PyTorch不一致（例如k=3 s=2的'SAME'填充）
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        conv = keras.layers.Conv2D(  # 创建Conv2D层
            filters=c2,  # 输出通道数
            kernel_size=k,  # 卷积核大小
            strides=s,  # 步幅
            padding="SAME" if s == 1 else "VALID",  # 填充方式
            use_bias=not hasattr(w, "bn"),  # 是否使用偏置
            kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),  # 卷积核初始化
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()),  # 偏置初始化
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])  # 根据步幅选择卷积层
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity  # 如果有bn，则使用TFBN
        self.act = activations(w.act) if act else tf.identity  # 根据是否激活选择激活函数

    def call(self, inputs):  # 前向传播函数
        """Applies convolution, batch normalization, and activation function to input tensors.
        对输入张量应用卷积、批量归一化和激活函数。
        """
        return self.act(self.bn(self.conv(inputs)))  # 返回经过处理的结果


class TFDWConv(keras.layers.Layer):  # 深度卷积层类
    # Depthwise convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):  # 初始化函数
        """
        Initializes a depthwise convolution layer with optional batch normalization and activation for TensorFlow
        models.
        初始化一个深度卷积层，支持可选的批量归一化和激活，用于TensorFlow模型。
        """
        super().__init__()  # 调用父类构造函数
        assert c2 % c1 == 0, f"TFDWConv() output={c2} must be a multiple of input={c1} channels"  # 确保输出通道是输入通道的倍数
        conv = keras.layers.DepthwiseConv2D(  # 创建DepthwiseConv2D层
            kernel_size=k,  # 卷积核大小
            depth_multiplier=c2 // c1,  # 深度乘数
            strides=s,  # 步幅
            padding="SAME" if s == 1 else "VALID",  # 填充方式
            use_bias=not hasattr(w, "bn"),  # 是否使用偏置
            depthwise_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),  # 深度卷积核初始化
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()),  # 偏置初始化
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])  # 根据步幅选择卷积层
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity  # 如果有bn，则使用TFBN
        self.act = activations(w.act) if act else tf.identity  # 根据是否激活选择激活函数

    def call(self, inputs):  # 前向传播函数
        """Applies convolution, batch normalization, and activation function to input tensors.
        对输入张量应用卷积、批量归一化和激活函数。
        """
        return self.act(self.bn(self.conv(inputs)))  # 返回经过处理的结果


class TFDWConvTranspose2d(keras.layers.Layer):  # 深度反卷积层类
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):  # 初始化函数
        """
        Initializes depthwise ConvTranspose2D layer with specific channel, kernel, stride, and padding settings.
        初始化深度ConvTranspose2D层，设置特定的通道、卷积核、步幅和填充。
        """
        super().__init__()  # 调用父类构造函数
        assert c1 == c2, f"TFDWConv() output={c2} must be equal to input={c1} channels"  # 确保输入输出通道相等
        assert k == 4 and p1 == 1, "TFDWConv() only valid for k=4 and p1=1"  # 确保卷积核大小和填充符合要求
        weight, bias = w.weight.permute(2, 3, 1, 0).numpy(), w.bias.numpy()  # 获取权重和偏置
        self.c1 = c1  # 保存输入通道数
        self.conv = [  # 创建多个ConvTranspose层
            keras.layers.Conv2DTranspose(
                filters=1,  # 输出通道数为1
                kernel_size=k,  # 卷积核大小
                strides=s,  # 步幅
                padding="VALID",  # 填充方式
                output_padding=p2,  # 输出填充
                use_bias=True,  # 使用偏置
                kernel_initializer=keras.initializers.Constant(weight[..., i : i + 1]),  # 权重初始化
                bias_initializer=keras.initializers.Constant(bias[i]),  # 偏置初始化
            )
            for i in range(c1)  # 为每个输入通道创建一个卷积层
        ]

    def call(self, inputs):  # 前向传播函数
        """Processes input through parallel convolutions and concatenates results, trimming border pixels.
        通过并行卷积处理输入并连接结果，裁剪边缘像素。
        """
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]  # 返回处理后的结果

class TFFocus(keras.layers.Layer):  # Focus wh information into c-space
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):  # 初始化函数
        """
        Initializes TFFocus layer to focus width and height information into channel space with custom convolution
        parameters.
        初始化TFFocus层，将宽度和高度信息聚焦到通道空间，支持自定义卷积参数。

        Inputs are ch_in, ch_out, kernel, stride, padding, groups.
        输入参数包括输入通道数、输出通道数、卷积核、步幅、填充和分组。
        """
        super().__init__()  # 调用父类构造函数
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)  # 创建TFConv层，输入通道数为c1的4倍

    def call(self, inputs):  # 前向传播函数
        """
        Performs pixel shuffling and convolution on input tensor, downsampling by 2 and expanding channels by 4.
        对输入张量执行像素重排和卷积，进行2倍下采样并将通道扩展4倍。

        Example x(b,w,h,c) -> y(b,w/2,h/2,4c).
        示例：输入形状为x(b,w,h,c)，输出形状为y(b,w/2,h/2,4c)。
        """
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]  # 将输入张量进行重排
        return self.conv(tf.concat(inputs, 3))  # 连接重排后的张量并通过卷积层处理


class TFBottleneck(keras.layers.Layer):  # 标准瓶颈层
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # 初始化函数
        """
        Initializes a standard bottleneck layer for TensorFlow models, expanding and contracting channels with optional
        shortcut.
        初始化一个标准瓶颈层，用于TensorFlow模型，支持通道扩展和收缩以及可选的快捷连接。

        Arguments are ch_in, ch_out, shortcut, groups, expansion.
        输入参数包括输入通道数、输出通道数、快捷连接、分组和扩展比例。
        """
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)  # 创建第一个卷积层
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)  # 创建第二个卷积层
        self.add = shortcut and c1 == c2  # 判断是否使用快捷连接

    def call(self, inputs):  # 前向传播函数
        """Performs forward pass; if shortcut is True & input/output channels match, adds input to the convolution
        result.
        执行前向传播；如果快捷连接为True且输入输出通道匹配，则将输入加到卷积结果中。
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))  # 返回结果


class TFCrossConv(keras.layers.Layer):  # 交叉卷积层
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):  # 初始化函数
        """Initializes cross convolution layer with optional expansion, grouping, and shortcut addition capabilities.
        初始化交叉卷积层，支持可选的扩展、分组和快捷连接功能。
        """
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)  # 创建第一个卷积层
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)  # 创建第二个卷积层
        self.add = shortcut and c1 == c2  # 判断是否使用快捷连接

    def call(self, inputs):  # 前向传播函数
        """Passes input through two convolutions optionally adding the input if channel dimensions match.
        将输入通过两个卷积处理，如果通道维度匹配，则可选地添加输入。
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))  # 返回结果


class TFConv2d(keras.layers.Layer):  # 替代PyTorch的nn.Conv2D
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):  # 初始化函数
        """Initializes a TensorFlow 2D convolution layer, mimicking PyTorch's nn.Conv2D functionality for given filter
        sizes and stride.
        初始化一个TensorFlow 2D卷积层，模拟PyTorch的nn.Conv2D功能，支持给定的卷积核大小和步幅。
        """
        super().__init__()  # 调用父类构造函数
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"  # 确保group为1
        self.conv = keras.layers.Conv2D(  # 创建Conv2D层
            filters=c2,  # 输出通道数
            kernel_size=k,  # 卷积核大小
            strides=s,  # 步幅
            padding="VALID",  # 填充方式
            use_bias=bias,  # 是否使用偏置
            kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).numpy()),  # 卷积核初始化
            bias_initializer=keras.initializers.Constant(w.bias.numpy()) if bias else None,  # 偏置初始化
        )

    def call(self, inputs):  # 前向传播函数
        """Applies a convolution operation to the inputs and returns the result.
        对输入执行卷积操作并返回结果。
        """
        return self.conv(inputs)  # 返回卷积结果


class TFBottleneckCSP(keras.layers.Layer):  # CSP瓶颈层
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):  # 初始化函数
        """
        Initializes CSP bottleneck layer with specified channel sizes, count, shortcut option, groups, and expansion
        ratio.
        初始化CSP瓶颈层，支持指定的通道大小、数量、快捷连接选项、分组和扩展比例。

        Inputs are ch_in, ch_out, number, shortcut, groups, expansion.
        输入参数包括输入通道数、输出通道数、数量、快捷连接、分组和扩展比例。
        """
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)  # 创建第一个卷积层
        self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2)  # 创建第二个卷积层
        self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3)  # 创建第三个卷积层
        self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4)  # 创建第四个卷积层
        self.bn = TFBN(w.bn)  # 创建批量归一化层
        self.act = lambda x: keras.activations.swish(x)  # 定义激活函数为Swish
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])  # 创建瓶颈序列

    def call(self, inputs):  # 前向传播函数
        """Processes input through the model layers, concatenates, normalizes, activates, and reduces the output
        dimensions.
        通过模型层处理输入，连接、归一化、激活，并减少输出维度。
        """
        y1 = self.cv3(self.m(self.cv1(inputs)))  # 处理输入并得到y1
        y2 = self.cv2(inputs)  # 处理输入得到y2
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))  # 返回最终结果


class TFC3(keras.layers.Layer):  # CSP瓶颈层，包含3个卷积
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):  # 初始化函数
        """
        Initializes CSP Bottleneck with 3 convolutions, supporting optional shortcuts and group convolutions.
        初始化CSP瓶颈，包含3个卷积，支持可选的快捷连接和分组卷积。

        Inputs are ch_in, ch_out, number, shortcut, groups, expansion.
        输入参数包括输入通道数、输出通道数、数量、快捷连接、分组和扩展比例。
        """
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)  # 创建第一个卷积层
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)  # 创建第二个卷积层
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)  # 创建第三个卷积层
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])  # 创建瓶颈序列

    def call(self, inputs):  # 前向传播函数
        """
        Processes input through a sequence of transformations for object detection (YOLOv5).
        通过一系列变换处理输入，用于目标检测（YOLOv5）。

        See https://github.com/ultralytics/yolov5.
        参见 https://github.com/ultralytics/yolov5。
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))  # 返回最终结果

class TFC3x(keras.layers.Layer):  # 3 module with cross-convolutions
    # 3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):  # 初始化函数
        """
        Initializes layer with cross-convolutions for enhanced feature extraction in object detection models.
        初始化层，使用交叉卷积以增强目标检测模型中的特征提取。

        Inputs are ch_in, ch_out, number, shortcut, groups, expansion.
        输入参数包括输入通道数、输出通道数、数量、快捷连接、分组和扩展比例。
        """
        super().__init__()  # 调用父类构造函数
        c_ = int(c2 * e)  # hidden channels  # 计算隐藏通道数
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)  # 创建第一个卷积层
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)  # 创建第二个卷积层
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)  # 创建第三个卷积层
        self.m = keras.Sequential(  # 创建一个顺序模型
            [TFCrossConv(c_, c_, k=3, s=1, g=g, e=1.0, shortcut=shortcut, w=w.m[j]) for j in range(n)]  # 添加n个交叉卷积层
        )

    def call(self, inputs):  # 前向传播函数
        """Processes input through cascaded convolutions and merges features, returning the final tensor output.
        通过级联卷积处理输入并合并特征，返回最终的张量输出。
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))  # 连接卷积结果并返回


class TFSPP(keras.layers.Layer):  # 空间金字塔池化层
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):  # 初始化函数
        """Initializes a YOLOv3-SPP layer with specific input/output channels and kernel sizes for pooling.
        初始化YOLOv3-SPP层，支持特定的输入/输出通道和池化的卷积核大小。
        """
        super().__init__()  # 调用父类构造函数
        c_ = c1 // 2  # hidden channels  # 计算隐藏通道数
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)  # 创建第一个卷积层
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)  # 创建第二个卷积层，输出通道数为k的长度加1倍的隐藏通道数
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding="SAME") for x in k]  # 创建多个最大池化层

    def call(self, inputs):  # 前向传播函数
        """Processes input through two TFConv layers and concatenates with max-pooled outputs at intermediate stage.
        通过两个TFConv层处理输入，并在中间阶段与最大池化输出连接。
        """
        x = self.cv1(inputs)  # 通过第一个卷积层处理输入
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))  # 连接卷积输出和池化输出，并通过第二个卷积层处理


class TFSPPF(keras.layers.Layer):  # 空间金字塔池化-快速层
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):  # 初始化函数
        """Initializes a fast spatial pyramid pooling layer with customizable in/out channels, kernel size, and
        weights.
        初始化一个快速空间金字塔池化层，支持自定义的输入/输出通道、卷积核大小和权重。
        """
        super().__init__()  # 调用父类构造函数
        c_ = c1 // 2  # hidden channels  # 计算隐藏通道数
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)  # 创建第一个卷积层
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)  # 创建第二个卷积层
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME")  # 创建最大池化层

    def call(self, inputs):  # 前向传播函数
        """Executes the model's forward pass, concatenating input features with three max-pooled versions before final
        convolution.
        执行模型的前向传播，将输入特征与三个最大池化版本连接，然后进行最终卷积。
        """
        x = self.cv1(inputs)  # 通过第一个卷积层处理输入
        y1 = self.m(x)  # 对x进行最大池化
        y2 = self.m(y1)  # 对y1进行最大池化
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))  # 连接所有输出并通过第二个卷积层处理

class TFDetect(keras.layers.Layer):
    # TF YOLOv5 Detect layer
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        """Initializes YOLOv5 detection layer for TensorFlow with configurable classes, anchors, channels, and image
        size.
        初始化YOLOv5检测层，支持可配置的类别、锚框、通道和图像大小。
        """
        super().__init__()
        self.stride = tf.convert_to_tensor(w.stride.numpy(), dtype=tf.float32)
        self.nc = nc  # number of classes  # 类别数量
        self.no = nc + 5  # number of outputs per anchor  # 每个锚框的输出数量
        self.nl = len(anchors)  # number of detection layers  # 检测层数量
        self.na = len(anchors[0]) // 2  # number of anchors  # 锚框数量
        self.grid = [tf.zeros(1)] * self.nl  # init grid  # 初始化网格
        self.anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float32)
        self.anchor_grid = tf.reshape(self.anchors * tf.reshape(self.stride, [self.nl, 1, 1]), [self.nl, 1, -1, 1, 2])
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.training = False  # set to False after building model  # 构建模型后设置为False
        self.imgsz = imgsz
        for i in range(self.nl):
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def call(self, inputs):
        """Performs forward pass through the model layers to predict object bounding boxes and classifications.
        通过模型层执行前向传播，以预测物体的边界框和分类。
        """
        z = []  # inference output  # 推理输出
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no])

            if not self.training:  # inference  # 推理阶段
                y = x[i]
                grid = tf.transpose(self.grid[i], [0, 2, 1, 3]) - 0.5
                anchor_grid = tf.transpose(self.anchor_grid[i], [0, 2, 1, 3]) * 4
                xy = (tf.sigmoid(y[..., 0:2]) * 2 + grid) * self.stride[i]  # xy
                wh = tf.sigmoid(y[..., 2:4]) ** 2 * anchor_grid
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                y = tf.concat([xy, wh, tf.sigmoid(y[..., 4 : 5 + self.nc]), y[..., 5 + self.nc :]], -1)
                z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no]))

        return tf.transpose(x, [0, 2, 1, 3]) if self.training else (tf.concat(z, 1),)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """Generates a 2D grid of coordinates in (x, y) format with shape [1, 1, ny*nx, 2].
        生成一个二维坐标网格，格式为(x, y)，形状为[1, 1, ny*nx, 2]。
        """
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)



class TFSegment(TFDetect):
    # YOLOv5 Segment head for segmentation models
    # YOLOv5分割头，用于分割模型
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
        """Initializes YOLOv5 Segment head with specified channel depths, anchors, and input size for segmentation
        models.
        """
        # 初始化YOLOv5分割头，指定通道深度、锚框和输入大小
        super().__init__(nc, anchors, ch, imgsz, w)  # 调用父类初始化方法
        self.nm = nm  # number of masks
        # 掩膜数量
        self.npr = npr  # number of protos
        # 原型数量
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        # 每个锚框的输出数量
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]  # output conv
        # 输出卷积层
        self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto)  # protos
        # 原型层
        self.detect = TFDetect.call  # 检测调用

    def call(self, x):
        """Applies detection and proto layers on input, returning detections and optionally protos if training."""
        # 在输入上应用检测和原型层，返回检测结果，并在训练时可选返回原型
        p = self.proto(x[0])  # 通过原型层处理输入
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # （可选）全尺寸原型
        p = tf.transpose(p, [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
        # 将形状从(1,160,160,32)转换为(1,32,160,160)
        x = self.detect(self, x)  # 进行检测
        return (x, p) if self.training else (x[0], p)  # 如果在训练中返回(x, p)，否则返回(x[0], p)


class TFProto(keras.layers.Layer):
    def __init__(self, c1, c_=256, c2=32, w=None):
        """Initializes TFProto layer with convolutional and upsampling layers for feature extraction and
        transformation.
        """
        # 初始化TFProto层，包含卷积和上采样层，用于特征提取和转换
        super().__init__()  # 调用父类初始化方法
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)  # 第一个卷积层
        self.upsample = TFUpsample(None, scale_factor=2, mode="nearest")  # 上采样层
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)  # 第二个卷积层
        self.cv3 = TFConv(c_, c2, w=w.cv3)  # 第三个卷积层

    def call(self, inputs):
        """Performs forward pass through the model, applying convolutions and upscaling on input tensor."""
        # 在模型中执行前向传播，对输入张量应用卷积和上采样
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))  # 返回经过三个卷积层和上采样的结果


class TFUpsample(keras.layers.Layer):
    # TF version of torch.nn.Upsample()
    # TensorFlow版本的torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):
        """
        Initializes a TensorFlow upsampling layer with specified size, scale_factor, and mode, ensuring scale_factor is
        even.

        Warning: all arguments needed including 'w'
        """
        # 初始化一个TensorFlow上采样层，指定大小、缩放因子和模式，确保缩放因子为偶数
        super().__init__()  # 调用父类初始化方法
        assert scale_factor % 2 == 0, "scale_factor must be multiple of 2"
        # 确保缩放因子是2的倍数
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode)
        # 使用tf.image.resize进行上采样，调整图像大小

        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # # 使用Keras的UpSampling2D层进行上采样，插值模式为mode
        # with default arguments: align_corners=False, half_pixel_centers=False
        # # 默认参数：align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))
        # # 使用原始操作ResizeNearestNeighbor进行上采样，大小为原来的2倍

    def call(self, inputs):
        """Applies upsample operation to inputs using nearest neighbor interpolation."""
        # 使用最近邻插值对输入应用上采样操作
        return self.upsample(inputs)  # 返回上采样后的结果


class TFConcat(keras.layers.Layer):
    # TF version of torch.concat()
    # TensorFlow版本的torch.concat()
    def __init__(self, dimension=1, w=None):
        """Initializes a TensorFlow layer for NCHW to NHWC concatenation, requiring dimension=1."""
        # 初始化一个TensorFlow层，用于NCHW到NHWC的拼接，要求维度为1
        super().__init__()  # 调用父类初始化方法
        assert dimension == 1, "convert only NCHW to NHWC concat"
        # 确保只进行NCHW到NHWC的拼接
        self.d = 3  # 设置拼接的维度为3（最后一个维度）

    def call(self, inputs):
        """Concatenates a list of tensors along the last dimension, used for NCHW to NHWC conversion."""
        # 在最后一个维度上拼接张量列表，用于NCHW到NHWC的转换
        return tf.concat(inputs, self.d)  # 返回拼接后的结果

def parse_model(d, ch, model, imgsz):
    """Parses a model definition dict `d` to create YOLOv5 model layers, including dynamic channel adjustments."""
    # 解析模型定义字典`d`，创建YOLOv5模型层，包括动态通道调整
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 打印日志信息，显示模型的来源、数量、参数、模块和参数列表
    anchors, nc, gd, gw, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("channel_multiple"),
    )
    # 从字典中提取锚框、类别数量、深度和宽度倍数以及通道倍数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # 计算锚框的数量
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    # 计算输出数量 = 锚框数量 * (类别数量 + 5)
    if not ch_mul:
        ch_mul = 8
    # 如果没有指定通道倍数，则默认设置为8

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 初始化层列表、保存列表和输出通道
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        # 遍历骨干网络和头部的层定义
        m_str = m  # 保存模块名称
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 如果模块名称是字符串，则使用eval将其转换为实际模块
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                # 对参数进行eval处理，如果是字符串则转换
            except NameError:
                pass  # 如果出现NameError，则跳过

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 计算深度增益，如果n大于1，则根据深度倍数调整n
        if m in [
            nn.Conv2d,
            Conv,
            DWConv,
            DWConvTranspose2d,
            Bottleneck,
            SPP,
            SPPF,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3x,
        ]:
            # 如果模块是卷积或其他特定类型
            c1, c2 = ch[f], args[0]  # 获取输入通道和输出通道
            c2 = make_divisible(c2 * gw, ch_mul) if c2 != no else c2
            # 根据宽度倍数调整输出通道，确保可被通道倍数整除

            args = [c1, c2, *args[1:]]  # 更新参数列表
            if m in [BottleneckCSP, C3, C3x]:
                args.insert(2, n)  # 在参数列表中插入n
                n = 1  # 将n设置为1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # 如果模块是BatchNorm，则只需输入通道
        elif m is Concat:
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)  # 计算输出通道
        elif m in [Detect, Segment]:
            args.append([ch[x + 1] for x in f])  # 添加后续通道
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)  # 如果是锚框数量，则生成锚框列表
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)  # 对于分割模块，调整参数
            args.append(imgsz)  # 添加图像大小参数
        else:
            c2 = ch[f]  # 对于其他模块，直接获取输出通道

        tf_m = eval("TF" + m_str.replace("nn.", ""))  # 将模块名称转换为TF模块
        m_ = (
            keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)])
            if n > 1
            else tf_m(*args, w=model.model[i])
        )  # 创建模块

        torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 创建Torch模块
        t = str(m)[8:-2].replace("__main__.", "")  # 获取模块类型
        np = sum(x.numel() for x in torch_m_.parameters())  # 计算参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 附加索引、来源索引、类型和参数数量
        LOGGER.info(f"{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}")  # 打印信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 将索引添加到保存列表
        layers.append(m_)  # 将模块添加到层列表
        ch.append(c2)  # 更新通道列表
    return keras.Sequential(layers), sorted(save)  # 返回构建的Sequential模型和保存列表

class TFModel:
    # TF YOLOv5 model
    # TensorFlow YOLOv5模型
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, model=None, imgsz=(640, 640)):
        """Initializes TF YOLOv5 model with specified configuration, channels, classes, model instance, and input
        size.
        """
        # 使用指定的配置、通道、类别、模型实例和输入大小初始化TF YOLOv5模型
        super().__init__()  # 调用父类初始化方法
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
            # 如果cfg是字典，则将其作为模型字典
        else:  # is *.yaml
            import yaml  # for torch hub
            # 如果cfg是yaml文件，则导入yaml库

            self.yaml_file = Path(cfg).name  # 获取yaml文件名
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
                # 读取yaml文件并加载为模型字典

        # Define model
        # 定义模型
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            # 如果提供的类别数量与yaml中的不一致，记录日志信息
            self.yaml["nc"] = nc  # override yaml value
            # 用提供的类别数量覆盖yaml中的值
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)
        # 解析模型，生成模型层和保存列表

    def predict(
        self,
        inputs,
        tf_nms=False,
        agnostic_nms=False,
        topk_per_class=100,
        topk_all=100,
        iou_thres=0.45,
        conf_thres=0.25,
    ):
        # 进行预测
        y = []  # outputs
        # 初始化输出列表
        x = inputs  # 输入数据
        for m in self.model.layers:
            if m.f != -1:  # if not from previous layer
                # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # 根据层的来源获取输入数据

            x = m(x)  # run
            # 运行当前层
            y.append(x if m.i in self.savelist else None)  # save output
            # 如果当前层的索引在保存列表中，则保存输出

        # Add TensorFlow NMS
        # 添加TensorFlow非极大值抑制
        if tf_nms:
            boxes = self._xywh2xyxy(x[0][..., :4])  # 将xywh格式转换为xyxy格式
            probs = x[0][:, :, 4:5]  # 获取置信度
            classes = x[0][:, :, 5:]  # 获取类别
            scores = probs * classes  # 计算得分
            if agnostic_nms:
                nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
                # 如果使用无关类别的NMS，则调用AgnosticNMS
            else:
                boxes = tf.expand_dims(boxes, 2)  # 扩展维度
                nms = tf.image.combined_non_max_suppression(
                    boxes, scores, topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes=False
                )
                # 使用TensorFlow的combined_non_max_suppression进行NMS
            return (nms,)  # 返回NMS结果
        return x  # output [1,6300,85] = [xywh, conf, class0, class1, ...]
        # 返回输出，格式为[xywh, 置信度, 类别0, 类别1, ...]

    @staticmethod
    def _xywh2xyxy(xywh):
        """Converts bounding box format from [x, y, w, h] to [x1, y1, x2, y2], where xy1=top-left and xy2=bottom-
        right.
        """
        # 将边界框格式从[x, y, w, h]转换为[x1, y1, x2, y2]，其中xy1为左上角，xy2为右下角
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)  # 将xywh拆分为x, y, w, h
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)
        # 返回拼接后的结果，计算左上角和右下角坐标


class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    # TensorFlow无关类别的非极大值抑制
    def call(self, input, topk_all, iou_thres, conf_thres):
        """Performs agnostic NMS on input tensors using given thresholds and top-K selection."""
        # 使用给定的阈值和Top-K选择在输入张量上执行无关类别的非极大值抑制
        return tf.map_fn(
            lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
            input,
            fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
            name="agnostic_nms",
        )
        # 对输入的每个元素应用_nms函数，返回处理后的结果

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):
        """Performs agnostic non-maximum suppression (NMS) on detected objects, filtering based on IoU and confidence
        thresholds.
        """
        # 对检测到的对象执行无关类别的非极大值抑制，基于IoU和置信度阈值进行过滤
        boxes, classes, scores = x  # 解包输入数据
        class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)  # 获取类别索引
        scores_inp = tf.reduce_max(scores, -1)  # 获取每个框的最大得分
        selected_inds = tf.image.non_max_suppression(
            boxes, scores_inp, max_output_size=topk_all, iou_threshold=iou_thres, score_threshold=conf_thres
        )
        # 使用非极大值抑制选择框，返回选择的索引
        selected_boxes = tf.gather(boxes, selected_inds)  # 根据选择的索引获取框
        padded_boxes = tf.pad(
            selected_boxes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
            mode="CONSTANT",
            constant_values=0.0,
        )
        # 对选择的框进行填充，确保输出大小为topk_all
        selected_scores = tf.gather(scores_inp, selected_inds)  # 根据选择的索引获取得分
        padded_scores = tf.pad(
            selected_scores,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        # 对选择的得分进行填充，确保输出大小为topk_all
        selected_classes = tf.gather(class_inds, selected_inds)  # 根据选择的索引获取类别
        padded_classes = tf.pad(
            selected_classes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        # 对选择的类别进行填充，确保输出大小为topk_all
        valid_detections = tf.shape(selected_inds)[0]  # 获取有效检测的数量
        return padded_boxes, padded_scores, padded_classes, valid_detections
        # 返回填充后的框、得分、类别和有效检测数量


def activations(act=nn.SiLU):
    """Converts PyTorch activations to TensorFlow equivalents, supporting LeakyReLU, Hardswish, and SiLU/Swish."""
    # 将PyTorch激活函数转换为TensorFlow等效函数，支持LeakyReLU、Hardswish和SiLU/Swish
    if isinstance(act, nn.LeakyReLU):
        return lambda x: keras.activations.relu(x, alpha=0.1)  # LeakyReLU转换
    elif isinstance(act, nn.Hardswish):
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667  # Hardswish转换
    elif isinstance(act, (nn.SiLU, SiLU)):
        return lambda x: keras.activations.swish(x)  # SiLU/Swish转换
    else:
        raise Exception(f"no matching TensorFlow activation found for PyTorch activation {act}")
        # 如果没有匹配的激活函数，则抛出异常


def representative_dataset_gen(dataset, ncalib=100):
    """Generates a representative dataset for calibration by yielding transformed numpy arrays from the input
    dataset.
    """
    # 通过从输入数据集中生成转换后的numpy数组来生成用于校准的代表性数据集
    for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
        im = np.transpose(img, [1, 2, 0])  # 转换图像维度
        im = np.expand_dims(im, axis=0).astype(np.float32)  # 扩展维度并转换为float32
        im /= 255  # 归一化图像
        yield [im]  # 生成处理后的图像
        if n >= ncalib:
            break  # 如果达到校准数量，则停止生成


def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    # 权重路径
    imgsz=(640, 640),  # inference size h,w
    # 推理时的图像尺寸（高度，宽度）
    batch_size=1,  # batch size
    # 批处理大小
    dynamic=False,  # dynamic batch size
    # 是否使用动态批处理大小
):
    # PyTorch model
    # PyTorch模型
    im = torch.zeros((batch_size, 3, *imgsz))  # BCHW image
    # 创建一个全零的张量，形状为（批大小，通道数，图像高度，图像宽度）
    model = attempt_load(weights, device=torch.device("cpu"), inplace=True, fuse=False)
    # 加载PyTorch模型
    _ = model(im)  # inference
    # 对输入图像进行推理
    model.info()  # 打印模型信息

    # TensorFlow model
    # TensorFlow模型
    im = tf.zeros((batch_size, *imgsz, 3))  # BHWC image
    # 创建一个全零的张量，形状为（批大小，图像高度，图像宽度，通道数）
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    # 创建TensorFlow模型实例
    _ = tf_model.predict(im)  # inference
    # 对输入图像进行推理

    # Keras model
    # Keras模型
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    # 创建Keras输入层，形状为（图像高度，图像宽度，通道数）
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    # 创建Keras模型实例
    keras_model.summary()  # 打印Keras模型摘要

    LOGGER.info("PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.")
    # 记录信息，表明PyTorch、TensorFlow和Keras模型已成功验证，使用export.py导出TensorFlow模型


def parse_opt():
    """Parses and returns command-line options for model inference, including weights path, image size, batch size, and
    dynamic batching.
    """
    # 解析并返回模型推理的命令行选项，包括权重路径、图像大小、批处理大小和动态批处理
    parser = argparse.ArgumentParser()  # 创建解析器
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="weights path")
    # 添加权重路径参数
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    # 添加图像大小参数
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    # 添加批处理大小参数
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch size")
    # 添加动态批处理参数
    opt = parser.parse_args()  # 解析参数
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 如果只提供一个图像尺寸，则将其扩展为（高度，宽度）
    print_args(vars(opt))  # 打印参数
    return opt  # 返回解析后的选项


def main(opt):
    """Executes the YOLOv5 model run function with parsed command line options."""
    # 使用解析后的命令行选项执行YOLOv5模型运行函数
    run(**vars(opt))  # 解包选项并传递给run函数


if __name__ == "__main__":
    opt = parse_opt()  # 解析命令行选项
    main(opt)  # 执行主函数