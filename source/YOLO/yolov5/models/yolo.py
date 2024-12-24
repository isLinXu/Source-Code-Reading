# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse  # 导入命令行参数解析模块
import contextlib  # 导入上下文管理模块
import math  # 导入数学模块
import os  # 导入操作系统模块
import platform  # 导入平台模块
import sys  # 导入系统模块
from copy import deepcopy  # 从复制模块导入深拷贝
from pathlib import Path  # 从路径模块导入Path类

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH 将根目录添加到系统路径中
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 如果不是Windows系统，获取相对路径

from models.common import (  # 从common模块导入多个类
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d  # 从experimental模块导入MixConv2d类
from utils.autoanchor import check_anchor_order  # 从autoanchor模块导入check_anchor_order函数
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args  # 从general模块导入多个函数
from utils.plots import feature_visualization  # 从plots模块导入feature_visualization函数
from utils.torch_utils import (  # 从torch_utils模块导入多个函数
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation  # 尝试导入thop库，用于计算FLOPs
except ImportError:
    thop = None  # 如果导入失败，则将thop设置为None


class Detect(nn.Module):  # YOLOv5检测头类，继承自nn.Module
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build  # 在构建过程中计算的步幅
    dynamic = False  # force grid reconstruction  # 强制网格重建
    export = False  # export mode  # 导出模式

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations.
        初始化YOLOv5检测层，指定类别、锚点、通道和就地操作。
        """
        super().__init__()  # 调用父类构造函数
        self.nc = nc  # number of classes  # 类别数量
        self.no = nc + 5  # number of outputs per anchor  # 每个锚点的输出数量
        self.nl = len(anchors)  # number of detection layers  # 检测层的数量
        self.na = len(anchors[0]) // 2  # number of anchors  # 锚点的数量
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid 初始化网格
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid 初始化锚点网格
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2) 注册锚点
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 输出卷积层
        self.inplace = inplace  # use inplace ops (e.g. slice assignment) 使用就地操作（例如切片赋值）

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`.
        通过YOLOv5层处理输入，改变形状以进行检测：`x(bs, 3, ny, nx, 85)`。
        """
        z = []  # inference output 推理输出
        for i in range(self.nl):  # 遍历每个检测层
            x[i] = self.m[i](x[i])  # conv 进行卷积操作
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85) 获取输入形状
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 改变形状并保持连续性

            if not self.training:  # inference 如果不是训练模式
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:  # 如果是动态或网格形状不匹配
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)  # 创建网格

                if isinstance(self, Segment):  # (boxes + masks) 如果是分割模型
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)  # 分割输出
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy坐标
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # 宽高
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)  # 合并输出
                else:  # Detect (boxes only) 检测（仅框）
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)  # 分割输出
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy坐标
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # 宽高
                    y = torch.cat((xy, wh, conf), 4)  # 合并输出
                z.append(y.view(bs, self.na * nx * ny, self.no))  # 添加到输出列表

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)  # 返回输出

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10.
        生成锚框的网格，兼容torch版本<1.10。
        """
        d = self.anchors[i].device  # 获取锚点设备
        t = self.anchors[i].dtype  # 获取锚点数据类型
        shape = 1, self.na, ny, nx, 2  # grid shape 网格形状
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)  # 创建y和x的范围
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7兼容性
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # 添加网格偏移
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)  # 计算锚点网格
        return grid, anchor_grid  # 返回网格和锚点网格


class Segment(Detect):  # YOLOv5分割头类，继承自Detect
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments.
        初始化YOLOv5分割头，指定掩码数量、原型和通道调整选项。
        """
        super().__init__(nc, anchors, ch, inplace)  # 调用父类构造函数
        self.nm = nm  # number of masks 掩码数量
        self.npr = npr  # number of protos 原型数量
        self.no = 5 + nc + self.nm  # number of outputs per anchor 每个锚点的输出数量
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 输出卷积层
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos 原型
        self.detect = Detect.forward  # 继承Detect的前向传播方法

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        通过网络处理输入，返回检测和原型；根据训练/导出模式调整输出。
        """
        p = self.proto(x[0])  # 计算原型
        x = self.detect(self, x)  # 调用Detect的前向传播方法
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])  # 返回输出



class BaseModel(nn.Module):  # YOLOv5基础模型类，继承自nn.Module
    """YOLOv5 base model. YOLOv5基础模型。"""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        执行YOLOv5基础模型的单尺度推理或训练过程，带有性能分析和可视化选项。
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train 返回单次前向传播的结果

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options.
        在YOLOv5模型上执行一次前向传播，启用性能分析和特征可视化选项。
        """
        y, dt = [], []  # outputs 输出列表和时间列表
        for m in self.model:  # 遍历模型中的每一层
            if m.f != -1:  # if not from previous layer 如果不是来自前一层
                # 根据层的索引获取输入，如果是整数则直接取y中的值，否则根据条件选择
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers 来自早期层的输出
            if profile:  # 如果启用性能分析
                self._profile_one_layer(m, x, dt)  # 分析当前层的性能
            x = m(x)  # run 运行当前层
            y.append(x if m.i in self.save else None)  # save output 保存输出
            if visualize:  # 如果启用可视化
                feature_visualization(x, m.type, m.i, save_dir=visualize)  # 可视化特征
        return x  # 返回最终输出

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters.
        通过计算GFLOPs、执行时间和参数来分析单个层的性能。
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix 判断是否为最后一层，如果是则复制输入以修复就地操作
        # FLOPs 计算当前层的FLOPs
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()  # 记录开始时间
        for _ in range(10):  # 运行10次以测量时间
            m(x.copy() if c else x)  # 运行当前层
        dt.append((time_sync() - t) * 100)  # 计算执行时间并添加到时间列表
        if m == self.model[0]:  # 如果是第一层
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")  # 打印表头
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")  # 打印当前层的执行时间、FLOPs和参数数量
        if c:  # 如果是最后一层
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")  # 打印总时间

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed.
        融合模型中的Conv2d()和BatchNorm2d()层以提高推理速度。
        """
        LOGGER.info("Fusing layers... ")  # 打印融合层的消息
        for m in self.model.modules():  # 遍历模型中的所有模块
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):  # 如果当前模块是卷积层并且有bn属性
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv 更新卷积层
                delattr(m, "bn")  # remove batchnorm 删除批归一化层
                m.forward = m.forward_fuse  # update forward 更新前向传播方法
        self.info()  # 打印模型信息
        return self  # 返回当前模型

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., [info(verbose=True, img_size=640)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov5/models/yolo.py:206:4-208:43).
        打印模型信息，给定详细程度和图像大小，例如[info(verbose=True, img_size=640)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov5/models/yolo.py:206:4-208:43)。
        """
        model_info(self, verbose, img_size)  # 调用model_info函数打印模型信息

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        对模型张量应用转换，如to()、cpu()、cuda()、half()，不包括参数或注册的缓冲区。
        """
        self = super()._apply(fn)  # 调用父类的_apply方法
        m = self.model[-1]  # Detect() 获取模型的最后一层
        if isinstance(m, (Detect, Segment)):  # 如果最后一层是Detect或Segment
            m.stride = fn(m.stride)  # 应用转换到步幅
            m.grid = list(map(fn, m.grid))  # 应用转换到网格
            if isinstance(m.anchor_grid, list):  # 如果锚点网格是列表
                m.anchor_grid = list(map(fn, m.anchor_grid))  # 应用转换到锚点网格
        return self  # 返回当前模型



class DetectionModel(BaseModel):  # YOLOv5检测模型类，继承自BaseModel
    # YOLOv5 detection model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors.
        使用配置文件、输入通道、类别数量和自定义锚点初始化YOLOv5模型。
        """
        super().__init__()  # 调用父类构造函数
        if isinstance(cfg, dict):  # 如果cfg是字典类型
            self.yaml = cfg  # model dict 模型字典
        else:  # 如果cfg是.yaml文件
            import yaml  # for torch hub 导入yaml模块

            self.yaml_file = Path(cfg).name  # 获取配置文件名
            with open(cfg, encoding="ascii", errors="ignore") as f:  # 打开配置文件
                self.yaml = yaml.safe_load(f)  # model dict 加载yaml文件为模型字典

        # Define model 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels 输入通道
        if nc and nc != self.yaml["nc"]:  # 如果提供了类别数量且与yaml中的类别数量不同
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")  # 打印覆盖信息
            self.yaml["nc"] = nc  # override yaml value 覆盖yaml中的类别数量
        if anchors:  # 如果提供了锚点
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")  # 打印覆盖信息
            self.yaml["anchors"] = round(anchors)  # override yaml value 覆盖yaml中的锚点
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist 解析模型并保存列表
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names 默认类别名称
        self.inplace = self.yaml.get("inplace", True)  # 获取是否使用就地操作的设置

        # Build strides, anchors 构建步幅和锚点
        m = self.model[-1]  # Detect() 获取模型的最后一层
        if isinstance(m, (Detect, Segment)):  # 如果最后一层是Detect或Segment
            s = 256  # 2x min stride 设置最小步幅
            m.inplace = self.inplace  # 设置是否使用就地操作
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)  # 定义前向传播函数
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward 计算步幅
            check_anchor_order(m)  # 检查锚点顺序
            m.anchors /= m.stride.view(-1, 1, 1)  # 根据步幅调整锚点
            self.stride = m.stride  # 保存步幅
            self._initialize_biases()  # 仅运行一次初始化偏置

        # Init weights, biases 初始化权重和偏置
        initialize_weights(self)  # 初始化权重
        self.info()  # 打印模型信息
        LOGGER.info("")  # 打印空行

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization.
        执行单尺度或增强推理，并可能包括性能分析或可视化。
        """
        if augment:  # 如果启用增强推理
            return self._forward_augment(x)  # augmented inference, None 返回增强推理的结果
        return self._forward_once(x, profile, visualize)  # single-scale inference, train 返回单次前向传播的结果

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections.
        在不同尺度和翻转下执行增强推理，返回组合检测结果。
        """
        img_size = x.shape[-2:]  # height, width 获取图像的高度和宽度
        s = [1, 0.83, 0.67]  # scales 设置不同的尺度
        f = [None, 3, None]  # flips (2-ud, 3-lr) 设置翻转类型（2为上下翻转，3为左右翻转）
        y = []  # outputs 输出列表
        for si, fi in zip(s, f):  # 遍历尺度和翻转
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  # 对图像进行缩放和翻转
            yi = self._forward_once(xi)[0]  # forward 执行前向传播
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save 保存处理后的图像
            yi = self._descale_pred(yi, fi, si, img_size)  # 反缩放预测结果
            y.append(yi)  # 添加到输出列表
        y = self._clip_augmented(y)  # clip augmented tails 裁剪增强推理的尾部
        return torch.cat(y, 1), None  # augmented inference, train 返回组合的增强推理结果

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size.
        反缩放增强推理的预测结果，调整翻转和图像大小。
        """
        if self.inplace:  # 如果使用就地操作
            p[..., :4] /= scale  # de-scale 反缩放
            if flips == 2:  # 如果是上下翻转
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud 反转y坐标
            elif flips == 3:  # 如果是左右翻转
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr 反转x坐标
        else:  # 如果不使用就地操作
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale 反缩放
            if flips == 2:  # 如果是上下翻转
                y = img_size[0] - y  # de-flip ud 反转y坐标
            elif flips == 3:  # 如果是左右翻转
                x = img_size[1] - x  # de-flip lr 反转x坐标
            p = torch.cat((x, y, wh, p[..., 4:]), -1)  # 合并反缩放后的结果
        return p  # 返回处理后的预测结果

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        裁剪YOLOv5模型的增强推理尾部，影响第一个和最后一个张量，基于网格点和层计数。
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5) 获取检测层的数量（P3-P5）
        g = sum(4**x for x in range(nl))  # grid points 计算网格点
        e = 1  # exclude layer count 排除层计数
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices 计算索引
        y[0] = y[0][:, :-i]  # large 裁剪大的输出
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices 计算索引
        y[-1] = y[-1][:, i:]  # small 裁剪小的输出
        return y  # 返回裁剪后的输出

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).
        初始化YOLOv5的Detect()模块的偏置，选用类别频率（cf）。
        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        有关详细信息，请参见 https://arxiv.org/abs/1708.02002 第3.3节。
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module 获取Detect模块
        for mi, s in zip(m.m, m.stride):  # from 遍历每个模块和步幅
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85) 将偏置从(255,)转换为(3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image) 计算目标偏置
            b.data[:, 5 : 5 + m.nc] += (  # 更新类别偏置
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)  # 将偏置设置为可训练的参数


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility 保留YOLOv5的'Model'类以兼容旧版本


class SegmentationModel(DetectionModel):  # YOLOv5分割模型类，继承自DetectionModel
    # YOLOv5 segmentation model
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list).
        使用可配置参数初始化YOLOv5分割模型：cfg（字符串）为配置文件，ch（整数）为通道数，nc（整数）为类别数量，anchors（列表）为锚点。
        """
        super().__init__(cfg, ch, nc, anchors)  # 调用父类构造函数

class ClassificationModel(BaseModel):  # YOLOv5分类模型类，继承自BaseModel
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels [ch](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov6/yolov6/utils/torch_utils.py:18:0-28:65), number of classes [nc](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov6/yolov6/utils/torch_utils.py:31:0-36:44), and `cuttoff` index.
        使用配置文件`cfg`、输入通道[ch](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov6/yolov6/utils/torch_utils.py:18:0-28:65)、类别数量[nc](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov6/yolov6/utils/torch_utils.py:31:0-36:44)和`cutoff`索引初始化YOLOv5模型。
        """
        super().__init__()  # 调用父类构造函数
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)  # 根据提供的模型或配置文件初始化模型

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification layer.
        从YOLOv5检测模型创建分类模型，在`cutoff`处切片并添加分类层。
        """
        if isinstance(model, DetectMultiBackend):  # 如果模型是DetectMultiBackend类型
            model = model.model  # unwrap DetectMultiBackend 解包DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone 设置模型的主干部分
        m = model.model[-1]  # last layer 获取最后一层
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module 获取输入通道
        c = Classify(ch, nc)  # Classify() 创建分类层
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type 设置索引、来源和类型
        model.model[-1] = c  # replace 替换最后一层
        self.model = model.model  # 设置模型
        self.stride = model.stride  # 设置步幅
        self.save = []  # 初始化保存列表
        self.nc = nc  # 设置类别数量

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file.
        从指定的*.yaml配置文件创建YOLOv5分类模型。
        """
        self.model = None  # 初始化模型为空

def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels [ch](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov6/yolov6/utils/torch_utils.py:18:0-28:65) and model architecture.
    从字典`d`解析YOLOv5模型，根据输入通道[ch](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolov6/yolov6/utils/torch_utils.py:18:0-28:65)和模型架构配置层。
    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")  # 打印表头
    anchors, nc, gd, gw, act, ch_mul = (  # 获取锚点、类别数量、深度倍增、宽度倍增、激活函数和通道倍增
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:  # 如果定义了激活函数
        Conv.default_act = eval(act)  # redefine default activation 重新定义默认激活函数，例如Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印激活函数信息
    if not ch_mul:  # 如果没有定义通道倍增
        ch_mul = 8  # 默认设置为8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors 获取锚点数量
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 计算输出数量

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out 初始化层、保存列表和输出通道
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args 遍历模型的主干和头部
        m = eval(m) if isinstance(m, str) else m  # eval strings 处理字符串类型的模块
        for j, a in enumerate(args):  # 遍历参数
            with contextlib.suppress(NameError):  # 忽略未定义错误
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings 处理字符串类型的参数

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain 计算深度增益
        if m in {  # 如果模块是以下类型
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]  # 获取输入通道和输出通道
            if c2 != no:  # if not output 如果不是输出
                c2 = make_divisible(c2 * gw, ch_mul)  # 调整输出通道

            args = [c1, c2, *args[1:]]  # 更新参数
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:  # 如果模块是以下类型
                args.insert(2, n)  # number of repeats 插入重复次数
                n = 1  # 重置重复次数为1
        elif m is nn.BatchNorm2d:  # 如果模块是BatchNorm2d
            args = [ch[f]]  # 更新参数为输入通道
        elif m is Concat:  # 如果模块是Concat
            c2 = sum(ch[x] for x in f)  # 计算输出通道为输入通道的和
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:  # 如果模块是Detect或Segment
            args.append([ch[x] for x in f])  # 添加输入通道
            if isinstance(args[1], int):  # number of anchors 如果第二个参数是整数（锚点数量）
                args[1] = [list(range(args[1] * 2))] * len(f)  # 创建锚点列表
            if m is Segment:  # 如果模块是Segment
                args[3] = make_divisible(args[3] * gw, ch_mul)  # 调整参数
        elif m is Contract:  # 如果模块是Contract
            c2 = ch[f] * args[0] ** 2  # 计算输出通道
        elif m is Expand:  # 如果模块是Expand
            c2 = ch[f] // args[0] ** 2  # 计算输出通道
        else:  # 其他类型
            c2 = ch[f]  # 直接获取输入通道

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module 创建模块
        t = str(m)[8:-2].replace("__main__.", "")  # module type 获取模块类型
        np = sum(x.numel() for x in m_.parameters())  # number params 计算参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params 附加索引、来源索引、类型和参数数量
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print 打印信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist 添加到保存列表
        layers.append(m_)  # 添加模块到层列表
        if i == 0:  # 如果是第一层
            ch = []  # 重置通道列表
        ch.append(c2)  # 添加输出通道到通道列表
    return nn.Sequential(*layers), sorted(save)  # 返回构建的模型和保存列表

if __name__ == "__main__":  # 如果是主模块
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")  # 添加配置文件参数
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")  # 添加批量大小参数
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # 添加设备参数
    parser.add_argument("--profile", action="store_true", help="profile model speed")  # 添加性能分析参数
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")  # 添加逐层性能分析参数
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")  # 添加测试参数
    opt = parser.parse_args()  # 解析参数
    opt.cfg = check_yaml(opt.cfg)  # check YAML 检查YAML文件
    print_args(vars(opt))  # 打印参数
    device = select_device(opt.device)  # 选择设备

    # Create model 创建模型
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)  # 创建随机输入图像
    model = Model(opt.cfg).to(device)  # 初始化模型并移动到指定设备

    # Options 选项
    if opt.line_profile:  # profile layer by layer 如果启用逐层性能分析
        model(im, profile=True)  # 执行逐层性能分析

    elif opt.profile:  # profile forward-backward 如果启用前向和反向性能分析
        results = profile(input=im, ops=[model], n=3)  # 执行性能分析

    elif opt.test:  # test all models 如果启用测试所有模型
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):  # 遍历所有yolo*.yaml文件
            try:
                _ = Model(cfg)  # 尝试初始化模型
            except Exception as e:  # 捕获异常
                print(f"Error in {cfg}: {e}")  # 打印错误信息

    else:  # report fused model summary 否则打印融合模型摘要
        model.fuse()  # 执行模型融合


if __name__ == "__main__":  # 如果是主模块
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")  # 添加配置文件参数，默认值为"yolov5s.yaml"
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")  # 添加批量大小参数，默认值为1
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # 添加设备参数，指定使用的CUDA设备
    parser.add_argument("--profile", action="store_true", help="profile model speed")  # 添加性能分析参数，用于分析模型速度
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")  # 添加逐层性能分析参数
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")  # 添加测试参数，用于测试所有yolo*.yaml文件
    opt = parser.parse_args()  # 解析命令行参数
    opt.cfg = check_yaml(opt.cfg)  # check YAML 检查YAML文件的有效性
    print_args(vars(opt))  # 打印解析后的参数
    device = select_device(opt.device)  # 选择设备（CPU或GPU）

    # Create model 创建模型
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)  # 创建随机输入图像，形状为(batch_size, 3, 640, 640)
    model = Model(opt.cfg).to(device)  # 初始化模型并将其移动到指定设备

    # Options 选项
    if opt.line_profile:  # profile layer by layer 如果启用逐层性能分析
        model(im, profile=True)  # 执行逐层性能分析

    elif opt.profile:  # profile forward-backward 如果启用前向和反向性能分析
        results = profile(input=im, ops=[model], n=3)  # 执行性能分析，运行3次

    elif opt.test:  # test all models 如果启用测试所有模型
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):  # 遍历所有yolo*.yaml文件
            try:
                _ = Model(cfg)  # 尝试初始化模型
            except Exception as e:  # 捕获异常
                print(f"Error in {cfg}: {e}")  # 打印错误信息

    else:  # report fused model summary 否则打印融合模型摘要
        model.fuse()  # 执行模型融合
