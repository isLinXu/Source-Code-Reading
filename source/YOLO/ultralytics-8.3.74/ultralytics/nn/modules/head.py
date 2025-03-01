# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy  # 导入copy模块
import math  # 导入数学库

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.nn.init import constant_, xavier_uniform_  # 导入初始化函数

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors  # 导入工具函数

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto  # 导入块模块中的类
from .conv import Conv, DWConv  # 导入卷积模块中的类
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer  # 导入变换器模块中的类
from .utils import bias_init_with_prob, linear_init  # 导入工具函数

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"
# 定义模块的公共接口

class Detect(nn.Module):
    """YOLO Detect head for detection models."""
    # YOLO检测头，用于检测模型

    dynamic = False  # force grid reconstruction
    # 动态标志，强制网格重建
    export = False  # export mode
    # 导出模式
    format = None  # export format
    # 导出格式
    end2end = False  # end2end
    # 是否为端到端
    max_det = 300  # max_det
    # 最大检测数量
    shape = None  # 初始化
    anchors = torch.empty(0)  # init
    # 初始化锚框
    strides = torch.empty(0)  # init
    # 初始化步幅
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    # 向后兼容性标志

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        # 初始化YOLO检测层，给定类别数和通道
        super().__init__()  # 调用父类构造函数
        self.nc = nc  # number of classes
        # 类别数
        self.nl = len(ch)  # number of detection layers
        # 检测层数
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        # DFL通道数
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        # 每个锚框的输出数量
        self.stride = torch.zeros(self.nl)  # strides computed during build
        # 在构建过程中计算的步幅
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # 计算通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )  # 定义第二个卷积层
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )  # 定义第三个卷积层
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  # 定义DFL层

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)  # 深拷贝cv2
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # 深拷贝cv3

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        # 拼接并返回预测的边界框和类别概率
        if self.end2end:
            return self.forward_end2end(x)  # 如果是端到端，调用相应方法

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 拼接卷积层的输出
        if self.training:  # Training path
            return x  # 如果是训练模式，返回x
        y = self._inference(x)  # 推理
        return y if self.export else (y, x)  # 根据导出模式返回结果

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        # 执行v10Detect模块的前向传播
        x_detach = [xi.detach() for xi in x]  # 分离输入
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]  # 计算一对一的输出
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 拼接卷积层的输出
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}  # 返回训练模式下的输出

        y = self._inference(one2one)  # 推理
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)  # 后处理
        return y if self.export else (y, {"one2many": x, "one2one": one2one})  # 根据导出模式返回结果

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # 根据多层特征图解码预测的边界框和类别概率
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 拼接特征
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # 生成锚框
            self.shape = shape  # 更新形状

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]  # 获取边界框
            cls = x_cat[:, self.reg_max * 4 :]  # 获取类别
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 分割边界框和类别

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # 预计算归一化因子以增加数值稳定性
            grid_h = shape[2]  # 获取高度
            grid_w = shape[3]  # 获取宽度
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)  # 创建网格大小
            norm = self.strides / (self.stride[0] * grid_size)  # 计算归一化因子
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])  # 解码边界框
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )  # 解码边界框
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)  # 返回结果
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # 解码边界框

        return torch.cat((dbox, cls.sigmoid()), 1)  # 返回拼接后的结果

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        # 初始化Detect()的偏置，警告：需要步幅可用
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        # 解码边界框
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)  # 返回解码后的边界框

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        # 后处理YOLO模型预测
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)  # 分割边界框和分数
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)  # 获取最大分数的索引
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))  # 根据索引收集边界框
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))  # 根据索引收集分数
        scores, index = scores.flatten(1).topk(min(max_det, anchors))  # 获取前k个分数和索引
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)  # 返回拼接后的结果


class Segment(Detect):
    """YOLO Segment head for segmentation models."""
    # YOLO分割头，用于分割模型

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        # 初始化YOLO模型属性，如掩码数量、原型和卷积层
        super().__init__(nc, ch)  # 调用父类构造函数
        self.nm = nm  # number of masks
        # 掩码数量
        self.npr = npr  # number of protos
        # 原型数量
        self.proto = Proto(ch[0], self.npr, self.nm)  # 定义原型

        c4 = max(ch[0] // 4, self.nm)  # 计算通道数
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)  # 定义卷积层

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        # 如果是训练模式，返回模型输出和掩码系数，否则返回输出和掩码系数
        p = self.proto(x[0])  # 计算掩码原型
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        # 计算掩码系数
        x = Detect.forward(self, x)  # 调用父类的forward方法
        if self.training:
            return x, mc, p  # 如果是训练模式，返回x、mc和p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))  # 根据导出模式返回结果


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""
    # YOLO OBB检测头，用于带旋转的检测模型

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes [nc](cci:2://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/nn/modules/conv.py:371:0-384:63) and layer channels [ch](cci:2://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/nn/modules/block.py:1215:0-1261:32)."""
        # 初始化OBB，给定类别数和层通道
        super().__init__(nc, ch)  # 调用父类构造函数
        self.ne = ne  # number of extra parameters
        # 额外参数数量

        c4 = max(ch[0] // 4, self.ne)  # 计算通道数
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)  # 定义卷积层

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        # 拼接并返回预测的边界框和类别概率
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # 将角度转换为[-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]  # 注释掉的代码
        if not self.training:
            self.angle = angle  # 保存角度
        x = Detect.forward(self, x)  # 调用父类的forward方法
        if self.training:
            return x, angle  # 如果是训练模式，返回x和角度
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))  # 根据导出模式返回结果

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        # 解码旋转的边界框
        return dist2rbox(bboxes, self.angle, anchors, dim=1)  # 返回解码后的边界框


class Pose(Detect):
    """YOLO Pose head for keypoints models."""
    # YOLO姿态检测头，用于关键点模型

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        # 使用默认参数和卷积层初始化YOLO网络
        super().__init__(nc, ch)  # 调用父类的初始化方法
        self.kpt_shape = kpt_shape  # 关键点的形状，表示关键点数量和维度（2表示x,y，3表示x,y,visible）
        self.nk = kpt_shape[0] * kpt_shape[1]  # 关键点总数

        c4 = max(ch[0] // 4, self.nk)  # 计算卷积层的输出通道数
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        # 创建一个模块列表，包含多个卷积层，处理输入通道数为ch的每个元素

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        # 通过YOLO模型执行前向传播并返回预测结果
        bs = x[0].shape[0]  # 获取批次大小
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        # 将每个输入经过卷积层处理后，按最后一个维度拼接，得到关键点的预测
        x = Detect.forward(self, x)  # 调用父类的前向传播方法
        if self.training:
            return x, kpt  # 如果是训练模式，返回原始输出和关键点
        pred_kpt = self.kpts_decode(bs, kpt)  # 解码关键点
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))
        # 根据是否导出，拼接输出

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        # 解码关键点
        ndim = self.kpt_shape[1]  # 获取关键点的维度
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # 为TFLite导出预计算归一化因子以提高数值稳定性
                y = kpts.view(bs, *self.kpt_shape, -1)  # 重新调整kpts的形状
                grid_h, grid_w = self.shape[2], self.shape[3]  # 获取网格的高和宽
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)  # 创建网格大小的张量
                norm = self.strides / (self.stride[0] * grid_size)  # 计算归一化因子
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm  # 计算归一化的关键点
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)  # 重新调整kpts的形状
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides  # 计算关键点
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)  # 如果有可见性维度，添加sigmoid处理
            return a.view(bs, self.nk, -1)  # 返回调整后的关键点
        else:
            y = kpts.clone()  # 克隆kpts以避免修改原始数据
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
                # 对可见性维度应用sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides  # 计算x坐标
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides  # 计算y坐标
            return y  # 返回解码后的关键点


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""
    # YOLO分类头，将输入张量从形状(x(b,c1,20,20))转换为(x(b,c2))

    export = False  # export mode
    # 导出模式标志

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        # 初始化YOLO分类头，将输入张量从形状(b,c1,20,20)转换为(b,c2)
        super().__init__()  # 调用父类的初始化方法
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)  # 创建卷积层
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        # 自适应平均池化层，将输出调整为(b,c_,1,1)的形状
        self.drop = nn.Dropout(p=0.0, inplace=True)  # dropout层，设置为0.0
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)
        # 线性层，将输出从c_转换为c2

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        # 对输入图像数据执行YOLO模型的前向传播
        if isinstance(x, list):
            x = torch.cat(x, 1)  # 如果输入是列表，将其拼接
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))  # 通过卷积、池化和线性层处理输入
        if self.training:
            return x  # 如果是训练模式，返回输出
        y = x.softmax(1)  # get final output
        # 获取最终输出，应用softmax
        return y if self.export else (y, x)  # 根据导出模式返回结果


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.
    # 实时可变形变换解码器模块（RTDETRDecoder），用于目标检测

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    # 此解码器模块利用变换器架构和可变形卷积来预测图像中对象的边界框和类别标签。它集成了来自多个层的特征，并通过一系列变换器解码层输出最终预测结果。
    """

    export = False  # export mode
    # 导出模式标志

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.
        # 使用给定参数初始化RTDETRDecoder模块

        Args:
            nc (int): Number of classes. Default is 80.
            # 类别数量，默认为80
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            # 主干特征图中的通道数，默认为(512, 1024, 2048)
            hd (int): Dimension of hidden layers. Default is 256.
            # 隐藏层的维度，默认为256
            nq (int): Number of query points. Default is 300.
            # 查询点的数量，默认为300
            ndp (int): Number of decoder points. Default is 4.
            # 解码器点的数量，默认为4
            nh (int): Number of heads in multi-head attention. Default is 8.
            # 多头注意力中的头数，默认为8
            ndl (int): Number of decoder layers. Default is 6.
            # 解码器层的数量，默认为6
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            # 前馈网络的维度，默认为1024
            dropout (float): Dropout rate. Default is 0.
            # dropout率，默认为0
            act (nn.Module): Activation function. Default is nn.ReLU.
            # 激活函数，默认为nn.ReLU
            eval_idx (int): Evaluation index. Default is -1.
            # 评估索引，默认为-1
            nd (int): Number of denoising. Default is 100.
            # 去噪的数量，默认为100
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            # 标签噪声比率，默认为0.5
            box_noise_scale (float): Box noise scale. Default is 1.0.
            # 边框噪声比例，默认为1.0
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            # 是否学习初始查询嵌入，默认为False
        """
        super().__init__()  # 调用父类的初始化方法
        self.hidden_dim = hd  # 隐藏层维度
        self.nhead = nh  # 头数
        self.nl = len(ch)  # 层数
        self.nc = nc  # 类别数量
        self.num_queries = nq  # 查询数量
        self.num_decoder_layers = ndl  # 解码器层数量

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # 创建输入投影模块列表，包含卷积层和批量归一化层
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        # 创建可变形变换器解码器层
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)
        # 创建可变形变换器解码器

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)  # 创建去噪分类嵌入
        self.num_denoising = nd  # 去噪数量
        self.label_noise_ratio = label_noise_ratio  # 标签噪声比率
        self.box_noise_scale = box_noise_scale  # 边框噪声比例

        # Decoder embedding
        self.learnt_init_query = learnt_init_query  # 是否学习初始查询
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)  # 创建目标嵌入
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)  # 创建查询位置头

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))  # 创建编码器输出
        self.enc_score_head = nn.Linear(hd, nc)  # 创建编码器得分头
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)  # 创建编码器边框头

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])  # 创建解码器得分头
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])  # 创建解码器边框头

        self._reset_parameters()  # 重置参数

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        # 执行模块的前向传播，返回输入的边框和分类得分
        from ultralytics.models.utils.ops import get_cdn_group  # 导入函数

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)  # 获取编码器输入

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        # 准备去噪训练

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        # 获取解码器输入

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        # 执行解码器

        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta  # 组合输出
        if self.training:
            return x  # 如果是训练模式，返回输出
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)  # 拼接边框和得分
        return y if self.export else (y, x)  # 根据导出模式返回结果

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        # 为给定形状生成锚框边界框，并进行验证
        anchors = []  # 初始化锚框列表
        for i, (h, w) in enumerate(shapes):  # 遍历形状
            sy = torch.arange(end=h, dtype=dtype, device=device)  # 创建y坐标
            sx = torch.arange(end=w, dtype=dtype, device=device)  # 创建x坐标
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)  # 创建有效宽高张量
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)  # 计算宽高
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))  # 计算锚框的对数
        anchors = anchors.masked_fill(~valid_mask, float("inf"))  # 用无效值填充
        return anchors, valid_mask  # 返回锚框和有效掩码

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # 通过获取输入的投影特征处理并返回编码器输入
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]  # 获取投影特征
        # Get encoder inputs
        feats = []  # 初始化特征列表
        shapes = []  # 初始化形状列表
        for feat in x:  # 遍历特征
            h, w = feat.shape[2:]  # 获取高度和宽度
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))  # 将特征展平并调整维度
            # [nl, 2]
            shapes.append([h, w])  # 记录形状

        # [b, h*w, c]
        feats = torch.cat(feats, 1)  # 拼接特征
        return feats, shapes  # 返回特征和形状

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        # 从提供的特征和形状生成并准备解码器所需的输入
        bs = feats.shape[0]  # 获取批次大小
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)  # 生成锚框
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256  # 通过编码器输出处理特征

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)  # 获取编码器输出得分

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)  # 获取top k索引
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)  # 创建批次索引

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # 获取top k特征
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)  # 获取top k锚框

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors  # 计算参考边框

        enc_bboxes = refer_bbox.sigmoid()  # 对边框应用sigmoid
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)  # 如果有去噪边框，拼接
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # 获取编码器得分

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        # 如果学习初始查询，重复目标嵌入；否则使用top k特征
        if self.training:
            refer_bbox = refer_bbox.detach()  # 在训练时分离边框
            if not self.learnt_init_query:
                embeddings = embeddings.detach()  # 在训练时分离嵌入
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)  # 如果有去噪嵌入，拼接

        return embeddings, refer_bbox, enc_bboxes, enc_scores  # 返回嵌入、参考边框、编码边框和编码得分

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # 初始化或重置模型各个组件的参数，使用预定义的权重和偏置
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc  # 初始化类偏置
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)  # 设置编码器得分头的偏置
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)  # 设置编码器边框头最后一层的权重
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)  # 设置编码器边框头最后一层的偏置
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):  # 遍历解码器得分头和边框头
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)  # 设置解码器得分头的偏置
            constant_(reg_.layers[-1].weight, 0.0)  # 设置解码器边框头最后一层的权重
            constant_(reg_.layers[-1].bias, 0.0)  # 设置解码器边框头最后一层的偏置

        linear_init(self.enc_output[0])  # 初始化编码器输出的第一层
        xavier_uniform_(self.enc_output[0].weight)  # 使用Xavier均匀分布初始化权重
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)  # 初始化目标嵌入权重
        xavier_uniform_(self.query_pos_head.layers[0].weight)  # 初始化查询位置头第一层的权重
        xavier_uniform_(self.query_pos_head.layers[1].weight)  # 初始化查询位置头第二层的权重
        for layer in self.input_proj:  # 遍历输入投影层
            xavier_uniform_(layer[0].weight)  # 初始化每个输入投影层的权重


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.
    # v10检测头，来源于 https://arxiv.org/pdf/2405.14458

    Args:
        nc (int): Number of classes.
        # 类别数量
        ch (tuple): Tuple of channel sizes.
        # 通道大小的元组

    Attributes:
        max_det (int): Maximum number of detections.
        # 最大检测数量

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        # 初始化v10Detect对象
        forward(self, x): Performs forward pass of the v10Detect module.
        # 执行v10Detect模块的前向传播
        bias_init(self): Initializes biases of the Detect module.
        # 初始化Detect模块的偏置
    """

    end2end = True  # 端到端标志

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        # 使用指定的类别数量和输入通道初始化v10Detect对象
        super().__init__(nc, ch)  # 调用父类的初始化方法
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # 计算通道数，取ch[0]的四分之一和类别数量的最小值
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),  # 创建卷积层
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),  # 创建卷积层
                nn.Conv2d(c3, self.nc, 1),  # 创建输出层
            )
            for x in ch  # 遍历输入通道
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)  # 深拷贝cv3