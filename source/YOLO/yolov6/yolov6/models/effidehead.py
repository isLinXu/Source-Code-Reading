import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    export = False
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    高效解耦头
    通过硬件感知设计，解耦头使用混合通道方法进行了优化。
    '''
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        """
        初始化检测层。

        参数:
            num_classes (int): 类别数量。
            num_layers (int): 检测层数量。
            inplace (bool): 是否进行原地操作。
            head_layers (list): 解耦头的层列表。
            use_dfl (bool): 是否使用分布焦点损失。
            reg_max (int): 回归的最大值。
        """
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes # 类别数
        self.no = num_classes + 5  # number of outputs per anchor # 每个锚点的输出数量
        self.nl = num_layers  # number of detection layers # 检测层数量
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # 初始化解耦头
        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # 高效解耦头层
        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])

    def initialize_biases(self):
        """
        初始化模型的偏置参数，以帮助模型在训练初期更好地收敛。
        此方法主要针对分类预测(cls_preds)和回归预测(reg_preds)的卷积层进行偏置和权重的初始化。
        """
        # 针对分类预测的卷积层进行初始化
        for conv in self.cls_preds:
            # 计算分类预测的偏置项，并使用prior_prob来反映类别不平衡的情况
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            # 初始化分类预测的权重为0
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        # 针对回归预测的卷积层进行初始化
        for conv in self.reg_preds:
            # 初始化回归预测的偏置项，设置初始值为1.0，有助于模型学习回归目标
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            # 初始化回归预测的权重为0
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        # 初始化投影参数，用于回归预测中的距离计算，不参与梯度更新
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)

        # 初始化投影卷积权重，同样不参与梯度更新
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        """
        实现前向传播过程，根据是否处于训练模式执行不同的逻辑。

        参数:
        - x: 输入数据，可以是单个张量或张量列表。

        返回:
        - x: 经过处理后的特征图。
        - cls_score_list: 分类得分列表。
        - reg_distri_list: 回归分布列表。
        """
        if self.training:
            # 训练模式下，处理每个输入张量，提取分类和回归特征，并进行相应的预测。
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            # 非训练模式下，执行类似的特征提取和预测过程，但根据模型的导出状态和是否使用DFl（分布 focal loss）进行调整。
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)

                if self.export:
                    cls_score_list.append(cls_output)
                    reg_dist_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                    reg_dist_list.append(reg_output.reshape([b, 4, l]))

            if self.export:
                return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_dist_list))

            # 生成锚点和步长张量，用于将回归分布转换为实际边界框。
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    """
    构建EffideHead层。

    EffideHead是一种高效的检测头结构，用于目标检测任务。本函数根据给定的通道列表、锚点数、类别数等参数，
    构建一系列卷积层和预测层，用于分类和回归任务。

    参数:
    - channels_list: 一个列表，包含每个层的通道数。
    - num_anchors: 锚点的数量。
    - num_classes: 类别的数量。
    - reg_max: 回归的最大值，默认为16。
    - num_layers: 层数的数量，默认为3。

    返回:
    - head_layers: 一个Sequential模块，包含构建的EffideHead层。
    """
    # 根据层数选择通道索引列表
    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    # 初始化头部层的Sequential模块
    head_layers = nn.Sequential(
        # stem0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        )
    )
    # 如果层数为4，则添加第四个层级的模块
    if num_layers == 4:
        head_layers.add_module('stem3',
            # stem3
            ConvBNSiLU(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=1,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            ConvBNSiLU(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            ConvBNSiLU(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('cls_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=4 * (reg_max + num_anchors),
                kernel_size=1
            )
        )
    # 返回构建的头部层
    return head_layers
