import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    export = False
    '''Efficient Decoupled Head for fusing anchor-base branches.
    高效解耦头，用于融合基于锚框的分支。
    '''
    def __init__(self, num_classes=80, anchors=None, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        """初始化检测层。

        参数:
            num_classes (int): 类别数量。
            anchors (list 或 tuple): 锚框参数。
            num_layers (int): 检测层数量。
            inplace (bool): 是否使用原地操作。
            head_layers (list): 解耦头的层。
            use_dfl (bool): 是否使用分布焦点损失。
            reg_max (int): 回归的最大值。
        """
        super().__init__()
        # 确保head_layers不为None，这是模型头部层数的重要参数
        assert head_layers is not None
        # 初始化类别数量
        self.nc = num_classes  # number of classes # 类别数量
        # 计算每个锚框的输出数量，包括类别概率和边界框坐标数量
        self.no = num_classes + 5  # number of outputs per anchor # 每个锚框的输出数量
        # 初始化检测层数量
        self.nl = num_layers  # number of detection layers # 检测层数量

        # 根据anchors的类型，计算每个检测层的锚框数量
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        # 初始化grid，用于存储每个检测层的网格信息
        self.grid = [torch.zeros(1)] * num_layers
        # 设置先验概率，用于初始化分类分数和回归预测
        self.prior_prob = 1e-2
        # 设置是否原地操作的标志
        self.inplace = inplace
        # 根据检测层数量，设定每个层的步长
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        # 将步长转化为tensor格式
        self.stride = torch.tensor(stride)
        # 设置是否使用分布焦点损失的标志
        self.use_dfl = use_dfl
        # 初始化reg_max，用于分布焦点损失的最大值
        self.reg_max = reg_max
        # 初始化投影卷积层，用于分布焦点损失的计算
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        # 设置网格单元偏移量
        self.grid_cell_offset = 0.5
        # 设置网格单元尺寸
        self.grid_cell_size = 5.0
        # 初始化anchors，根据步长进行缩放
        self.anchors_init= ((torch.tensor(anchors) / self.stride[:,None])).reshape(self.nl, self.na, 2)

        # Init decouple head
        # 初始化模型的各个模块列表
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_preds_ab = nn.ModuleList()
        self.reg_preds_ab = nn.ModuleList()

        # Efficient decoupled head layers         # 高效解耦头层
        # 解释：下面的循环将根据head_layers中的顺序和数量，为每个模块列表添加相应的层。
        # 这种解耦设计允许每个分支（分类、回归等）有独立的处理层，提高模型的灵活性和性能。
        for i in range(num_layers):
            idx = i*7
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])
            self.cls_preds_ab.append(head_layers[idx+5])
            self.reg_preds_ab.append(head_layers[idx+6])

    def initialize_biases(self):
        """
        初始化模型中的偏置参数，以促进训练稳定性和性能。
        此方法主要针对分类和回归预测层的偏置和权重进行初始化。
        """
        # 初始化分类预测卷积层的偏置和权重
        for conv in self.cls_preds:
            # 计算并设置偏置参数
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            # 初始化权重参数
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        # 初始化另一组分类预测卷积层的偏置和权重
        for conv in self.cls_preds_ab:
            # 计算并设置偏置参数
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            # 初始化权重参数
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        # 初始化回归预测卷积层的偏置和权重
        for conv in self.reg_preds:
            # 计算并设置偏置参数
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)

            # 初始化权重参数
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        # 初始化另一组回归预测卷积层的偏置和权重
        for conv in self.reg_preds_ab:
            # 计算并设置偏置参数
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            # 初始化权重参数
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        # 初始化投影参数，用于回归预测中的距离计算
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)

        # 初始化投影卷积权重参数，用于将投影参数应用于卷积操作
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        if self.training:
            device = x[0].device
            cls_score_list_af = []
            reg_dist_list_af = []
            cls_score_list_ab = []
            reg_dist_list_ab = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)

                #anchor_base
                cls_output_ab = self.cls_preds_ab[i](cls_feat)
                reg_output_ab = self.reg_preds_ab[i](reg_feat)

                cls_output_ab = torch.sigmoid(cls_output_ab)
                cls_output_ab = cls_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                cls_score_list_ab.append(cls_output_ab.flatten(1,3))

                reg_output_ab = reg_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                reg_output_ab[..., 2:4] = ((reg_output_ab[..., 2:4].sigmoid() * 2) ** 2 ) * (self.anchors_init[i].reshape(1, self.na, 1, 1, 2).to(device))
                reg_dist_list_ab.append(reg_output_ab.flatten(1,3))

                #anchor_free
                cls_output_af = self.cls_preds[i](cls_feat)
                reg_output_af = self.reg_preds[i](reg_feat)

                cls_output_af = torch.sigmoid(cls_output_af)
                cls_score_list_af.append(cls_output_af.flatten(2).permute((0, 2, 1)))
                reg_dist_list_af.append(reg_output_af.flatten(2).permute((0, 2, 1)))


            cls_score_list_ab = torch.cat(cls_score_list_ab, axis=1)
            reg_dist_list_ab = torch.cat(reg_dist_list_ab, axis=1)
            cls_score_list_af = torch.cat(cls_score_list_af, axis=1)
            reg_dist_list_af = torch.cat(reg_dist_list_af, axis=1)

            return x, cls_score_list_ab, reg_dist_list_ab, cls_score_list_af, reg_dist_list_af

        else:
            device = x[0].device
            cls_score_list_af = []
            reg_dist_list_af = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)

                #anchor_free
                cls_output_af = self.cls_preds[i](cls_feat)
                reg_output_af = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output_af = reg_output_af.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output_af = self.proj_conv(F.softmax(reg_output_af, dim=1))

                cls_output_af = torch.sigmoid(cls_output_af)

                if self.export:
                    cls_score_list_af.append(cls_output_af)
                    reg_dist_list_af.append(reg_output_af)
                else:
                    cls_score_list_af.append(cls_output_af.reshape([b, self.nc, l]))
                    reg_dist_list_af.append(reg_output_af.reshape([b, 4, l]))

            if self.export:
                return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list_af, reg_dist_list_af))

            cls_score_list_af = torch.cat(cls_score_list_af, axis=-1).permute(0, 2, 1)
            reg_dist_list_af = torch.cat(reg_dist_list_af, axis=-1).permute(0, 2, 1)


            #anchor_free
            anchor_points_af, stride_tensor_af = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes_af = dist2bbox(reg_dist_list_af, anchor_points_af, box_format='xywh')
            pred_bboxes_af *= stride_tensor_af

            pred_bboxes = pred_bboxes_af
            cls_score_list = cls_score_list_af

            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

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
        # cls_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # cls_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * num_anchors,
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
        # cls_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # cls_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * num_anchors,
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
        # cls_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # cls_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
    )

    return head_layers
