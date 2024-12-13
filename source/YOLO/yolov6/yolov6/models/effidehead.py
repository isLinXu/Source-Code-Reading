# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 导入YOLOv6自定义模块
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    
    高效解耦头
    通过硬件感知设计，解耦头使用混合通道方法进行了优化。
    该模块实现了一个高效的检测头，将分类和回归任务解耦，并针对硬件特性进行了优化。
    '''
    # 用于标记是否处于模型导出模式
    export = False

    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        """初始化检测层
        Args:
            num_classes (int): 类别数量，默认80
            num_layers (int): 检测层数量，默认3
            inplace (bool): 是否进行原地操作，默认True
            head_layers (list): 解耦头的层列表，必须提供
            use_dfl (bool): 是否使用分布焦点损失，默认True
            reg_max (int): 回归的最大值，默认16
        """
        super().__init__()
        # 确保提供了头部层
        assert head_layers is not None
        
        # 基本参数设置
        self.nc = num_classes  # number of classes | 类别数量
        self.no = num_classes + 5  # number of outputs per anchor | 每个锚点的输出数量（类别数+5个框体参数）
        self.nl = num_layers  # number of detection layers | 检测层数量
        self.grid = [torch.zeros(1)] * num_layers  # 初始化网格列表
        self.prior_prob = 1e-2  # 先验概率
        self.inplace = inplace  # 是否原地操作
        
        # 设置步长
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]  # strides computed during build | 根据层数计算步长
        self.stride = torch.tensor(stride)
        
        # 分布式焦点损失相关参数
        self.use_dfl = use_dfl  # 是否使用分布焦点损失
        self.reg_max = reg_max  # 回归最大值
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)  # 投影卷积层
        
        # 网格相关参数
        self.grid_cell_offset = 0.5  # 网格单元偏移量
        self.grid_cell_size = 5.0  # 网格单元大小

        # 初始化解耦头的各个组件
        self.stems = nn.ModuleList()  # 主干网络
        self.cls_convs = nn.ModuleList()  # 分类卷积层
        self.reg_convs = nn.ModuleList()  # 回归卷积层
        self.cls_preds = nn.ModuleList()  # 分类预测层
        self.reg_preds = nn.ModuleList()  # 回归预测层

        # Efficient decoupled head layers | 构建高效解耦头层
        for i in range(num_layers):
            idx = i*5  # 每个检测层包含5个组件
            # 按顺序添加各个组件
            self.stems.append(head_layers[idx])  # 添加主干网络层
            self.cls_convs.append(head_layers[idx+1])  # 添加分类卷积层
            self.reg_convs.append(head_layers[idx+2])  # 添加回归卷积层
            self.cls_preds.append(head_layers[idx+3])  # 添加分类预测层
            self.reg_preds.append(head_layers[idx+4])  # 添加回归预测层

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
        """前向传播函数
        根据模型是否处于训练模式，执行不同的处理逻辑。训练模式下直接输出特征，
        而在推理模式下会进行更多后处理操作。

        Args:
            x (List[Tensor]): 输入特征图列表，包含多个尺度的特征图

        Returns:
            训练模式下返回:
                x: 处理后的特征图
                cls_score_list: 分类预测结果
                reg_distri_list: 回归预测结果
            推理模式下返回:
                输出张量，包含预测框坐标、置信度和类别分数
        """
        if self.training:
            # 训练模式的处理逻辑
            cls_score_list = []  # 存储所有尺度的分类预测结果
            reg_distri_list = []  # 存储所有尺度的回归预测结果

            for i in range(self.nl):  # 遍历每个检测层
                x[i] = self.stems[i](x[i])  # 通过主干网络处理特征
                cls_x = x[i]  # 分类分支的输入
                reg_x = x[i]  # 回归分支的输入
                
                # 分类分支处理
                cls_feat = self.cls_convs[i](cls_x)  # 分类特征提取
                cls_output = self.cls_preds[i](cls_feat)  # 分类预测
                
                # 回归分支处理
                reg_feat = self.reg_convs[i](reg_x)  # 回归特征提取
                reg_output = self.reg_preds[i](reg_feat)  # 回归预测

                # 对分类输出进行sigmoid激活
                cls_output = torch.sigmoid(cls_output)
                
                # 重塑输出维度并添加到结果列表
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))  # [batch, h*w, num_classes]
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))  # [batch, h*w, 4*(reg_max+1)]

            # 合并所有尺度的预测结果
            cls_score_list = torch.cat(cls_score_list, axis=1)  # 在第1维度上拼接所有分类预测
            reg_distri_list = torch.cat(reg_distri_list, axis=1)  # 在第1维度上拼接所有回归预测

            return x, cls_score_list, reg_distri_list
        else:
            # 推理模式的处理逻辑
            cls_score_list = []  # 存储所有尺度的分类得分
            reg_dist_list = []   # 存储所有尺度的回归分布

            for i in range(self.nl):  # 遍历每个检测层
                b, _, h, w = x[i].shape  # 获取特征图尺寸
                l = h * w  # 特征图的像素总数
                
                # 特征提取和预测过程
                x[i] = self.stems[i](x[i])  # 主干网络处理
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)  # 分类特征提取
                cls_output = self.cls_preds[i](cls_feat)  # 分类预测
                reg_feat = self.reg_convs[i](reg_x)  # 回归特征提取
                reg_output = self.reg_preds[i](reg_feat)  # 回归预测

                # 使用分布式焦点损失（DFL）时的特殊处理
                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)  # 重塑维度
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))  # 应用投影卷积

                # 分类输出进行sigmoid激活
                cls_output = torch.sigmoid(cls_output)

                # 根据是否为导出模式选择不同的输出处理方式
                if self.export:
                    cls_score_list.append(cls_output)
                    reg_dist_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.nc, l]))  # [batch, num_classes, h*w]
                    reg_dist_list.append(reg_output.reshape([b, 4, l]))  # [batch, 4, h*w]

            # 导出模式下直接返回拼接结果
            if self.export:
                return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_dist_list))

            # 合并所有尺度的预测结果
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)  # [batch, total_anchors, num_classes]
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)    # [batch, total_anchors, 4]

            # 生成锚点和对应的步长张量
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, 
                device=x[0].device, is_eval=True, mode='af'
            )

            # 将回归分布转换为边界框坐标
            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')  # 预测框坐标
            pred_bboxes *= stride_tensor  # 应用步长缩放

            # 组合最终输出：预测框坐标、置信度和类别分数
            return torch.cat(
                [
                    pred_bboxes,  # 预测框坐标 [batch, n_anchors, 4]
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),  # 置信度
                    cls_score_list  # 类别分数 [batch, n_anchors, n_classes]
                ],
                axis=-1)  # 最终输出形状 [batch, n_anchors, 5+n_classes]


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    """构建高效解耦检测头（EffideHead）层
    
    该函数构建YOLOv6中的高效解耦检测头，包含多个检测层级。每个层级包含以下组件：
    1. stem: 1x1卷积进行通道调整
    2. cls_conv: 3x3卷积提取分类特征
    3. reg_conv: 3x3卷积提取回归特征
    4. cls_pred: 1x1卷积输出分类预测
    5. reg_pred: 1x1卷积输出回归预测

    Args:
        channels_list (List[int]): 各层的通道数列表
        num_anchors (int): 每个网格点的锚框数量
        num_classes (int): 目标类别数量
        reg_max (int): 回归最大值，用于分布式预测，默认16
        num_layers (int): 检测头的层数，支持3或4层，默认3

    Returns:
        nn.Sequential: 包含所有检测头层的顺序模块
    """
    # 根据检测层数选择特征图索引
    # 3层结构使用第6、8、10层特征图
    # 4层结构使用第8、9、10、11层特征图
    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    # 创建顺序模块存储所有层
    head_layers = nn.Sequential(
        # stem0: 1x1卷积调整通道数，保持空间分辨率不变
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],  # 输入通道数
            out_channels=channels_list[chx[0]],  # 输出通道数与输入相同
            kernel_size=1,  # 1x1卷积核
            stride=1  # 步长为1，维持特征图大小
        ),
        # cls_conv0: 3x3卷积提取分类特征
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,  # 3x3卷积核提取空间特征
            stride=1
        ),
        # reg_conv0: 3x3卷积提取回归特征
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0: 1x1卷积输出分类预测
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,  # 输出通道数 = 类别数 × 锚框数
            kernel_size=1  # 1x1卷积进行预测
        ),
        # reg_pred0: 1x1卷积输出回归预测
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + num_anchors),  # 输出通道数 = 4(坐标) × (reg_max + 锚框数)
            kernel_size=1
        ),
        # 第二层检测头 (stride=16)
        # stem1: 特征调整
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1: 分类特征提取
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1: 回归特征提取
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1: 分类预测
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1: 回归预测
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # 第三层检测头 (stride=32)
        # stem2: 特征调整
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2: 分类特征提取
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2: 回归特征提取
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2: 分类预测
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2: 回归预测
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        )
    )

    # 如果使用4层检测头，添加第四层 (stride=64)
    if num_layers == 4:
        # stem3: 特征调整层
        head_layers.add_module('stem3',
            ConvBNSiLU(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=1,
                stride=1
            )
        )
        # cls_conv3: 分类特征提取层
        head_layers.add_module('cls_conv3',
            ConvBNSiLU(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        # reg_conv3: 回归特征提取层
        head_layers.add_module('reg_conv3',
            ConvBNSiLU(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        # cls_pred3: 分类预测层
        head_layers.add_module('cls_pred3',
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
        )
        # reg_pred3: 回归预测层
        head_layers.add_module('reg_pred3',
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=4 * (reg_max + num_anchors),
                kernel_size=1
            )
        )

    return head_layers  # 返回构建好的检测头模块
