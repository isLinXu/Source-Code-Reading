from pickle import FALSE
from torch import nn
from yolov6.layers.common import BottleRep, RepVGGBlock, RepBlock, BepC3, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvBNSiLU, \
                                MBLABlock, ConvBNHS, Lite_EffiBlockS2, Lite_EffiBlockS1


class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''
    # EfficientRep主干网络
    # EfficientRep是通过硬件感知的神经网络设计手工制作的
    # 使用rep-style结构，EfficientRep对高计算硬件(如GPU)很友好

    def __init__(
        self,
        in_channels=3,          # 输入通道数，默认为3（RGB图像）
        channels_list=None,     # 各个阶段的通道数列表
        num_repeats=None,       # 各个阶段的重复次数列表
        block=RepVGGBlock,      # 基础构建块，默认使用RepVGGBlock
        fuse_P2=False,         # 是否融合P2特征层
        cspsppf=False          # 是否使用CSPSPPF模块
    ):
        super().__init__()

        assert channels_list is not None    # 确保通道数列表不为空
        assert num_repeats is not None      # 确保重复次数列表不为空
        self.fuse_P2 = fuse_P2

        # stem层：第一个卷积层，用于初始特征提取
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        # ERBlock_2：第二阶段，包含一个下采样块和多个重复块
        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2            # 步长为2，实现特征图尺寸减半
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],   # 重复次数由num_repeats指定
                block=block,
            )
        )

        # ERBlock_3：第三阶段，结构同ERBlock_2
        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        # ERBlock_4：第四阶段，结构同ERBlock_2
        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        # 根据block类型和cspsppf参数选择不同的通道融合层
        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        # ERBlock_5：第五阶段，包含额外的特征融合层
        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            ),
            channel_merge_layer(           # 特征融合层，用于增强特征表达
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):
        # 前向传播函数，返回多尺度特征图
        outputs = []
        x = self.stem(x)              # 初始特征提取
        x = self.ERBlock_2(x)         # 第二阶段处理
        if self.fuse_P2:              # 如果需要，保存P2特征
            outputs.append(x)
        x = self.ERBlock_3(x)         # 第三阶段处理
        outputs.append(x)             # 保存P3特征
        x = self.ERBlock_4(x)         # 第四阶段处理
        outputs.append(x)             # 保存P4特征
        x = self.ERBlock_5(x)         # 第五阶段处理
        outputs.append(x)             # 保存P5特征

        return tuple(outputs)         # 返回多尺度特征图元组


class EfficientRep6(nn.Module):
    '''EfficientRep+P6 Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''
    # EfficientRep+P6主干网络
    # EfficientRep是通过硬件感知的神经网络设计手工制作的
    # 使用rep-style结构，EfficientRep对高计算硬件(如GPU)很友好

    def __init__(
        self,
        in_channels=3,          # 输入通道数，默认为3（RGB图像）
        channels_list=None,     # 各个阶段的通道数列表
        num_repeats=None,       # 各个阶段的重复次数列表
        block=RepVGGBlock,      # 基础构建块，默认使用RepVGGBlock
        fuse_P2=False,         # 是否融合P2特征层
        cspsppf=False          # 是否使用CSPSPPF模块
    ):
        super().__init__()

        assert channels_list is not None    # 确保通道数列表不为空
        assert num_repeats is not None      # 确保重复次数列表不为空
        self.fuse_P2 = fuse_P2

        # stem层：第一个卷积层，用于初始特征提取
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        # ERBlock_2：第二阶段，包含一个下采样块和多个重复块
        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2            # 步长为2，实现特征图尺寸减半
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],   # 重复次数由num_repeats指定
                block=block,
            )
        )

        # ERBlock_3：第三阶段，结构同ERBlock_2
        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        # ERBlock_4：第四阶段，结构同ERBlock_2
        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        # ERBlock_5：第五阶段，结构同ERBlock_2
        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            )
        )

        # 根据block类型和cspsppf参数选择不同的通道融合层
        channel_merge_layer = SimSPPF if not cspsppf else SimCSPSPPF

        # ERBlock_6：第六阶段，包含额外的特征融合层
        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                block=block,
            ),
            channel_merge_layer(           # 特征融合层，用于增强特征表达
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):
        # 前向传播函数，返回多尺度特征图
        outputs = []
        x = self.stem(x)              # 初始特征提取
        x = self.ERBlock_2(x)         # 第二阶段处理
        if self.fuse_P2:              # 如果需要，保存P2特征
            outputs.append(x)
        x = self.ERBlock_3(x)         # 第三阶段处理
        outputs.append(x)             # 保存P3特征
        x = self.ERBlock_4(x)         # 第四阶段处理
        outputs.append(x)             # 保存P4特征
        x = self.ERBlock_5(x)         # 第五阶段处理
        outputs.append(x)             # 保存P5特征
        x = self.ERBlock_6(x)         # 第六阶段处理
        outputs.append(x)             # 保存P6特征

        return tuple(outputs)         # 返回多尺度特征图元组


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module.
    """
    # CSPBepBackbone模块

    def __init__(
        self,
        in_channels=3,          # 输入通道数，默认为3（RGB图像）
        channels_list=None,     # 各个阶段的通道数列表
        num_repeats=None,       # 各个阶段的重复次数列表
        block=RepVGGBlock,      # 基础构建块，默认使用RepVGGBlock
        csp_e=float(1)/2,       # CSP模块的扩张系数
        fuse_P2=False,         # 是否融合P2特征层
        cspsppf=False,         # 是否使用CSPSPPF模块
        stage_block_type="BepC3" # 阶段块类型，默认为BepC3
    ):
        super().__init__()

        assert channels_list is not None    # 确保通道数列表不为空
        assert num_repeats is not None      # 确保重复次数列表不为空

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError
        
        self.fuse_P2 = fuse_P2

        # stem层：第一个卷积层，用于初始特征提取
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        # ERBlock_2：第二阶段，包含一个下采样块和多个重复块
        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2            # 步长为2，实现特征图尺寸减半
            ),
            stage_block(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],   # 重复次数由num_repeats指定
                e=csp_e,            # 扩张系数
                block=block,
            )
        )

        # ERBlock_3：第三阶段，结构同ERBlock_2
        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        # ERBlock_4：第四阶段，结构同ERBlock_2
        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        # 根据block类型和cspsppf参数选择不同的通道融合层
        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        # ERBlock_5：第五阶段，包含额外的特征融合层
        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(           # 特征融合层，用于增强特征表达
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):
        # 前向传播函数，返回多尺度特征图
        outputs = []
        x = self.stem(x)              # 初始特征提取
        x = self.ERBlock_2(x)         # 第二阶段处理
        if self.fuse_P2:              # 如果需要，保存P2特征
            outputs.append(x)
        x = self.ERBlock_3(x)         # 第三阶段处理
        outputs.append(x)             # 保存P3特征
        x = self.ERBlock_4(x)         # 第四阶段处理
        outputs.append(x)             # 保存P4特征
        x = self.ERBlock_5(x)         # 第五阶段处理
        outputs.append(x)             # 保存P5特征

        return tuple(outputs)         # 返回多尺度特征图元组


class CSPBepBackbone_P6(nn.Module):
    """
    CSPBepBackbone+P6 module.
    """
    # CSPBepBackbone+P6模块

    def __init__(
        self,
        in_channels=3,          # 输入通道数，默认为3（RGB图像）
        channels_list=None,     # 各个阶段的通道数列表
        num_repeats=None,       # 各个阶段的重复次数列表
        block=RepVGGBlock,      # 基础构建块，默认使用RepVGGBlock
        csp_e=float(1)/2,       # CSP模块的扩张系数
        fuse_P2=False,         # 是否融合P2特征层
        cspsppf=False,         # 是否使用CSPSPPF模块
        stage_block_type="BepC3" # 阶段块类型，默认为BepC3
    ):
        super().__init__()
        assert channels_list is not None    # 确保通道数列表不为空
        assert num_repeats is not None      # 确保重复次数列表不为空

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError
        
        self.fuse_P2 = fuse_P2

        # stem层：第一个卷积层，用于初始特征提取
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        # ERBlock_2：第二阶段，包含一个下采样块和多个重复块
        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2            # 步长为2，实现特征图尺寸减半
            ),
            stage_block(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],   # 重复次数由num_repeats指定
                e=csp_e,            # 扩张系数
                block=block,
            )
        )

        # ERBlock_3：第三阶段，结构同ERBlock_2
        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        # ERBlock_4：第四阶段，结构同ERBlock_2
        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        # 根据block类型和cspsppf参数选择不同的通道融合层
        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        # ERBlock_5：第五阶段，结构同ERBlock_2
        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
        )
        # ERBlock_6：第六阶段，包含额外的特征融合层
        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(           # 特征融合层，用于增强特征表达
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):
        # 前向传播函数，返回多尺度特征图
        outputs = []
        x = self.stem(x)              # 初始特征提取
        x = self.ERBlock_2(x)         # 第二阶段处理
        outputs.append(x)             # 保存P2特征
        x = self.ERBlock_3(x)         # 第三阶段处理
        outputs.append(x)             # 保存P3特征
        x = self.ERBlock_4(x)         # 第四阶段处理
        outputs.append(x)             # 保存P4特征
        x = self.ERBlock_5(x)         # 第五阶段处理
        outputs.append(x)             # 保存P5特征
        x = self.ERBlock_6(x)         # 第六阶段处理
        outputs.append(x)             # 保存P6特征

        return tuple(outputs)         # 返回多尺度特征图元组

class Lite_EffiBackbone(nn.Module):
    '''轻量级高效主干网络
    该网络是YOLOv6中专门为资源受限场景设计的轻量级主干网络。
    主要特点：
    1. 使用轻量级高效块(Lite_EffiBlock)作为基本构建单元
    2. 采用渐进式特征提取策略，逐步增加感受野
    3. 输出多尺度特征图，用于后续的目标检测任务
    '''
    def __init__(self,
                 in_channels,          # 输入通道数，通常为3(RGB图像)
                 mid_channels,         # 中间层通道数列表，控制网络各阶段的特征维度
                 out_channels,         # 输出通道数列表，定义每个阶段的输出特征维度
                 num_repeat=[1, 3, 7, 3]  # 各阶段重复次数，默认为[1,3,7,3]，表示每个阶段堆叠的块数
    ):
        super().__init__()
        out_channels[0]=24            # 将第一层的输出通道数固定为24，这是经验值
        
        # 初始特征提取层
        # 使用ConvBNHS(Conv+BatchNorm+HardSwish)进行特征提取和下采样
        # stride=2将特征图尺寸减半，padding=1保持合适的特征图大小
        self.conv_0 = ConvBNHS(in_channels=in_channels,
                             out_channels=out_channels[0],
                             kernel_size=3,
                             stride=2,
                             padding=1)

        # 第一个轻量级特征提取块
        # 输入为conv_0的输出，通过build_block构建num_repeat[0]个重复单元
        self.lite_effiblock_1 = self.build_block(num_repeat[0],
                                                 out_channels[0],
                                                 mid_channels[1],
                                                 out_channels[1])

        # 第二个轻量级特征提取块
        # 输入为lite_effiblock_1的输出，继续提取更高层次的特征
        self.lite_effiblock_2 = self.build_block(num_repeat[1],
                                                 out_channels[1],
                                                 mid_channels[2],
                                                 out_channels[2])

        # 第三个轻量级特征提取块
        # 输入为lite_effiblock_2的输出，进一步增加感受野
        self.lite_effiblock_3 = self.build_block(num_repeat[2],
                                                 out_channels[2],
                                                 mid_channels[3],
                                                 out_channels[3])

        # 第四个轻量级特征提取块
        # 输入为lite_effiblock_3的输出，提取最高层次的特征
        self.lite_effiblock_4 = self.build_block(num_repeat[3],
                                                 out_channels[3],
                                                 mid_channels[4],
                                                 out_channels[4])

    def forward(self, x):
        '''前向传播函数
        Args:
            x: 输入张量，形状为(batch_size, in_channels, height, width)
        Returns:
            tuple: 包含多尺度特征图的元组，用于后续的检测头
        '''
        outputs = []                    # 存储多尺度特征图的列表
        x = self.conv_0(x)             # 初始特征提取，1/2下采样
        x = self.lite_effiblock_1(x)   # 第一阶段特征提取，进一步下采样
        x = self.lite_effiblock_2(x)   # 第二阶段特征提取
        outputs.append(x)              # 保存中等尺度特征图
        x = self.lite_effiblock_3(x)   # 第三阶段特征提取
        outputs.append(x)              # 保存较大尺度特征图
        x = self.lite_effiblock_4(x)   # 第四阶段特征提取
        outputs.append(x)              # 保存最大尺度特征图
        return tuple(outputs)          # 返回特征图元组，包含三个不同尺度的特征图

    @staticmethod
    def build_block(num_repeat, in_channels, mid_channels, out_channels):
        '''构建轻量级特征提取块
        Args:
            num_repeat: 重复次数，控制块的堆叠数量
            in_channels: 输入通道数
            mid_channels: 中间层通道数，用于通道数的调整
            out_channels: 输出通道数
        Returns:
            nn.Sequential: 包含多个轻量级高效块的序列
        '''
        block_list = nn.Sequential()    # 创建一个Sequential容器，用于顺序存放多个块
        
        # 循环构建num_repeat个块
        for i in range(num_repeat):
            if i == 0:                  # 第一个块使用S2型，进行特征图下采样
                block = Lite_EffiBlockS2(
                            in_channels=in_channels,
                            mid_channels=mid_channels,
                            out_channels=out_channels,
                            stride=2)    # stride=2实现特征图的空间下采样
            else:                       # 其余块使用S1型，保持特征图大小不变
                block = Lite_EffiBlockS1(
                            in_channels=out_channels,  # 注意这里输入通道数变为out_channels
                            mid_channels=mid_channels,
                            out_channels=out_channels,
                            stride=1)    # stride=1保持特征图空间尺寸不变
            block_list.add_module(str(i), block)  # 将构建的块添加到Sequential中
        return block_list              # 返回构建好的特征提取块序列
