class RepPANNeck(nn.Module):
    """RepPANNeck模块
    EfficientRep是该模型的默认主干网络。
    RepPANNeck在特征融合能力和硬件效率之间取得了平衡。
    """

    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock):
        super().__init__()

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None  # 确保重复次数不为空

        # 定义RepPAN的各个层
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],  # 输入通道数为第4和第6层通道数之和
            out_channels=channels_list[5],  # 输出通道数为第6层通道数
            n=num_repeats[5],  # 重复次数为第6层的重复次数
            block=block  # 使用指定的块类型
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],  # 输入通道数为第3和第7层通道数之和
            out_channels=channels_list[6],  # 输出通道数为第7层通道数
            n=num_repeats[6],  # 重复次数为第7层的重复次数
            block=block  # 使用指定的块类型
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],  # 输入通道数为第7和第8层通道数之和
            out_channels=channels_list[8],  # 输出通道数为第9层通道数
            n=num_repeats[7],  # 重复次数为第8层的重复次数
            block=block  # 使用指定的块类型
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],  # 输入通道数为第6和第10层通道数之和
            out_channels=channels_list[10],  # 输出通道数为第11层通道数
            n=num_repeats[8],  # 重复次数为第9层的重复次数
            block=block  # 使用指定的块类型
        )

        # 定义减少通道的卷积层
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4],  # 输入通道数为第5层通道数
            out_channels=channels_list[5],  # 输出通道数为第6层通道数
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        # 定义上采样层
        self.upsample0 = Transpose(
            in_channels=channels_list[5],  # 输入通道数为第6层通道数
            out_channels=channels_list[5],  # 输出通道数为第6层通道数
        )

        # 定义第二个减少通道的卷积层
        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5],  # 输入通道数为第6层通道数
            out_channels=channels_list[6],  # 输出通道数为第7层通道数
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        # 定义第二个上采样层
        self.upsample1 = Transpose(
            in_channels=channels_list[6],  # 输入通道数为第7层通道数
            out_channels=channels_list[6]  # 输出通道数为第7层通道数
        )

        # 定义下采样层
        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6],  # 输入通道数为第7层通道数
            out_channels=channels_list[7],  # 输出通道数为第8层通道数
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        # 定义第三个下采样层
        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8],  # 输入通道数为第9层通道数
            out_channels=channels_list[9],  # 输出通道数为第10层通道数
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

    def upsample_enable_quant(self, num_bits, calib_method):
        """启用上采样量化
        
        Args:
            num_bits: 量化位数
            calib_method: 校准方法
        """
        print("Insert fakequant after upsample")
        # 在上采样操作后插入假量化，以构建TensorRT引擎
        from pytorch_quantization import nn as quant_nn  # 导入量化相关模块
        from pytorch_quantization.tensor_quant import QuantDescriptor  # 导入量化描述符
        conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)  # 创建量化描述符
        self.upsample_feat0_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)  # 为第一个上采样特征创建量化器
        self.upsample_feat1_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)  # 为第二个上采样特征创建量化器
        # global _QUANT
        self._QUANT = True  # 设置量化标志为True

    def forward(self, input):
        """前向传播函数
        
        Args:
            input: 输入张量，包含多个特征图
            
        Returns:
            outputs: 包含多个PAN输出的列表
        """
        (x2, x1, x0) = input  # 解包输入特征图

        fpn_out0 = self.reduce_layer0(x0)  # 通过减少层处理第一个输入特征图
        upsample_feat0 = self.upsample0(fpn_out0)  # 对处理后的特征图进行上采样
        if hasattr(self, '_QUANT') and self._QUANT is True:  # 检查是否启用了量化
            upsample_feat0 = self.upsample_feat0_quant(upsample_feat0)  # 对上采样特征进行量化处理
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)  # 将上采样特征与第二个输入特征图连接
        f_out0 = self.Rep_p4(f_concat_layer0)  # 通过RepBlock处理连接后的特征图

        fpn_out1 = self.reduce_layer1(f_out0)  # 通过减少层处理第二个输出特征图
        upsample_feat1 = self.upsample1(fpn_out1)  # 对处理后的特征图进行上采样
        if hasattr(self, '_QUANT') and self._QUANT is True:  # 检查是否启用了量化
            upsample_feat1 = self.upsample_feat1_quant(upsample_feat1)  # 对上采样特征进行量化处理
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)  # 将上采样特征与第三个输入特征图连接
        pan_out2 = self.Rep_p3(f_concat_layer1)  # 通过RepBlock处理连接后的特征图

        down_feat1 = self.downsample2(pan_out2)  # 对第二个输出特征图进行下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 将下采样特征与前一个特征图连接
        pan_out1 = self.Rep_n3(p_concat_layer1)  # 通过RepBlock处理连接后的特征图

        down_feat0 = self.downsample1(pan_out1)  # 对第一个输出特征图进行下采样
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)  # 将下采样特征与前一个特征图连接
        pan_out0 = self.Rep_n4(p_concat_layer2)  # 通过RepBlock处理连接后的特征图

        outputs = [pan_out2, pan_out1, pan_out0]  # 将所有输出特征图放入列表中

        return outputs  # 返回输出特征图列表


class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck模块
    """

    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock):
        super().__init__()

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None  # 确保重复次数不为空

        # 定义减少通道的卷积层
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4],  # 输入通道数为第5层通道数
            out_channels=channels_list[5],  # 输出通道数为第6层通道数
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]],  # 输入通道数为第4和第3层通道数
            out_channels=channels_list[5],  # 输出通道数为第6层通道数
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5],  # 输入通道数为第6层通道数
            out_channels=channels_list[5],  # 输出通道数为第6层通道数
            n=num_repeats[5],  # 重复次数为第6层的重复次数
            block=block  # 使用指定的块类型
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5],  # 输入通道数为第6层通道数
            out_channels=channels_list[6],  # 输出通道数为第7层通道数
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]],  # 输入通道数为第3和第2层通道数
            out_channels=channels_list[6],  # 输出通道数为第7层通道数
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6],  # 输入通道数为第7层通道数
            out_channels=channels_list[6],  # 输出通道数为第7层通道数
            n=num_repeats[6],  # 重复次数为第7层的重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6],  # 输入通道数为第7层通道数
            out_channels=channels_list[7],  # 输出通道数为第8层通道数
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],  # 输入通道数为第7和第8层通道数之和
            out_channels=channels_list[8],  # 输出通道数为第9层通道数
            n=num_repeats[7],  # 重复次数为第8层的重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8],  # 输入通道数为第9层通道数
            out_channels=channels_list[9],  # 输出通道数为第10层通道数
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],  # 输入通道数为第6和第10层通道数之和
            out_channels=channels_list[10],  # 输出通道数为第11层通道数
            n=num_repeats[8],  # 重复次数为第9层的重复次数
            block=block  # 使用指定的块类型
        )

    def forward(self, input):
        """前向传播函数
        
        Args:
            input: 输入张量，包含多个特征图
        
        Returns:
            outputs: 包含多个PAN输出的列表
        """
        (x3, x2, x1, x0) = input  # 解包输入特征图

        fpn_out0 = self.reduce_layer0(x0)  # 通过减少层处理第一个输入特征图
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])  # 将减少后的特征图与x1和x2进行BiFusion
        f_out0 = self.Rep_p4(f_concat_layer0)  # 通过RepBlock处理连接后的特征图

        fpn_out1 = self.reduce_layer1(f_out0)  # 通过减少层处理第二个输出特征图
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])  # 将减少后的特征图与x2和x3进行BiFusion
        pan_out2 = self.Rep_p3(f_concat_layer1)  # 通过RepBlock处理连接后的特征图

        down_feat1 = self.downsample2(pan_out2)  # 对第二个输出特征图进行下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 将下采样特征与前一个特征图连接
        pan_out1 = self.Rep_n3(p_concat_layer1)  # 通过RepBlock处理连接后的特征图

        down_feat0 = self.downsample1(pan_out1)  # 对第一个输出特征图进行下采样
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)  # 将下采样特征与前一个特征图连接
        pan_out0 = self.Rep_n4(p_concat_layer2)  # 通过RepBlock处理连接后的特征图

        outputs = [pan_out2, pan_out1, pan_out0]  # 将所有输出特征图放入列表中

        return outputs  # 返回输出特征图列表


class RepPANNeck6(nn.Module):
    """RepPANNeck+P6模块
    EfficientRep是该模型的默认主干网络。
    RepPANNeck在特征融合能力和硬件效率之间取得平衡。
    """
    
    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock):
        super().__init__()

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None  # 确保重复次数不为空

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5],  # 输入通道数为1024
            out_channels=channels_list[6],  # 输出通道数为512
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[6],  # 输入通道数为512
            out_channels=channels_list[6],  # 输出通道数为512
        )

        self.Rep_p5 = RepBlock(
            in_channels=channels_list[4] + channels_list[6],  # 输入通道数为768（256 + 512）
            out_channels=channels_list[6],  # 输出通道数为512
            n=num_repeats[6],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 输入通道数为512
            out_channels=channels_list[7],  # 输出通道数为256
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[7],  # 输入通道数为256
            out_channels=channels_list[7]  # 输出通道数为256
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[7],  # 输入通道数为512（256 + 256）
            out_channels=channels_list[7],  # 输出通道数为256
            n=num_repeats[7],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 输入通道数为256
            out_channels=channels_list[8],  # 输出通道数为128
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.upsample2 = Transpose(
            in_channels=channels_list[8],  # 输入通道数为128
            out_channels=channels_list[8]  # 输出通道数为128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[8],  # 输入通道数为256（128 + 128）
            out_channels=channels_list[8],  # 输出通道数为128
            n=num_repeats[8],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 输入通道数为128
            out_channels=channels_list[8],  # 输出通道数为128
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8],  # 输入通道数为256（128 + 128）
            out_channels=channels_list[9],  # 输出通道数为256
            n=num_repeats[9],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 输入通道数为256
            out_channels=channels_list[9],  # 输出通道数为256
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9],  # 输入通道数为512（256 + 256）
            out_channels=channels_list[10],  # 输出通道数为512
            n=num_repeats[10],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 输入通道数为512
            out_channels=channels_list[10],  # 输出通道数为512
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10],  # 输入通道数为1024（512 + 512）
            out_channels=channels_list[11],  # 输出通道数为1024
            n=num_repeats[11],  # 重复次数
            block=block  # 使用指定的块类型
        )

    def forward(self, input):
        """前向传播函数
        
        Args:
            input: 输入张量，包含多个特征图
        
        Returns:
            outputs: 包含多个PAN输出的列表
        """
        (x3, x2, x1, x0) = input  # 解包输入特征图

        fpn_out0 = self.reduce_layer0(x0)  # 通过减少层处理第一个输入特征图
        upsample_feat0 = self.upsample0(fpn_out0)  # 对fpn_out0进行上采样
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)  # 将上采样特征与x1连接
        f_out0 = self.Rep_p5(f_concat_layer0)  # 通过RepBlock处理连接后的特征图

        fpn_out1 = self.reduce_layer1(f_out0)  # 通过减少层处理第二个输出特征图
        upsample_feat1 = self.upsample1(fpn_out1)  # 对fpn_out1进行上采样
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)  # 将上采样特征与x2连接
        f_out1 = self.Rep_p4(f_concat_layer1)  # 通过RepBlock处理连接后的特征图

        fpn_out2 = self.reduce_layer2(f_out1)  # 通过减少层处理第三个输出特征图
        upsample_feat2 = self.upsample2(fpn_out2)  # 对fpn_out2进行上采样
        f_concat_layer2 = torch.cat([upsample_feat2, x3], 1)  # 将上采样特征与x3连接
        pan_out3 = self.Rep_p3(f_concat_layer2)  # 通过RepBlock处理连接后的特征图，得到P3

        down_feat2 = self.downsample2(pan_out3)  # 对P3进行下采样
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)  # 将下采样特征与fpn_out2连接
        pan_out2 = self.Rep_n4(p_concat_layer2)  # 通过RepBlock处理连接后的特征图，得到P4

        down_feat1 = self.downsample1(pan_out2)  # 对P4进行下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 将下采样特征与fpn_out1连接
        pan_out1 = self.Rep_n5(p_concat_layer1)  # 通过RepBlock处理连接后的特征图，得到P5

        down_feat0 = self.downsample0(pan_out1)  # 对P5进行下采样
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)  # 将下采样特征与fpn_out0连接
        pan_out0 = self.Rep_n6(p_concat_layer0)  # 通过RepBlock处理连接后的特征图，得到P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]  # 将所有输出特征图放入列表中

        return outputs  # 返回输出特征图列表
class RepBiFPANNeck6(nn.Module):
    """RepBiFPANNeck_P6模块
    """

    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock):
        super().__init__()

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None  # 确保重复次数不为空

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5],  # 输入通道数为1024
            out_channels=channels_list[6],  # 输出通道数为512
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]],  # 输入通道数为768（256 + 512）
            out_channels=channels_list[6],  # 输出通道数为512
        )

        self.Rep_p5 = RepBlock(
            in_channels=channels_list[6],  # 输入通道数为512
            out_channels=channels_list[6],  # 输出通道数为512
            n=num_repeats[6],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 输入通道数为512
            out_channels=channels_list[7],  # 输出通道数为256
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]],  # 输入通道数为512（256 + 256）
            out_channels=channels_list[7],  # 输出通道数为256
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[7],  # 输入通道数为256
            out_channels=channels_list[7],  # 输出通道数为256
            n=num_repeats[7],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 输入通道数为256
            out_channels=channels_list[8],  # 输出通道数为128
            kernel_size=1,  # 卷积核大小为1
            stride=1  # 步长为1
        )

        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]],  # 输入通道数为256（128 + 128）
            out_channels=channels_list[8],  # 输出通道数为128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[8],  # 输入通道数为128
            out_channels=channels_list[8],  # 输出通道数为128
            n=num_repeats[8],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 输入通道数为128
            out_channels=channels_list[8],  # 输出通道数为128
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8],  # 输入通道数为256（128 + 128）
            out_channels=channels_list[9],  # 输出通道数为256
            n=num_repeats[9],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 输入通道数为256
            out_channels=channels_list[9],  # 输出通道数为256
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9],  # 输入通道数为512（256 + 256）
            out_channels=channels_list[10],  # 输出通道数为512
            n=num_repeats[10],  # 重复次数
            block=block  # 使用指定的块类型
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 输入通道数为512
            out_channels=channels_list[10],  # 输出通道数为512
            kernel_size=3,  # 卷积核大小为3
            stride=2  # 步长为2
        )

        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10],  # 输入通道数为1024（512 + 512）
            out_channels=channels_list[11],  # 输出通道数为1024
            n=num_repeats[11],  # 重复次数
            block=block  # 使用指定的块类型
        )

    def forward(self, input):
        """前向传播函数
        
        Args:
            input: 输入张量，包含多个特征图
        
        Returns:
            outputs: 包含多个PAN输出的列表
        """
        (x4, x3, x2, x1, x0) = input  # 解包输入特征图

        fpn_out0 = self.reduce_layer0(x0)  # 通过减少层处理第一个输入特征图
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])  # 将减少后的特征图与x1和x2连接
        f_out0 = self.Rep_p5(f_concat_layer0)  # 通过RepBlock处理连接后的特征图

        fpn_out1 = self.reduce_layer1(f_out0)  # 通过减少层处理第二个输出特征图
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])  # 将减少后的特征图与x2和x3连接
        f_out1 = self.Rep_p4(f_concat_layer1)  # 通过RepBlock处理连接后的特征图

        fpn_out2 = self.reduce_layer2(f_out1)  # 通过减少层处理第三个输出特征图
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])  # 将减少后的特征图与x3和x4连接
        pan_out3 = self.Rep_p3(f_concat_layer2)  # 通过RepBlock处理连接后的特征图，得到P3

        down_feat2 = self.downsample2(pan_out3)  # 对P3进行下采样
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)  # 将下采样特征与fpn_out2连接
        pan_out2 = self.Rep_n4(p_concat_layer2)  # 通过RepBlock处理连接后的特征图，得到P4

        down_feat1 = self.downsample1(pan_out2)  # 对P4进行下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 将下采样特征与fpn_out1连接
        pan_out1 = self.Rep_n5(p_concat_layer1)  # 通过RepBlock处理连接后的特征图，得到P5

        down_feat0 = self.downsample0(pan_out1)  # 对P5进行下采样
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)  # 将下采样特征与fpn_out0连接
        pan_out0 = self.Rep_n6(p_concat_layer0)  # 通过RepBlock处理连接后的特征图，得到P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]  # 将所有输出特征图放入列表中

        return outputs  # 返回输出特征图列表

class RepBiFPANNeck6(nn.Module):
    """RepBiFPANNeck_P6 Module
    RepBiFPANNeck P6模块 - 一个改进的特征金字塔网络，用于特征融合
    """
    # [64, 128, 256, 512, 768, 1024]  # 输入通道数列表
    # [512, 256, 128, 256, 512, 1024] # 输出通道数列表

    def __init__(
        self,
        channels_list=None,  # 通道数列表
        num_repeats=None,    # 重复次数列表
        block=RepVGGBlock    # 基础构建块，默认使用RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None    # 确保重复次数列表不为空

        # 第一个下采样层，将通道数从1024降到512
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        # 第一个双向融合层，融合768和512通道的特征
        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]], # 768, 512
            out_channels=channels_list[6], # 512
        )

        # 第一个RepBlock，处理512通道的特征
        self.Rep_p5 = RepBlock(
            in_channels=channels_list[6], # 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            block=block
        )

        # 第二个下采样层，将通道数从512降到256
        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        # 第二个双向融合层，融合512和256通道的特征
        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]], # 512, 256
            out_channels=channels_list[7], # 256
        )

        # 第二个RepBlock，处理256通道的特征
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            block=block
        )

        # 第三个下采样层，将通道数从256降到128
        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        # 第三个双向融合层，融合256和128通道的特征
        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]], # 256, 128
            out_channels=channels_list[8], # 128
        )

        # 第三个RepBlock，处理128通道的特征
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            block=block
        )

        # 上采样层2，通道数保持128不变
        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        # 第四个RepBlock，处理256通道的特征(128+128)
        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            block=block
        )

        # 上采样层1，通道数保持256不变
        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        # 第五个RepBlock，处理512通道的特征(256+256)
        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            block=block
        )

        # 上采样层0，通道数保持512不变
        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        # 第六个RepBlock，处理1024通道的特征(512+512)
        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            block=block
        )


    def forward(self, input):
        # 解包输入的特征图元组，从小特征图到大特征图的顺序
        (x4, x3, x2, x1, x0) = input  # x0最大，x4最小

        # 自顶向下路径(top-down path) - 特征图尺寸逐渐变大，通道数逐渐减小
        fpn_out0 = self.reduce_layer0(x0)  # 1024->512通道
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])  # 融合三个特征图
        f_out0 = self.Rep_p5(f_concat_layer0)  # 通过RepBlock处理

        fpn_out1 = self.reduce_layer1(f_out0)  # 512->256通道
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])  # 融合三个特征图
        f_out1 = self.Rep_p4(f_concat_layer1)  # 通过RepBlock处理

        fpn_out2 = self.reduce_layer2(f_out1)  # 256->128通道
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])  # 融合三个特征图
        pan_out3 = self.Rep_p3(f_concat_layer2)  # P3输出 - 最小的特征图

        # 自底向上路径(bottom-up path) - 特征图尺寸逐渐变小，通道数逐渐增加
        down_feat2 = self.downsample2(pan_out3)  # 对P3下采样
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)  # 特征图拼接
        pan_out2 = self.Rep_n4(p_concat_layer2)  # P4输出

        down_feat1 = self.downsample1(pan_out2)  # 对P4下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 特征图拼接
        pan_out1 = self.Rep_n5(p_concat_layer1)  # P5输出

        down_feat0 = self.downsample0(pan_out1)  # 对P5下采样
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)  # 特征图拼接
        pan_out0 = self.Rep_n6(p_concat_layer0)  # P6输出 - 最大的特征图

        # 返回所有输出特征图，从小到大的顺序：[P3, P4, P5, P6]
        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepPANNeck(nn.Module):
    """
    CSPRepPANNeck module.
    CSP(Cross Stage Partial Network)结构的RepPAN颈部网络模块
    """

    def __init__(
        self,
        channels_list=None,     # 通道数列表
        num_repeats=None,       # 重复次数列表
        block=BottleRep,        # 基础构建块，默认使用BottleRep
        csp_e=float(1)/2,       # CSP网络的扩展因子，默认为0.5
        stage_block_type="BepC3" # 阶段块类型，可选BepC3或MBLABlock
    ):
        super().__init__()

        # 根据stage_block_type选择不同的块类型
        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None    # 确保重复次数列表不为空

        # 第一个RepBlock，处理768通道的特征(512 + 256)
        self.Rep_p4 = stage_block(
            in_channels=channels_list[3] + channels_list[5], # 512 + 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        # 第二个RepBlock，处理384通道的特征(256 + 128)
        self.Rep_p3 = stage_block(
            in_channels=channels_list[2] + channels_list[6], # 256 + 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        # 第三个RepBlock，处理256通道的特征(128 + 128)
        self.Rep_n3 = stage_block(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        # 第四个RepBlock，处理512通道的特征(256 + 256)
        self.Rep_n4 = stage_block(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        # 第一个下采样层，将通道数从1024降到256
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        # 第一个上采样层，通道数保持256不变
        self.upsample0 = Transpose(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
        )

        # 第二个下采样层，将通道数从256降到128
        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        # 第二个上采样层，通道数保持128不变
        self.upsample1 = Transpose(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6] # 128
        )

        # 第一个下采样层，通道数从128变为128
        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        # 第二个下采样层，通道数从256变为256
        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

    def forward(self, input):
        # 解包输入的特征图元组，从小特征图到大特征图的顺序
        (x2, x1, x0) = input  # x0最大，x2最小

        # 自顶向下路径(top-down path) - 特征图尺寸逐渐变大，通道数逐渐减小
        fpn_out0 = self.reduce_layer0(x0)  # 1024->256通道
        upsample_feat0 = self.upsample0(fpn_out0)  # 上采样
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)  # 特征图拼接
        f_out0 = self.Rep_p4(f_concat_layer0)  # 通过RepBlock处理

        fpn_out1 = self.reduce_layer1(f_out0)  # 256->128通道
        upsample_feat1 = self.upsample1(fpn_out1)  # 上采样
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)  # 特征图拼接
        pan_out2 = self.Rep_p3(f_concat_layer1)  # 通过RepBlock处理

        down_feat1 = self.downsample2(pan_out2)  # 对P2下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 特征图拼接
        pan_out1 = self.Rep_n3(p_concat_layer1)  # P1输出

        down_feat0 = self.downsample1(pan_out1)  # 对P1下采样
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)  # 特征图拼接
        pan_out0 = self.Rep_n4(p_concat_layer2)  # P0输出

        # 返回所有输出特征图，从小到大的顺序：[P2, P1, P0]
        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck module.
    CSP(Cross Stage Partial Network)结构的RepPAN颈部网络模块
    """

    def __init__(
        self,
        channels_list=None,     # 通道数列表
        num_repeats=None,       # 重复次数列表
        block=BottleRep,        # 基础构建块，默认使用BottleRep
        csp_e=float(1)/2,       # CSP网络的扩展因子，默认为0.5
        stage_block_type="BepC3" # 阶段块类型，可选BepC3或MBLABlock
    ):
        super().__init__()

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None    # 确保重复次数列表不为空

        # 根据stage_block_type选择不同的块类型
        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        # 第一个下采样层，将通道数从1024降到256
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        # 第一个双向融合层，融合512和256通道的特征
        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]], # 512, 256
            out_channels=channels_list[5], # 256
        )

        # 第一个RepBlock，处理256通道的特征
        self.Rep_p4 = stage_block(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        # 第二个下采样层，将通道数从256降到128
        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        # 第二个双向融合层，融合256和128通道的特征
        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]], # 256, 128
            out_channels=channels_list[6], # 128
        )

        # 第二个RepBlock，处理128通道的特征
        self.Rep_p3 = stage_block(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        # 第一个下采样层，将通道数从128降到128
        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        # 第三个RepBlock，处理256通道的特征(128 + 128)
        self.Rep_n3 = stage_block(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        # 第二个下采样层，将通道数从256降到256
        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        # 第四个RepBlock，处理512通道的特征(256 + 256)
        self.Rep_n4 = stage_block(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

    def forward(self, input):
        # 解包输入的特征图元组，从小特征图到大特征图的顺序
        (x3, x2, x1, x0) = input  # x0最大，x3最小

        # 自顶向下路径(top-down path) - 特征图尺寸逐渐变大，通道数逐渐减小
        fpn_out0 = self.reduce_layer0(x0)  # 1024->256通道
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])  # 融合三个特征图
        f_out0 = self.Rep_p4(f_concat_layer0)  # 通过RepBlock处理

        fpn_out1 = self.reduce_layer1(f_out0)  # 256->128通道
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])  # 融合三个特征图
        pan_out2 = self.Rep_p3(f_concat_layer1)  # 通过RepBlock处理

        down_feat1 = self.downsample2(pan_out2)  # 对P2下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 特征图拼接
        pan_out1 = self.Rep_n3(p_concat_layer1)  # P1输出

        down_feat0 = self.downsample1(pan_out1)  # 对P1下采样
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)  # 特征图拼接
        pan_out0 = self.Rep_n4(p_concat_layer2)  # P0输出

        # 返回所有输出特征图，从小到大的顺序：[P2, P1, P0]
        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepPANNeck_P6(nn.Module):
    """CSPRepPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2,
        stage_block_type="BepC3"
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = stage_block(
            in_channels=channels_list[4] + channels_list[6], # 768 + 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7] # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=channels_list[3] + channels_list[7], # 512 + 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample2 = Transpose(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8] # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[2] + channels_list[8], # 256 + 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = stage_block(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            e=csp_e,
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = stage_block(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            e=csp_e,
            block=block
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = stage_block(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            e=csp_e,
            block=block
        )


    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        upsample_feat2 = self.upsample2(fpn_out2)
        f_concat_layer2 = torch.cat([upsample_feat2, x3], 1)
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputsclass CSPRepPANNeck_P6(nn.Module):
            """CSPRepPANNeck_P6 Module
            CSPRepPANNeck P6模块 - 一个改进的特征金字塔网络，用于特征融合
            """
            # [64, 128, 256, 512, 768, 1024]  # 输入通道数列表
            # [512, 256, 128, 256, 512, 1024] # 输出通道数列表
        
            def __init__(
                self,
                channels_list=None,  # 通道数列表
                num_repeats=None,    # 重复次数列表
                block=BottleRep,     # 基础构建块，默认使用BottleRep
                csp_e=float(1)/2,    # CSP网络的扩展因子，默认为0.5
                stage_block_type="BepC3" # 阶段块类型，可选BepC3或MBLABlock
            ):
                super().__init__()
        
                assert channels_list is not None  # 确保通道列表不为空
                assert num_repeats is not None    # 确保重复次数列表不为空
        
                # 根据stage_block_type选择不同的块类型
                if stage_block_type == "BepC3":
                    stage_block = BepC3
                elif stage_block_type == "MBLABlock":
                    stage_block = MBLABlock
                else:
                    raise NotImplementedError
        
                # 第一个下采样层，将通道数从1024降到512
                self.reduce_layer0 = ConvBNReLU(
                    in_channels=channels_list[5], # 1024
                    out_channels=channels_list[6], # 512
                    kernel_size=1,
                    stride=1
                )
        
                # 第一个上采样层，通道数保持512不变
                self.upsample0 = Transpose(
                    in_channels=channels_list[6],  # 512
                    out_channels=channels_list[6], # 512
                )
        
                # 第一个RepBlock，处理768通道的特征(512 + 256)
                self.Rep_p5 = stage_block(
                    in_channels=channels_list[4] + channels_list[6], # 768 + 512
                    out_channels=channels_list[6], # 512
                    n=num_repeats[6],
                    e=csp_e,
                    block=block
                )
        
                # 第二个下采样层，将通道数从512降到256
                self.reduce_layer1 = ConvBNReLU(
                    in_channels=channels_list[6],  # 512
                    out_channels=channels_list[7], # 256
                    kernel_size=1,
                    stride=1
                )
        
                # 第二个上采样层，通道数保持256不变
                self.upsample1 = Transpose(
                    in_channels=channels_list[7], # 256
                    out_channels=channels_list[7] # 256
                )
        
                # 第二个RepBlock，处理512通道的特征(256 + 256)
                self.Rep_p4 = stage_block(
                    in_channels=channels_list[3] + channels_list[7], # 512 + 256
                    out_channels=channels_list[7], # 256
                    n=num_repeats[7],
                    e=csp_e,
                    block=block
                )
        
                # 第三个下采样层，将通道数从256降到128
                self.reduce_layer2 = ConvBNReLU(
                    in_channels=channels_list[7],  # 256
                    out_channels=channels_list[8], # 128
                    kernel_size=1,
                    stride=1
                )
        
                # 第三个上采样层，通道数保持128不变
                self.upsample2 = Transpose(
                    in_channels=channels_list[8], # 128
                    out_channels=channels_list[8] # 128
                )
        
                # 第三个RepBlock，处理256通道的特征(128 + 128)
                self.Rep_p3 = stage_block(
                    in_channels=channels_list[2] + channels_list[8], # 256 + 128
                    out_channels=channels_list[8], # 128
                    n=num_repeats[8],
                    e=csp_e,
                    block=block
                )
        
                # 第一个下采样层，将通道数从128降到128
                self.downsample2 = ConvBNReLU(
                    in_channels=channels_list[8],  # 128
                    out_channels=channels_list[8], # 128
                    kernel_size=3,
                    stride=2
                )
        
                # 第四个RepBlock，处理256通道的特征(128 + 128)
                self.Rep_n4 = stage_block(
                    in_channels=channels_list[8] + channels_list[8], # 128 + 128
                    out_channels=channels_list[9], # 256
                    n=num_repeats[9],
                    e=csp_e,
                    block=block
                )
        
                # 第二个下采样层，将通道数从256降到256
                self.downsample1 = ConvBNReLU(
                    in_channels=channels_list[9],  # 256
                    out_channels=channels_list[9], # 256
                    kernel_size=3,
                    stride=2
                )
        
                # 第五个RepBlock，处理512通道的特征(256 + 256)
                self.Rep_n5 = stage_block(
                    in_channels=channels_list[7] + channels_list[9], # 256 + 256
                    out_channels=channels_list[10], # 512
                    n=num_repeats[10],
                    e=csp_e,
                    block=block
                )
        
                # 第三个下采样层，将通道数从512降到512
                self.downsample0 = ConvBNReLU(
                    in_channels=channels_list[10],  # 512
                    out_channels=channels_list[10], # 512
                    kernel_size=3,
                    stride=2
                )
        
                # 第六个RepBlock，处理1024通道的特征(512 + 512)
                self.Rep_n6 = stage_block(
                    in_channels=channels_list[6] + channels_list[10], # 512 + 512
                    out_channels=channels_list[11], # 1024
                    n=num_repeats[11],
                    e=csp_e,
                    block=block
                )
        
            def forward(self, input):
                # 解包输入的特征图元组，从小特征图到大特征图的顺序
                (x3, x2, x1, x0) = input  # x0最大，x3最小
        
                # 自顶向下路径(top-down path) - 特征图尺寸逐渐变大，通道数逐渐减小
                fpn_out0 = self.reduce_layer0(x0)  # 1024->512通道
                upsample_feat0 = self.upsample0(fpn_out0)  # 上采样
                f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)  # 特征图拼接
                f_out0 = self.Rep_p5(f_concat_layer0)  # 通过RepBlock处理
        
                fpn_out1 = self.reduce_layer1(f_out0)  # 512->256通道
                upsample_feat1 = self.upsample1(fpn_out1)  # 上采样
                f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)  # 特征图拼接
                f_out1 = self.Rep_p4(f_concat_layer1)  # 通过RepBlock处理
        
                fpn_out2 = self.reduce_layer2(f_out1)  # 256->128通道
                upsample_feat2 = self.upsample2(fpn_out2)  # 上采样
                f_concat_layer2 = torch.cat([upsample_feat2, x3], 1)  # 特征图拼接
                pan_out3 = self.Rep_p3(f_concat_layer2)  # P3输出
        
                down_feat2 = self.downsample2(pan_out3)  # 对P3下采样
                p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)  # 特征图拼接
                pan_out2 = self.Rep_n4(p_concat_layer2)  # P4输出
        
                down_feat1 = self.downsample1(pan_out2)  # 对P4下采样
                p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 特征图拼接
                pan_out1 = self.Rep_n5(p_concat_layer1)  # P5输出
        
                down_feat0 = self.downsample0(pan_out1)  # 对P5下采样
                p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)  # 特征图拼接
                pan_out0 = self.Rep_n6(p_concat_layer0)  # P6输出
        
                # 返回所有输出特征图，从小到大的顺序：[P3, P4, P5, P6]
                outputs = [pan_out3, pan_out2, pan_out1, pan_out0]
        
                return outputs


class CSPRepBiFPANNeck_P6(nn.Module):
    """CSPRepBiFPANNeck_P6 Module
    CSPRepPANNeck P6模块 - 一个改进的特征金字塔网络，用于特征融合
    """
    # [64, 128, 256, 512, 768, 1024]  # 输入通道数列表
    # [512, 256, 128, 256, 512, 1024] # 输出通道数列表

    def __init__(
        self,
        channels_list=None,  # 通道数列表
        num_repeats=None,    # 重复次数列表
        block=BottleRep,     # 基础构建块，默认使用BottleRep
        csp_e=float(1)/2,    # CSP网络的扩展因子，默认为0.5
        stage_block_type="BepC3" # 阶段块类型，可选BepC3或MBLABlock
    ):
        super().__init__()

        assert channels_list is not None  # 确保通道列表不为空
        assert num_repeats is not None    # 确保重复次数列表不为空

        # 根据stage_block_type选择不同的块类型
        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        # 第一个下采样层，将通道数从1024降到512
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        # 第一个双向融合层，融合768和512通道的特征
        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]], # 768, 512
            out_channels=channels_list[6], # 512
        )

        # 第一个RepBlock，处理512通道的特征
        self.Rep_p5 = stage_block(
            in_channels=channels_list[6], # 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        # 第二个下采样层，将通道数从512降到256
        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        # 第二个双向融合层，融合512和256通道的特征
        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]], # 512, 256
            out_channels=channels_list[7], # 256
        )

        # 第二个RepBlock，处理256通道的特征
        self.Rep_p4 = stage_block(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        # 第三个下采样层，将通道数从256降到128
        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        # 第三个双向融合层，融合256和128通道的特征
        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]], # 256, 128
            out_channels=channels_list[8], # 128
        )

        # 第三个RepBlock，处理128通道的特征
        self.Rep_p3 = stage_block(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        # 第一个下采样层，将通道数从128降到128
        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        # 第四个RepBlock，处理256通道的特征(128 + 128)
        self.Rep_n4 = stage_block(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            e=csp_e,
            block=block
        )

        # 第二个下采样层，将通道数从256降到256
        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        # 第五个RepBlock，处理512通道的特征(256 + 256)
        self.Rep_n5 = stage_block(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            e=csp_e,
            block=block
        )

        # 第三个下采样层，将通道数从512降到512
        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        # 第六个RepBlock，处理1024通道的特征(512 + 512)
        self.Rep_n6 = stage_block(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            e=csp_e,
            block=block
        )

    def forward(self, input):
        # 解包输入的特征图元组，从小特征图到大特征图的顺序
        (x4, x3, x2, x1, x0) = input  # x0最大，x4最小

        # 自顶向下路径(top-down path) - 特征图尺寸逐渐变大，通道数逐渐减小
        fpn_out0 = self.reduce_layer0(x0)  # 1024->512通道
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])  # 融合三个特征图
        f_out0 = self.Rep_p5(f_concat_layer0)  # 通过RepBlock处理

        fpn_out1 = self.reduce_layer1(f_out0)  # 512->256通道
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])  # 融合三个特征图
        f_out1 = self.Rep_p4(f_concat_layer1)  # 通过RepBlock处理

        fpn_out2 = self.reduce_layer2(f_out1)  # 256->128通道
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])  # 融合三个特征图
        pan_out3 = self.Rep_p3(f_concat_layer2)  # P3输出

        down_feat2 = self.downsample2(pan_out3)  # 对P3下采样
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)  # 特征图拼接
        pan_out2 = self.Rep_n4(p_concat_layer2)  # P4输出

        down_feat1 = self.downsample1(pan_out2)  # 对P4下采样
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)  # 特征图拼接
        pan_out1 = self.Rep_n5(p_concat_layer1)  # P5输出

        down_feat0 = self.downsample0(pan_out1)  # 对P5下采样
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)  # 特征图拼接
        pan_out0 = self.Rep_n6(p_concat_layer0)  # P6输出

        # 返回所有输出特征图，从小到大的顺序：[P3, P4, P5, P6]
        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class Lite_EffiNeck(nn.Module):

    def __init__(
        self,
        in_channels,         # 输入通道数列表
        unified_channels,    # 统一通道数
    ):
        super().__init__()
        
        # 第一个下采样层，将输入通道数转换为统一通道数
        self.reduce_layer0 = ConvBNHS(
            in_channels=in_channels[0],  # 输入通道数
            out_channels=unified_channels, # 输出统一通道数
            kernel_size=1,                 # 卷积核大小
            stride=1,                      # 步幅
            padding=0                      # 填充
        )
        
        # 第二个下采样层，将输入通道数转换为统一通道数
        self.reduce_layer1 = ConvBNHS(
            in_channels=in_channels[1],  # 输入通道数
            out_channels=unified_channels, # 输出统一通道数
            kernel_size=1,                 # 卷积核大小
            stride=1,                      # 步幅
            padding=0                      # 填充
        )
        
        # 第三个下采样层，将输入通道数转换为统一通道数
        self.reduce_layer2 = ConvBNHS(
            in_channels=in_channels[2],  # 输入通道数
            out_channels=unified_channels, # 输出统一通道数
            kernel_size=1,                 # 卷积核大小
            stride=1,                      # 步幅
            padding=0                      # 填充
        )
        
        # 第一个上采样层，将特征图尺寸扩大2倍
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')

        # 第二个上采样层，将特征图尺寸扩大2倍
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        # 第一个CSPBlock，处理统一通道数的特征
        self.Csp_p4 = CSPBlock(
            in_channels=unified_channels*2, # 输入通道数
            out_channels=unified_channels,    # 输出统一通道数
            kernel_size=5                     # 卷积核大小
        )
        
        # 第二个CSPBlock，处理统一通道数的特征
        self.Csp_p3 = CSPBlock(
            in_channels=unified_channels*2, # 输入通道数
            out_channels=unified_channels,    # 输出统一通道数
            kernel_size=5                     # 卷积核大小
        )
        
        # 第三个CSPBlock，处理统一通道数的特征
        self.Csp_n3 = CSPBlock(
            in_channels=unified_channels*2, # 输入通道数
            out_channels=unified_channels,    # 输出统一通道数
            kernel_size=5                     # 卷积核大小
        )
        
        # 第四个CSPBlock，处理统一通道数的特征
        self.Csp_n4 = CSPBlock(
            in_channels=unified_channels*2, # 输入通道数
            out_channels=unified_channels,    # 输出统一通道数
            kernel_size=5                     # 卷积核大小
        )
        
        # 第一个下采样层，将通道数从统一通道数降到统一通道数
        self.downsample2 = DPBlock(
            in_channel=unified_channels,     # 输入通道数
            out_channel=unified_channels,     # 输出统一通道数
            kernel_size=5,                   # 卷积核大小
            stride=2                         # 步幅
        )
        
        # 第二个下采样层，将通道数从统一通道数降到统一通道数
        self.downsample1 = DPBlock(
            in_channel=unified_channels,     # 输入通道数
            out_channel=unified_channels,     # 输出统一通道数
            kernel_size=5,                   # 卷积核大小
            stride=2                         # 步幅
        )
        
        # 第三个下采样层，将通道数从统一通道数降到统一通道数
        self.p6_conv_1 = DPBlock(
            in_channel=unified_channels,     # 输入通道数
            out_channel=unified_channels,     # 输出统一通道数
            kernel_size=5,                   # 卷积核大小
            stride=2                         # 步幅
        )
        
        # 第四个下采样层，将通道数从统一通道数降到统一通道数
        self.p6_conv_2 = DPBlock(
            in_channel=unified_channels,     # 输入通道数
            out_channel=unified_channels,     # 输出统一通道数
            kernel_size=5,                   # 卷积核大小
            stride=2                         # 步幅
        )

    def forward(self, input):
        # 解包输入的特征图元组，从小特征图到大特征图的顺序
        (x2, x1, x0) = input  # x0最大，x2最小

        # 通过下采样层处理输入特征图
        fpn_out0 = self.reduce_layer0(x0) # c5
        x1 = self.reduce_layer1(x1)       # c4
        x2 = self.reduce_layer2(x2)       # c3

        # 对fpn_out0进行上采样
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)  # 特征图拼接
        f_out1 = self.Csp_p4(f_concat_layer0)  # 通过CSPBlock处理

        # 对f_out1进行上采样
        upsample_feat1 = self.upsample1(f_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)  # 特征图拼接
        pan_out3 = self.Csp_p3(f_concat_layer1) # p3

        # 对pan_out3进行下采样
        down_feat1 = self.downsample2(pan_out3)
        p_concat_layer1 = torch.cat([down_feat1, f_out1], 1)  # 特征图拼接
        pan_out2 = self.Csp_n3(p_concat_layer1)  # p4

        # 对pan_out2进行下采样
        down_feat0 = self.downsample1(pan_out2)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)  # 特征图拼接
        pan_out1 = self.Csp_n4(p_concat_layer2)  # p5

        # 对pan_out1进行下采样
        top_features = self.p6_conv_1(fpn_out0)
        pan_out0 = top_features + self.p6_conv_2(pan_out1)  # p6

        # 返回所有输出特征图，从小到大的顺序：[P3, P4, P5, P6]
        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs
