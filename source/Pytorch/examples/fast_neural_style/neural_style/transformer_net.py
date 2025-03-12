import torch  # 导入PyTorch库


class TransformerNet(torch.nn.Module):  # 定义TransformerNet类，继承自torch.nn.Module
    def __init__(self):  # 初始化函数
        super(TransformerNet, self).__init__()  # 调用父类构造函数
        # Initial convolution layers  # 初始卷积层
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)  # 创建第一个卷积层，输入通道3，输出通道32，卷积核大小9，步幅1
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)  # 创建实例归一化层，处理32个通道
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)  # 创建第二个卷积层，输入通道32，输出通道64，卷积核大小3，步幅2
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)  # 创建实例归一化层，处理64个通道
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)  # 创建第三个卷积层，输入通道64，输出通道128，卷积核大小3，步幅2
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)  # 创建实例归一化层，处理128个通道
        # Residual layers  # 残差层
        self.res1 = ResidualBlock(128)  # 创建第一个残差块，输入通道128
        self.res2 = ResidualBlock(128)  # 创建第二个残差块，输入通道128
        self.res3 = ResidualBlock(128)  # 创建第三个残差块，输入通道128
        self.res4 = ResidualBlock(128)  # 创建第四个残差块，输入通道128
        self.res5 = ResidualBlock(128)  # 创建第五个残差块，输入通道128
        # Upsampling Layers  # 上采样层
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)  # 创建第一个上采样卷积层，输入通道128，输出通道64，卷积核大小3，步幅1，上采样因子2
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)  # 创建实例归一化层，处理64个通道
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)  # 创建第二个上采样卷积层，输入通道64，输出通道32，卷积核大小3，步幅1，上采样因子2
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)  # 创建实例归一化层，处理32个通道
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)  # 创建第三个卷积层，输入通道32，输出通道3，卷积核大小9，步幅1
        # Non-linearities  # 非线性激活函数
        self.relu = torch.nn.ReLU()  # 创建ReLU激活函数

    def forward(self, X):  # 前向传播函数
        y = self.relu(self.in1(self.conv1(X)))  # 通过第一个卷积层和实例归一化层，应用ReLU激活
        y = self.relu(self.in2(self.conv2(y)))  # 通过第二个卷积层和实例归一化层，应用ReLU激活
        y = self.relu(self.in3(self.conv3(y)))  # 通过第三个卷积层和实例归一化层，应用ReLU激活
        y = self.res1(y)  # 通过第一个残差块
        y = self.res2(y)  # 通过第二个残差块
        y = self.res3(y)  # 通过第三个残差块
        y = self.res4(y)  # 通过第四个残差块
        y = self.res5(y)  # 通过第五个残差块
        y = self.relu(self.in4(self.deconv1(y)))  # 通过第一个上采样卷积层和实例归一化层，应用ReLU激活
        y = self.relu(self.in5(self.deconv2(y)))  # 通过第二个上采样卷积层和实例归一化层，应用ReLU激活
        y = self.deconv3(y)  # 通过第三个卷积层
        return y  # 返回输出


class ConvLayer(torch.nn.Module):  # 定义卷积层类
    def __init__(self, in_channels, out_channels, kernel_size, stride):  # 初始化函数
        super(ConvLayer, self).__init__()  # 调用父类构造函数
        reflection_padding = kernel_size // 2  # 计算反射填充大小
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)  # 创建反射填充层
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # 创建卷积层

    def forward(self, x):  # 前向传播函数
        out = self.reflection_pad(x)  # 应用反射填充
        out = self.conv2d(out)  # 通过卷积层
        return out  # 返回输出


class ResidualBlock(torch.nn.Module):  # 定义残差块类
    """ResidualBlock  # 残差块
    introduced in: https://arxiv.org/abs/1512.03385  # 引入于此文献
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html  # 推荐架构
    """

    def __init__(self, channels):  # 初始化函数
        super(ResidualBlock, self).__init__()  # 调用父类构造函数
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)  # 创建第一个卷积层
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)  # 创建实例归一化层
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)  # 创建第二个卷积层
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)  # 创建实例归一化层
        self.relu = torch.nn.ReLU()  # 创建ReLU激活函数

    def forward(self, x):  # 前向传播函数
        residual = x  # 保存输入作为残差
        out = self.relu(self.in1(self.conv1(x)))  # 通过第一个卷积层和实例归一化层，应用ReLU激活
        out = self.in2(self.conv2(out))  # 通过第二个卷积层和实例归一化层
        out = out + residual  # 将残差添加到输出
        return out  # 返回输出


class UpsampleConvLayer(torch.nn.Module):  # 定义上采样卷积层类
    """UpsampleConvLayer  # 上采样卷积层
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.  # 这种方法相比于ConvTranspose2d效果更好
    ref: http://distill.pub/2016/deconv-checkerboard/  # 参考文献
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):  # 初始化函数
        super(UpsampleConvLayer, self).__init__()  # 调用父类构造函数
        self.upsample = upsample  # 保存上采样因子
        reflection_padding = kernel_size // 2  # 计算反射填充大小
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)  # 创建反射填充层
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # 创建卷积层

    def forward(self, x):  # 前向传播函数
        x_in = x  # 保存输入
        if self.upsample:  # 如果设置了上采样
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)  # 进行上采样
        out = self.reflection_pad(x_in)  # 应用反射填充
        out = self.conv2d(out)  # 通过卷积层
        return out  # 返回输出