import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.init as init  # 导入权重初始化模块


class Net(nn.Module):
    """Define the neural network architecture.
    定义神经网络架构。"""

    def __init__(self, upscale_factor):
        """Initialize the network.
        初始化网络。

        Args:
            upscale_factor: 放大因子
        """
        super(Net, self).__init__()  # 调用父类构造函数

        self.relu = nn.ReLU()  # ReLU 激活函数
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))  # 第一层卷积，输入通道为 1，输出通道为 64，卷积核大小为 5x5，步幅为 1，填充为 2
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))  # 第二层卷积，输入通道为 64，输出通道为 64，卷积核大小为 3x3
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))  # 第三层卷积，输入通道为 64，输出通道为 32，卷积核大小为 3x3
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))  # 第四层卷积，输入通道为 32，输出通道为 upscale_factor^2，卷积核大小为 3x3
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 像素重排层，用于上采样

        self._initialize_weights()  # 初始化权重

    def forward(self, x):
        """Forward pass through the network.
        网络的前向传播。

        Args:
            x: 输入张量
        Returns:
            x: 输出张量
        """
        x = self.relu(self.conv1(x))  # 通过第一层卷积和 ReLU 激活
        x = self.relu(self.conv2(x))  # 通过第二层卷积和 ReLU 激活
        x = self.relu(self.conv3(x))  # 通过第三层卷积和 ReLU 激活
        x = self.pixel_shuffle(self.conv4(x))  # 通过第四层卷积并进行像素重排
        return x  # 返回输出张量

    def _initialize_weights(self):
        """Initialize the weights of the network.
        初始化网络的权重。"""
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))  # 使用正交初始化为卷积层 1 的权重
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))  # 使用正交初始化为卷积层 2 的权重
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))  # 使用正交初始化为卷积层 3 的权重
        init.orthogonal_(self.conv4.weight)  # 使用正交初始化为卷积层 4 的权重