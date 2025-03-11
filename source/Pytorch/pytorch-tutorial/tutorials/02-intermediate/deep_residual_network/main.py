# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #  # 实现自 https://arxiv.org/pdf/1512.03385.pdf
# See section 4.2 for the model architecture on CIFAR-10                       #  # 请参见第 4.2 节了解 CIFAR-10 的模型架构
# Some part of the code was referenced from below                              #  # 部分代码参考自以下内容
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# ---------------------------------------------------------------------------- #

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torchvision  # 导入 torchvision 库
import torchvision.transforms as transforms  # 导入 torchvision 的数据转换模块


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备配置：如果有可用的 GPU，则使用 GPU，否则使用 CPU

# Hyper-parameters
num_epochs = 80  # 训练的轮数
batch_size = 100  # 每个批次的样本数量
learning_rate = 0.001  # 学习率

# Image preprocessing modules
transform = transforms.Compose([  # 图像预处理模块
    transforms.Pad(4),  # 在图像周围填充 4 像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.RandomCrop(32),  # 随机裁剪图像到 32x32
    transforms.ToTensor()])  # 将图像转换为 Tensor

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',  # CIFAR-10 数据集的根目录
                                             train=True,  # 训练集
                                             transform=transform,  # 应用图像预处理
                                             download=True)  # 如果数据集不存在，则下载

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',  # CIFAR-10 数据集的根目录
                                            train=False,  # 测试集
                                            transform=transforms.ToTensor())  # 数据转换为 Tensor

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # 训练集的数据加载器
                                           batch_size=batch_size,  # 每个批次的样本数量
                                           shuffle=True)  # 打乱数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  # 测试集的数据加载器
                                          batch_size=batch_size,  # 每个批次的样本数量
                                          shuffle=False)  # 不打乱数据

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):  # 定义 3x3 卷积函数
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,  # 创建卷积层
                     stride=stride, padding=1, bias=False)  # 步幅，填充和是否使用偏置

# Residual block
class ResidualBlock(nn.Module):  # 定义残差块类
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):  # 初始化方法
        super(ResidualBlock, self).__init__()  # 调用父类的初始化方法
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活函数
        self.conv2 = conv3x3(out_channels, out_channels)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批归一化层
        self.downsample = downsample  # 下采样层
        
    def forward(self, x):  # 前向传播方法
        residual = x  # 保存输入以便后续加法
        out = self.conv1(x)  # 输入通过第一个卷积层
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # 应用 ReLU 激活函数
        out = self.conv2(out)  # 输入通过第二个卷积层
        out = self.bn2(out)  # 批归一化
        if self.downsample:  # 如果存在下采样
            residual = self.downsample(x)  # 对输入进行下采样
        out += residual  # 将残差添加到输出
        out = self.relu(out)  # 应用 ReLU 激活函数
        return out  # 返回输出

# ResNet
class ResNet(nn.Module):  # 定义 ResNet 类
    def __init__(self, block, layers, num_classes=10):  # 初始化方法
        super(ResNet, self).__init__()  # 调用父类的初始化方法
        self.in_channels = 16  # 初始输入通道数
        self.conv = conv3x3(3, 16)  # 初始卷积层
        self.bn = nn.BatchNorm2d(16)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活函数
        self.layer1 = self.make_layer(block, 16, layers[0])  # 创建第一层
        self.layer2 = self.make_layer(block, 32, layers[1], 2)  # 创建第二层
        self.layer3 = self.make_layer(block, 64, layers[2], 2)  # 创建第三层
        self.avg_pool = nn.AvgPool2d(8)  # 平均池化层
        self.fc = nn.Linear(64, num_classes)  # 全连接层，输出特征为类别数量
        
    def make_layer(self, block, out_channels, blocks, stride=1):  # 创建层的方法
        downsample = None  # 初始化下采样
        if (stride != 1) or (self.in_channels != out_channels):  # 如果步幅不为 1 或输入通道数不等于输出通道数
            downsample = nn.Sequential(  # 创建下采样层
                conv3x3(self.in_channels, out_channels, stride=stride),  # 卷积层
                nn.BatchNorm2d(out_channels))  # 批归一化层
        layers = []  # 初始化层列表
        layers.append(block(self.in_channels, out_channels, stride, downsample))  # 添加第一个残差块
        self.in_channels = out_channels  # 更新输入通道数
        for i in range(1, blocks):  # 添加剩余的残差块
            layers.append(block(out_channels, out_channels))  # 添加残差块
        return nn.Sequential(*layers)  # 返回层的序列
    
    def forward(self, x):  # 前向传播方法
        out = self.conv(x)  # 输入通过初始卷积层
        out = self.bn(out)  # 批归一化
        out = self.relu(out)  # 应用 ReLU 激活函数
        out = self.layer1(out)  # 输入通过第一层
        out = self.layer2(out)  # 输入通过第二层
        out = self.layer3(out)  # 输入通过第三层
        out = self.avg_pool(out)  # 平均池化
        out = out.view(out.size(0), -1)  # 展平输出
        out = self.fc(out)  # 输入通过全连接层
        return out  # 返回输出
    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)  # 创建 ResNet 模型并转移到指定设备


# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器为 Adam

# For updating learning rate
def update_lr(optimizer, lr):  # 更新学习率的函数
    for param_group in optimizer.param_groups:  # 遍历优化器的参数组
        param_group['lr'] = lr  # 更新学习率

# Train the model
total_step = len(train_loader)  # 训练集的总步数
curr_lr = learning_rate  # 当前学习率
for epoch in range(num_epochs):  # 遍历每个训练轮
    for i, (images, labels) in enumerate(train_loader):  # 遍历训练数据
        images = images.to(device)  # 将图像转移到设备
        labels = labels.to(device)  # 将标签转移到设备
        
        # Forward pass
        outputs = model(images)  # 前向传播，获取输出
        loss = criterion(outputs, labels)  # 计算损失
        
        # Backward and optimize
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        if (i+1) % 100 == 0:  # 每 100 个步骤打印一次信息
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))  # 打印当前轮次、步骤和损失值

    # Decay learning rate
    if (epoch+1) % 20 == 0:  # 每 20 个轮次衰减学习率
        curr_lr /= 3  # 将学习率除以 3
        update_lr(optimizer, curr_lr)  # 更新学习率

# Test the model
model.eval()  # 评估模式
with torch.no_grad():  # 在测试阶段，不需要计算梯度（节省内存）
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量
    for images, labels in test_loader:  # 遍历测试数据
        images = images.to(device)  # 将图像转移到设备
        labels = labels.to(device)  # 将标签转移到设备
        outputs = model(images)  # 前向传播，获取输出
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 更新总样本数量
        correct += (predicted == labels).sum().item()  # 更新正确预测的数量

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))  # 打印测试集上的准确率

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')  # 保存模型的状态字典