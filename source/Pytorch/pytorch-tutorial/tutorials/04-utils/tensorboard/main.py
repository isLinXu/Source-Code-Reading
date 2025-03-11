import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torchvision  # 导入 torchvision 库
from torchvision import transforms  # 从 torchvision 导入数据转换模块
from logger import Logger  # 从 logger 模块导入 Logger 类


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 配置设备，使用 GPU（如果可用）或 CPU

# MNIST dataset 
dataset = torchvision.datasets.MNIST(root='../../data',  # 下载 MNIST 数据集
                                     train=True,  # 使用训练集
                                     transform=transforms.ToTensor(),  # 将图像转换为张量
                                     download=True)  # 如果未下载则下载数据集

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,  # 创建数据加载器
                                          batch_size=100,  # 设置批次大小
                                          shuffle=True)  # 打乱数据


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):  # 定义全连接神经网络类
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):  # 初始化方法
        super(NeuralNet, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层全连接层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 第二层全连接层（输出层）
    
    def forward(self, x):  # 前向传播方法
        out = self.fc1(x)  # 通过第一层
        out = self.relu(out)  # 应用 ReLU 激活函数
        out = self.fc2(out)  # 通过第二层
        return out  # 返回输出

model = NeuralNet().to(device)  # 实例化神经网络模型并移动到指定设备

logger = Logger('./logs')  # 创建 Logger 实例，指定日志目录

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 定义优化器为 Adam，设置学习率

data_iter = iter(data_loader)  # 创建数据迭代器
iter_per_epoch = len(data_loader)  # 每个周期的迭代次数
total_step = 50000  # 总训练步数

# Start training
for step in range(total_step):  # 遍历每个训练步骤
    
    # Reset the data_iter
    if (step + 1) % iter_per_epoch == 0:  # 如果到达一个周期的末尾
        data_iter = iter(data_loader)  # 重置数据迭代器

    # Fetch images and labels
    images, labels = next(data_iter)  # 获取下一个批次的图像和标签
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)  # 展平图像并移动到设备
    
    # Forward pass
    outputs = model(images)  # 前向传播，获取模型输出
    loss = criterion(outputs, labels)  # 计算损失
    
    # Backward and optimize
    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)  # 获取输出中最大值的索引
    accuracy = (labels == argmax.squeeze()).float().mean()  # 计算准确率

    if (step + 1) % 100 == 0:  # 每 100 步打印一次信息
        print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'  # 打印当前步数、损失和准确率
              .format(step + 1, total_step, loss.item(), accuracy.item()))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss.item(), 'accuracy': accuracy.item()}  # 创建信息字典

        for tag, value in info.items():  # 遍历信息字典
            logger.scalar_summary(tag, value, step + 1)  # 记录标量值

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():  # 遍历模型的命名参数
            tag = tag.replace('.', '/')  # 替换参数名称中的点
            logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)  # 记录参数值的直方图
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)  # 记录参数梯度的直方图

        # 3. Log training images (image summary)
        info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}  # 获取前 10 张图像

        for tag, images in info.items():  # 遍历信息字典
            logger.image_summary(tag, images, step + 1)  # 记录图像摘要