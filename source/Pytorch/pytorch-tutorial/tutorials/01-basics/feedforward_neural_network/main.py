import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torchvision  # 导入 torchvision 库
import torchvision.transforms as transforms  # 导入 torchvision 的数据转换模块

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备配置：如果有可用的 GPU，则使用 GPU，否则使用 CPU

# Hyper-parameters 
input_size = 784  # 输入层大小（28*28 像素图像展平为 784）
hidden_size = 500  # 隐藏层大小
num_classes = 10  # 类别数量（手写数字 0-9）
num_epochs = 5  # 训练的轮数
batch_size = 100  # 每个批次的样本数量
learning_rate = 0.001  # 学习率

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data',  # MNIST 数据集的根目录
                                           train=True,  # 训练集
                                           transform=transforms.ToTensor(),  # 数据转换为 Tensor
                                           download=True)  # 如果数据集不存在，则下载

test_dataset = torchvision.datasets.MNIST(root='../../data',  # MNIST 数据集的根目录
                                          train=False,  # 测试集
                                          transform=transforms.ToTensor())  # 数据转换为 Tensor

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # 训练集的数据加载器
                                           batch_size=batch_size,  # 每个批次的样本数量
                                           shuffle=True)  # 打乱数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  # 测试集的数据加载器
                                          batch_size=batch_size,  # 每个批次的样本数量
                                          shuffle=False)  # 不打乱数据

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):  # 定义一个神经网络类，继承自 nn.Module
    def __init__(self, input_size, hidden_size, num_classes):  # 初始化方法
        super(NeuralNet, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层全连接层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 第二层全连接层
    
    def forward(self, x):  # 前向传播方法
        out = self.fc1(x)  # 输入通过第一层
        out = self.relu(out)  # 应用 ReLU 激活函数
        out = self.fc2(out)  # 输入通过第二层
        return out  # 返回输出

model = NeuralNet(input_size, hidden_size, num_classes).to(device)  # 创建神经网络模型并转移到指定设备

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器为 Adam

# Train the model
total_step = len(train_loader)  # 训练集的总步数
for epoch in range(num_epochs):  # 遍历每个训练轮
    for i, (images, labels) in enumerate(train_loader):  # 遍历训练数据
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)  # 将图像展平并转移到设备
        labels = labels.to(device)  # 将标签转移到设备
        
        # Forward pass
        outputs = model(images)  # 前向传播，获取输出
        loss = criterion(outputs, labels)  # 计算损失
        
        # Backward and optimize
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        if (i+1) % 100 == 0:  # 每 100 个步骤打印一次信息
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))  # 打印当前轮次、步骤和损失值

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():  # 在测试阶段，不需要计算梯度（节省内存）
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量
    for images, labels in test_loader:  # 遍历测试数据
        images = images.reshape(-1, 28*28).to(device)  # 将图像展平并转移到设备
        labels = labels.to(device)  # 将标签转移到设备
        outputs = model(images)  # 前向传播，获取输出
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 更新总样本数量
        correct += (predicted == labels).sum().item()  # 更新正确预测的数量

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))  # 打印测试集上的准确率

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')  # 保存模型的状态字典