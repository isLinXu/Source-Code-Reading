import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torchvision  # 导入 torchvision 库
import torchvision.transforms as transforms  # 导入 torchvision 的数据转换模块


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备配置：如果有可用的 GPU，则使用 GPU，否则使用 CPU

# Hyper-parameters
sequence_length = 28  # 序列的长度
input_size = 28  # 输入的大小
hidden_size = 128  # 隐藏层的大小
num_layers = 2  # LSTM 的层数
num_classes = 10  # 类别数量
batch_size = 100  # 每个批次的样本数量
num_epochs = 2  # 训练的轮数
learning_rate = 0.01  # 学习率

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',  # MNIST 数据集的根目录
                                           train=True,  # 训练集
                                           transform=transforms.ToTensor(),  # 数据转换为 Tensor
                                           download=True)  # 如果数据集不存在，则下载

test_dataset = torchvision.datasets.MNIST(root='../../data/',  # MNIST 数据集的根目录
                                          train=False,  # 测试集
                                          transform=transforms.ToTensor())  # 数据转换为 Tensor

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # 训练集的数据加载器
                                           batch_size=batch_size,  # 每个批次的样本数量
                                           shuffle=True)  # 打乱数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  # 测试集的数据加载器
                                          batch_size=batch_size,  # 每个批次的样本数量
                                          shuffle=False)  # 不打乱数据

# Recurrent neural network (many-to-one)
class RNN(nn.Module):  # 定义 RNN 类
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  # 初始化方法
        super(RNN, self).__init__()  # 调用父类的初始化方法
        self.hidden_size = hidden_size  # 隐藏层的大小
        self.num_layers = num_layers  # LSTM 的层数
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM 层
        self.fc = nn.Linear(hidden_size, num_classes)  # 全连接层，输出特征为类别数量
    
    def forward(self, x):  # 前向传播方法
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 初始化细胞状态
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # 前向传播 LSTM，返回输出和隐藏状态
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # 通过全连接层解码最后一个时间步的隐藏状态
        return out  # 返回输出

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)  # 创建 RNN 模型并转移到指定设备


# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器为 Adam

# Train the model
total_step = len(train_loader)  # 计算训练集的总步数
for epoch in range(num_epochs):  # 遍历每个训练轮
    for i, (images, labels) in enumerate(train_loader):  # 遍历训练数据
        images = images.reshape(-1, sequence_length, input_size).to(device)  # 将图像重塑为 (batch_size, sequence_length, input_size)
        labels = labels.to(device)  # 将标签转移到设备
        
        # Forward pass
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        
        # Backward and optimize
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        if (i+1) % 100 == 0:  # 每 100 步打印一次信息
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'  # 打印当前轮次、步骤和损失值
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))  # 打印当前轮次、步骤和损失值

# Test the model
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在测试阶段，不需要计算梯度（节省内存）
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量
    for images, labels in test_loader:  # 遍历测试数据
        images = images.reshape(-1, sequence_length, input_size).to(device)  # 将图像重塑为 (batch_size, sequence_length, input_size)
        labels = labels.to(device)  # 将标签转移到设备
        outputs = model(images)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 更新总样本数量
        correct += (predicted == labels).sum().item()  # 更新正确预测的数量

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))  # 打印测试集上的准确率

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')  # 保存模型的状态字典