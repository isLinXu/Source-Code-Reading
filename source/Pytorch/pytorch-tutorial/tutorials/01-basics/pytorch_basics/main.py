import torch  # 导入 PyTorch 库
import torchvision  # 导入 torchvision 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库
import torchvision.transforms as transforms  # 导入 torchvision 的数据转换模块


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)  # 基本自动求导示例 1
# 2. Basic autograd example 2               (Line 46 to 83)  # 基本自动求导示例 2
# 3. Loading data from numpy                (Line 90 to 97)  # 从 NumPy 加载数据
# 4. Input pipeline                          (Line 104 to 129)  # 输入管道
# 5. Input pipeline for custom dataset       (Line 136 to 156)  # 自定义数据集的输入管道
# 6. Pretrained model                       (Line 163 to 176)  # 预训练模型
# 7. Save and load model                    (Line 183 to 189)  # 保存和加载模型


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors.
x = torch.tensor(1., requires_grad=True)  # 创建张量 x，设置 requires_grad=True 以计算梯度
w = torch.tensor(2., requires_grad=True)  # 创建张量 w，设置 requires_grad=True 以计算梯度
b = torch.tensor(3., requires_grad=True)  # 创建张量 b，设置 requires_grad=True 以计算梯度

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3  # 构建计算图，计算 y

# Compute gradients.
y.backward()  # 计算梯度

# Print out the gradients.
print(x.grad)    # x.grad = 2  # 打印 x 的梯度
print(w.grad)    # w.grad = 1  # 打印 w 的梯度
print(b.grad)    # b.grad = 1  # 打印 b 的梯度


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)  # 创建形状为 (10, 3) 的随机张量 x
y = torch.randn(10, 2)  # 创建形状为 (10, 2) 的随机张量 y

# Build a fully connected layer.
linear = nn.Linear(3, 2)  # 创建一个全连接层，输入特征为 3，输出特征为 2
print ('w: ', linear.weight)  # 打印权重
print ('b: ', linear.bias)  # 打印偏置

# Build loss function and optimizer.
criterion = nn.MSELoss()  # 定义损失函数为均方误差损失
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)  # 定义优化器为随机梯度下降（SGD）

# Forward pass.
pred = linear(x)  # 前向传播，获取预测结果

# Compute loss.
loss = criterion(pred, y)  # 计算损失
print('loss: ', loss.item())  # 打印损失值

# Backward pass.
loss.backward()  # 反向传播

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad)  # 打印权重的梯度
print ('dL/db: ', linear.bias.grad)  # 打印偏置的梯度

# 1-step gradient descent.
optimizer.step()  # 执行一步梯度下降

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)  # 低级别的梯度下降
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)  # 低级别的梯度下降

# Print out the loss after 1-step gradient descent.
pred = linear(x)  # 再次前向传播，获取新的预测结果
loss = criterion(pred, y)  # 计算新的损失
print('loss after 1 step optimization: ', loss.item())  # 打印经过一步优化后的损失


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])  # 创建一个 NumPy 数组

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)  # 将 NumPy 数组转换为 PyTorch 张量

# Convert the torch tensor to a numpy array.
z = y.numpy()  # 将 PyTorch 张量转换为 NumPy 数组


# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',  # 下载并构建 CIFAR-10 数据集
                                             train=True,  # 训练集
                                             transform=transforms.ToTensor(),  # 数据转换为 Tensor
                                             download=True)  # 如果数据集不存在，则下载

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]  # 获取一对数据（从磁盘读取）
print (image.size())  # 打印图像的大小
print (label)  # 打印标签

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # 训练集的数据加载器
                                           batch_size=64,  # 每个批次的样本数量
                                           shuffle=True)  # 打乱数据

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)  # 创建数据迭代器

# Mini-batch images and labels.
images, labels = data_iter.next()  # 获取一个小批量的图像和标签

# Actual usage of the data loader is as below.
for images, labels in train_loader:  # 遍历数据加载器
    # Training code should be written here.
    pass  # 训练代码应在此处编写


# ================================================================== #
#                5. Input pipeline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):  # 定义自定义数据集类
    def __init__(self):  # 初始化方法
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass  # TODO: 初始化文件路径或文件名列表

    def __getitem__(self, index):  # 获取数据项的方法
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass  # TODO: 从文件中读取数据，进行预处理，并返回数据对（例如图像和标签）

    def __len__(self):  # 获取数据集大小的方法
        # You should change 0 to the total size of your dataset.
        return 0  # 返回数据集的总大小，当前为 0


# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()  # 创建自定义数据集实例
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,  # 使用预构建的数据加载器
                                           batch_size=64,  # 每个批次的样本数量
                                           shuffle=True)  # 打乱数据


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)  # 下载并加载预训练的 ResNet-18 模型

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():  # 遍历模型的所有参数
    param.requires_grad = False  # 将 requires_grad 设置为 False，以冻结参数

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 替换顶层以进行微调，100 是一个示例

# Forward pass.
images = torch.randn(64, 3, 224, 224)  # 创建随机输入图像
outputs = resnet(images)  # 前向传播，获取输出
print (outputs.size())     # (64, 100)  # 打印输出的大小


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')  # 保存整个模型
model = torch.load('model.ckpt')  # 加载模型

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')  # 保存模型参数
resnet.load_state_dict(torch.load('params.ckpt'))  # 加载模型参数