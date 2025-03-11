import os  # 导入操作系统库
import torch  # 导入 PyTorch 库
import torchvision  # 导入 torchvision 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torchvision import transforms  # 从 torchvision 导入数据转换模块
from torchvision.utils import save_image  # 从 torchvision 导入保存图像的工具


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备配置：如果有可用的 GPU，则使用 GPU，否则使用 CPU

# Hyper-parameters
latent_size = 64  # 潜在空间的大小
hidden_size = 256  # 隐藏层的大小
image_size = 784  # 图像的大小（28x28 = 784）
num_epochs = 200  # 训练的轮数
batch_size = 100  # 每个批次的样本数量
sample_dir = 'samples'  # 样本保存的目录

# Create a directory if not exists
if not os.path.exists(sample_dir):  # 如果样本目录不存在
    os.makedirs(sample_dir)  # 创建样本目录

# Image processing
# transform = transforms.Compose([  # 图像处理
#                 transforms.ToTensor(),  # 转换为 Tensor
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])  # 归一化
transform = transforms.Compose([  # 图像处理
                transforms.ToTensor(),  # 转换为 Tensor
                transforms.Normalize(mean=[0.5],   # 1 for greyscale channels
                                     std=[0.5])])  # 归一化

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../../data/',  # MNIST 数据集的根目录
                                   train=True,  # 训练集
                                   transform=transform,  # 应用图像处理
                                   download=True)  # 如果数据集不存在，则下载

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,  # 数据加载器
                                          batch_size=batch_size,  # 每个批次的样本数量
                                          shuffle=True)  # 打乱数据

# Discriminator
D = nn.Sequential(  # 定义判别器
    nn.Linear(image_size, hidden_size),  # 输入层到隐藏层的线性变换
    nn.LeakyReLU(0.2),  # Leaky ReLU 激活函数
    nn.Linear(hidden_size, hidden_size),  # 隐藏层到隐藏层的线性变换
    nn.LeakyReLU(0.2),  # Leaky ReLU 激活函数
    nn.Linear(hidden_size, 1),  # 隐藏层到输出层的线性变换
    nn.Sigmoid())  # Sigmoid 激活函数，输出为概率

# Generator 
G = nn.Sequential(  # 定义生成器
    nn.Linear(latent_size, hidden_size),  # 潜在空间到隐藏层的线性变换
    nn.ReLU(),  # ReLU 激活函数
    nn.Linear(hidden_size, hidden_size),  # 隐藏层到隐藏层的线性变换
    nn.ReLU(),  # ReLU 激活函数
    nn.Linear(hidden_size, image_size),  # 隐藏层到输出层的线性变换
    nn.Tanh())  # Tanh 激活函数，输出范围在 -1 到 1 之间

# Device setting
D = D.to(device)  # 将判别器转移到指定设备
G = G.to(device)  # 将生成器转移到指定设备

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()  # 定义二元交叉熵损失
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)  # 定义判别器的优化器
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)  # 定义生成器的优化器

def denorm(x):  # 定义去归一化的方法
    out = (x + 1) / 2  # 将图像从 [-1, 1] 范围转换到 [0, 1] 范围
    return out.clamp(0, 1)  # 限制范围在 [0, 1] 之间

def reset_grad():  # 定义重置梯度的方法
    d_optimizer.zero_grad()  # 清空判别器的梯度
    g_optimizer.zero_grad()  # 清空生成器的梯度

# Start training
total_step = len(data_loader)  # 计算训练集的总步数
for epoch in range(num_epochs):  # 遍历每个训练轮
    for i, (images, _) in enumerate(data_loader):  # 遍历训练数据
        images = images.reshape(batch_size, -1).to(device)  # 将图像重塑为 (batch_size, image_size)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)  # 真实标签
        fake_labels = torch.zeros(batch_size, 1).to(device)  # 假标签

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)  # 判别器输出
        d_loss_real = criterion(outputs, real_labels)  # 计算真实图像的损失
        real_score = outputs  # 真实图像的得分
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)  # 随机生成潜在空间的输入
        fake_images = G(z)  # 生成假图像
        outputs = D(fake_images)  # 判别器输出
        d_loss_fake = criterion(outputs, fake_labels)  # 计算假图像的损失
        fake_score = outputs  # 假图像的得分
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake  # 总损失
        reset_grad()  # 重置梯度
        d_loss.backward()  # 反向传播
        d_optimizer.step()  # 更新判别器参数
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)  # 随机生成潜在空间的输入
        fake_images = G(z)  # 生成假图像
        outputs = D(fake_images)  # 判别器输出
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)  # 计算生成器的损失
        
        # Backprop and optimize
        reset_grad()  # 重置梯度
        g_loss.backward()  # 反向传播
        g_optimizer.step()  # 更新生成器参数
        
        if (i+1) % 200 == 0:  # 每 200 步打印一次信息
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))  # 打印损失和得分
    
    # Save real images
    if (epoch+1) == 1:  # 仅在第一次训练时保存真实图像
        images = images.reshape(images.size(0), 1, 28, 28)  # 重塑图像
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))  # 保存真实图像
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)  # 重塑假图像
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))  # 保存假图像

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')  # 保存生成器的状态字典
torch.save(D.state_dict(), 'D.ckpt')  # 保存判别器的状态字典