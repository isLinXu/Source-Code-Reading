import os  # 导入操作系统库
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
import torchvision  # 导入 torchvision 库
from torchvision import transforms  # 从 torchvision 导入数据转换模块
from torchvision.utils import save_image  # 从 torchvision 导入保存图像的工具


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 配置设备，使用 GPU（如果可用）或 CPU

# Create a directory if not exists
sample_dir = 'samples'  # 定义样本保存目录
if not os.path.exists(sample_dir):  # 如果样本目录不存在
    os.makedirs(sample_dir)  # 创建样本目录

# Hyper-parameters
image_size = 784  # 图像大小（28x28 像素展平为 784）
h_dim = 400  # 隐藏层维度
z_dim = 20  # 潜在空间维度
num_epochs = 15  # 训练周期数
batch_size = 128  # 批次大小
learning_rate = 1e-3  # 学习率

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',  # 下载 MNIST 数据集
                                     train=True,  # 使用训练集
                                     transform=transforms.ToTensor(),  # 将图像转换为张量
                                     download=True)  # 如果未下载则下载数据集

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,  # 创建数据加载器
                                          batch_size=batch_size,  # 设置批次大小
                                          shuffle=True)  # 打乱数据


# VAE model
class VAE(nn.Module):  # 定义变分自编码器类
    def __init__(self, image_size=784, h_dim=400, z_dim=20):  # 初始化方法
        super(VAE, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(image_size, h_dim)  # 第一层全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # 第二层全连接层（用于生成潜在变量的均值）
        self.fc3 = nn.Linear(h_dim, z_dim)  # 第三层全连接层（用于生成潜在变量的对数方差）
        self.fc4 = nn.Linear(z_dim, h_dim)  # 第四层全连接层（解码器的一部分）
        self.fc5 = nn.Linear(h_dim, image_size)  # 第五层全连接层（输出层）
        
    def encode(self, x):  # 编码方法
        h = F.relu(self.fc1(x))  # 通过第一层并应用 ReLU 激活函数
        return self.fc2(h), self.fc3(h)  # 返回均值和对数方差
    
    def reparameterize(self, mu, log_var):  # 重参数化方法
        std = torch.exp(log_var / 2)  # 计算标准差
        eps = torch.randn_like(std)  # 生成与标准差相同形状的随机噪声
        return mu + eps * std  # 返回重参数化的潜在变量

    def decode(self, z):  # 解码方法
        h = F.relu(self.fc4(z))  # 通过第四层并应用 ReLU 激活函数
        return F.sigmoid(self.fc5(h))  # 通过第五层并应用 Sigmoid 激活函数，返回重构的图像
    
    def forward(self, x):  # 前向传播方法
        mu, log_var = self.encode(x)  # 编码输入以获得均值和对数方差
        z = self.reparameterize(mu, log_var)  # 重参数化以获得潜在变量
        x_reconst = self.decode(z)  # 解码潜在变量以重构输入
        return x_reconst, mu, log_var  # 返回重构的图像、均值和对数方差

model = VAE().to(device)  # 实例化 VAE 模型并移动到指定设备
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器为 Adam

# Start training
for epoch in range(num_epochs):  # 遍历每个训练周期
    for i, (x, _) in enumerate(data_loader):  # 遍历数据加载器中的每个批次
        # Forward pass
        x = x.to(device).view(-1, image_size)  # 将图像移动到设备并展平
        x_reconst, mu, log_var = model(x)  # 前向传播，获取重构图像、均值和对数方差
        
        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)  # 计算重构损失
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # 计算 KL 散度
        
        # Backprop and optimize
        loss = reconst_loss + kl_div  # 计算总损失
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        
        if (i + 1) % 10 == 0:  # 每 10 个批次打印一次信息
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"  # 打印当前周期、步数、重构损失和 KL 散度
                  .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()))
    
    with torch.no_grad():  # 在不计算梯度的情况下执行
        # Save the sampled images
        z = torch.randn(batch_size, z_dim).to(device)  # 生成随机潜在变量
        out = model.decode(z).view(-1, 1, 28, 28)  # 解码潜在变量并调整形状
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))  # 保存生成的图像

        # Save the reconstructed images
        out, _, _ = model(x)  # 获取重构图像
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)  # 拼接原始图像和重构图像
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))  # 保存拼接图像