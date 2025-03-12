from __future__ import print_function  # 为了兼容 Python 2 和 3 的 print 函数
import argparse  # 导入 argparse 库，用于处理命令行参数
import torch  # 导入 PyTorch 库
import torch.utils.data  # 导入 PyTorch 数据处理工具
from torch import nn, optim  # 导入 PyTorch 的神经网络模块和优化器
from torch.nn import functional as F  # 导入 PyTorch 的功能性神经网络模块
from torchvision import datasets, transforms  # 导入 torchvision 数据集和转换工具
from torchvision.utils import save_image  # 导入保存图像的工具

parser = argparse.ArgumentParser(description='VAE MNIST Example')  # 创建参数解析器，描述为 VAE MNIST 示例
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')  # 添加批次大小参数，默认值为 128
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')  # 添加训练周期数参数，默认值为 10
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')  # 添加参数以禁用 CUDA 训练
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')  # 添加参数以禁用 macOS GPU 训练
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')  # 添加随机种子参数，默认值为 1
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')  # 添加日志间隔参数，默认值为 10
args = parser.parse_args()  # 解析命令行参数
args.cuda = not args.no_cuda and torch.cuda.is_available()  # 检查是否可以使用 CUDA
use_mps = not args.no_mps and torch.backends.mps.is_available()  # 检查是否可以使用 macOS GPU

torch.manual_seed(args.seed)  # 设置随机种子以确保可重复性

if args.cuda:  # 如果 CUDA 可用
    device = torch.device("cuda")  # 设置设备为 CUDA
elif use_mps:  # 如果 MPS 可用
    device = torch.device("mps")  # 设置设备为 MPS
else:  # 否则
    device = torch.device("cpu")  # 设置设备为 CPU

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # 设置 DataLoader 的参数
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),  # 下载并加载 MNIST 训练数据集
    batch_size=args.batch_size, shuffle=True, **kwargs)  # 创建训练数据加载器
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),  # 加载 MNIST 测试数据集
    batch_size=args.batch_size, shuffle=False, **kwargs)  # 创建测试数据加载器

class VAE(nn.Module):
    """Variational Autoencoder model.
    变分自编码器模型。"""

    def __init__(self):
        """Initialize the VAE model.
        初始化 VAE 模型。"""
        super(VAE, self).__init__()  # 调用父类构造函数

        self.fc1 = nn.Linear(784, 400)  # 第一层全连接层，将输入维度从 784 映射到 400
        self.fc21 = nn.Linear(400, 20)  # 第二层全连接层，输出均值
        self.fc22 = nn.Linear(400, 20)  # 第二层全连接层，输出对数方差
        self.fc3 = nn.Linear(20, 400)  # 第三层全连接层，将隐变量映射回 400
        self.fc4 = nn.Linear(400, 784)  # 第四层全连接层，将输出维度映射回 784

    def encode(self, x):
        """Encode the input.
        编码输入。

        Args:
            x: 输入数据
        Returns:
            mu: 均值
            logvar: 对数方差
        """
        h1 = F.relu(self.fc1(x))  # 通过第一层并应用 ReLU 激活函数
        return self.fc21(h1), self.fc22(h1)  # 返回均值和对数方差

    def reparameterize(self, mu, logvar):
        """Reparameterization trick.
        重参数化技巧。

        Args:
            mu: 均值
            logvar: 对数方差
        Returns:
            z: 隐变量
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 生成与标准差相同形状的随机噪声
        return mu + eps * std  # 返回重参数化的隐变量

    def decode(self, z):
        """Decode the latent variable.
        解码隐变量。

        Args:
            z: 隐变量
        Returns:
            输出数据
        """
        h3 = F.relu(self.fc3(z))  # 通过第三层并应用 ReLU 激活函数
        return torch.sigmoid(self.fc4(h3))  # 通过第四层并应用 sigmoid 激活函数

    def forward(self, x):
        """Forward pass.
        前向传播。

        Args:
            x: 输入数据
        Returns:
            decoded: 解码后的输出
            mu: 均值
            logvar: 对数方差
        """
        mu, logvar = self.encode(x.view(-1, 784))  # 将输入展平并编码
        z = self.reparameterize(mu, logvar)  # 重参数化得到隐变量
        return self.decode(z), mu, logvar  # 返回解码后的输出、均值和对数方差


model = VAE().to(device)  # 实例化 VAE 模型并移动到指定设备
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 使用 Adam 优化器

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    """Compute the loss function.
    计算损失函数。

    Args:
        recon_x: 重建的输出
        x: 原始输入
        mu: 均值
        logvar: 对数方差
    Returns:
        总损失
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')  # 计算二元交叉熵损失

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 计算 KL 散度损失

    return BCE + KLD  # 返回总损失


def train(epoch):
    """Train the model for one epoch.
    训练模型一个周期。

    Args:
        epoch: 当前周期
    """
    model.train()  # 设置模型为训练模式
    train_loss = 0  # 初始化训练损失
    for batch_idx, (data, _) in enumerate(train_loader):  # 遍历训练数据加载器
        data = data.to(device)  # 将数据移动到指定设备
        optimizer.zero_grad()  # 清除梯度
        recon_batch, mu, logvar = model(data)  # 前向传播
        loss = loss_function(recon_batch, data, mu, logvar)  # 计算损失
        loss.backward()  # 反向传播
        train_loss += loss.item()  # 累加训练损失
        optimizer.step()  # 更新模型参数
        if batch_idx % args.log_interval == 0:  # 每 log_interval 批次打印一次信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))  # 打印训练信息

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))  # 打印平均损失


def test(epoch):
    """Test the model.
    测试模型。

    Args:
        epoch: 当前周期
    """
    model.eval()  # 设置模型为评估模式
    test_loss = 0  # 初始化测试损失
    with torch.no_grad():  # 在不跟踪历史的情况下执行
        for i, (data, _) in enumerate(test_loader):  # 遍历测试数据加载器
            data = data.to(device)  # 将数据移动到指定设备
            recon_batch, mu, logvar = model(data)  # 前向传播
            test_loss += loss_function(recon_batch, data, mu, logvar).item()  # 累加测试损失
            if i == 0:  # 仅在第一个批次保存重建图像
                n = min(data.size(0), 8)  # 取最小的样本数量
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])  # 拼接原始和重建图像
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)  # 保存重建图像

    test_loss /= len(test_loader.dataset)  # 计算平均测试损失
    print('====> Test set loss: {:.4f}'.format(test_loss))  # 打印测试集损失

if __name__ == "__main__":  # 如果是主程序
    for epoch in range(1, args.epochs + 1):  # 遍历每个训练周期
        train(epoch)  # 训练模型
        test(epoch)  # 测试模型
        with torch.no_grad():  # 在不跟踪历史的情况下执行
            sample = torch.randn(64, 20).to(device)  # 生成随机样本
            sample = model.decode(sample).cpu()  # 解码样本
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')  # 保存生成的样本图像