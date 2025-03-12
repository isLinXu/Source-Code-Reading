import argparse  # 导入argparse模块，用于处理命令行参数
import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch导入神经网络模块
import torch.nn.functional as F  # 从PyTorch导入功能性模块
import torch.optim as optim  # 从PyTorch导入优化器模块
from torchvision import datasets, transforms  # 从torchvision导入数据集和数据转换模块
from torch.optim.lr_scheduler import StepLR  # 从PyTorch导入学习率调度器


class Net(nn.Module):  # 定义神经网络类Net，继承自nn.Module
    def __init__(self):  # 初始化函数
        super(Net, self).__init__()  # 调用父类构造函数
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 定义第一个卷积层，输入通道1，输出通道32，卷积核大小3，步幅1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 定义第二个卷积层，输入通道32，输出通道64，卷积核大小3，步幅1
        self.dropout1 = nn.Dropout(0.25)  # 定义第一个丢弃层，丢弃率为25%
        self.dropout2 = nn.Dropout(0.5)  # 定义第二个丢弃层，丢弃率为50%
        self.fc1 = nn.Linear(9216, 128)  # 定义第一个全连接层，输入9216，输出128
        self.fc2 = nn.Linear(128, 10)  # 定义第二个全连接层，输入128，输出10（分类数）

    def forward(self, x):  # 前向传播函数
        x = self.conv1(x)  # 通过第一个卷积层
        x = F.relu(x)  # 应用ReLU激活函数
        x = self.conv2(x)  # 通过第二个卷积层
        x = F.relu(x)  # 应用ReLU激活函数
        x = F.max_pool2d(x, 2)  # 进行2x2的最大池化
        x = self.dropout1(x)  # 应用第一个丢弃层
        x = torch.flatten(x, 1)  # 将多维输入展平为一维
        x = self.fc1(x)  # 通过第一个全连接层
        x = F.relu(x)  # 应用ReLU激活函数
        x = self.dropout2(x)  # 应用第二个丢弃层
        x = self.fc2(x)  # 通过第二个全连接层
        output = F.log_softmax(x, dim=1)  # 应用log_softmax激活函数
        return output  # 返回输出


def train(args, model, device, train_loader, optimizer, epoch):  # 定义训练函数
    model.train()  # 将模型设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练数据加载器
        data, target = data.to(device), target.to(device)  # 将数据和目标移动到设备
        optimizer.zero_grad()  # 清零梯度
        output = model(data)  # 前向传播，获取模型输出
        loss = F.nll_loss(output, target)  # 计算负对数似然损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新优化器
        if batch_idx % args.log_interval == 0:  # 每log_interval个批次打印一次日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))  # 打印训练状态
            if args.dry_run:  # 如果是干运行模式
                break  # 退出循环


def test(model, device, test_loader):  # 定义测试函数
    model.eval()  # 将模型设置为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测数量
    with torch.no_grad():  # 在不计算梯度的情况下进行测试
        for data, target in test_loader:  # 遍历测试数据加载器
            data, target = data.to(device), target.to(device)  # 将数据和目标移动到设备
            output = model(data)  # 前向传播，获取模型输出
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加测试损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测数量

    test_loss /= len(test_loader.dataset)  # 计算平均测试损失

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))  # 打印测试结果


def main():  # 定义主函数
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')  # 创建参数解析器
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',  # 输入批量大小
                        help='input batch size for training (default: 64)')  # 帮助信息
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',  # 输入测试批量大小
                        help='input batch size for testing (default: 1000)')  # 帮助信息
    parser.add_argument('--epochs', type=int, default=14, metavar='N',  # 训练周期数
                        help='number of epochs to train (default: 14)')  # 帮助信息
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',  # 学习率
                        help='learning rate (default: 1.0)')  # 帮助信息
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',  # 学习率衰减系数
                        help='Learning rate step gamma (default: 0.7)')  # 帮助信息
    parser.add_argument('--no-cuda', action='store_true', default=False,  # 禁用CUDA训练
                        help='disables CUDA training')  # 帮助信息
    parser.add_argument('--no-mps', action='store_true', default=False,  # 禁用macOS GPU训练
                        help='disables macOS GPU training')  # 帮助信息
    parser.add_argument('--dry-run', action='store_true', default=False,  # 干运行模式
                        help='quickly check a single pass')  # 帮助信息
    parser.add_argument('--seed', type=int, default=1, metavar='S',  # 随机种子
                        help='random seed (default: 1)')  # 帮助信息
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',  # 日志输出间隔
                        help='how many batches to wait before logging training status')  # 帮助信息
    parser.add_argument('--save-model', action='store_true', default=False,  # 保存模型
                        help='For Saving the current Model')  # 帮助信息
    args = parser.parse_args()  # 解析命令行参数
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # 检查是否使用CUDA
    use_mps = not args.no_mps and torch.backends.mps.is_available()  # 检查是否使用MPS

    torch.manual_seed(args.seed)  # 设置随机种子

    if use_cuda:  # 如果使用CUDA
        device = torch.device("cuda")  # 创建CUDA设备对象
    elif use_mps:  # 如果使用MPS
        device = torch.device("mps")  # 创建MPS设备对象
    else:  # 如果没有可用的GPU或MPS
        device = torch.device("cpu")  # 使用CPU设备

    train_kwargs = {'batch_size': args.batch_size}  # 定义训练参数
    test_kwargs = {'batch_size': args.test_batch_size}  # 定义测试参数
    if use_cuda:  # 如果使用CUDA
        cuda_kwargs = {'num_workers': 1,  # 工作线程数
                       'pin_memory': True,  # 固定内存
                       'shuffle': True}  # 随机打乱数据
        train_kwargs.update(cuda_kwargs)  # 更新训练参数
        test_kwargs.update(cuda_kwargs)  # 更新测试参数

    transform = transforms.Compose([  # 定义数据转换
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化处理
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,  # 加载训练集
                       transform=transform)  # 应用数据转换
    dataset2 = datasets.MNIST('../data', train=False,  # 加载测试集
                       transform=transform)  # 应用数据转换
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)  # 创建训练数据加载器
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)  # 创建测试数据加载器

    model = Net().to(device)  # 创建模型实例并移动到指定设备
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 创建Adadelta优化器

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # 创建学习率调度器
    for epoch in range(1, args.epochs + 1):  # 遍历每个训练周期
        train(args, model, device, train_loader, optimizer, epoch)  # 训练模型
        test(model, device, test_loader)  # 测试模型
        scheduler.step()  # 更新学习率

    if args.save_model:  # 如果指定了保存模型
        torch.save(model.state_dict(), "mnist_cnn.pt")  # 保存模型状态字典


if __name__ == '__main__':  # 主程序入口
    main()  # 调用主函数