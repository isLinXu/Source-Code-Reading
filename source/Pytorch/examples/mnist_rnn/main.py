from __future__ import print_function  # 导入print_function以支持Python 2和3的兼容性

import argparse  # 导入argparse模块，用于处理命令行参数

import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch导入神经网络模块
import torch.nn.functional as F  # 从PyTorch导入功能性模块
import torch.optim as optim  # 从PyTorch导入优化器模块
from torch.optim.lr_scheduler import StepLR  # 从PyTorch导入学习率调度器
from torchvision import datasets, transforms  # 从torchvision导入数据集和数据转换模块


class Net(nn.Module):  # 定义神经网络类Net，继承自nn.Module
    def __init__(self):  # 初始化函数
        super(Net, self).__init__()  # 调用父类构造函数
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)  # 定义LSTM层
        self.batchnorm = nn.BatchNorm1d(64)  # 定义批归一化层
        self.dropout1 = nn.Dropout2d(0.25)  # 定义丢弃层，丢弃率为25%
        self.dropout2 = nn.Dropout2d(0.5)  # 定义丢弃层，丢弃率为50%
        self.fc1 = nn.Linear(64, 32)  # 定义第一个全连接层，输入64，输出32
        self.fc2 = nn.Linear(32, 10)  # 定义第二个全连接层，输入32，输出10（分类数）

    def forward(self, input):  # 前向传播函数
        # Shape of input is (batch_size,1, 28, 28)
        # converting shape of input to (batch_size, 28, 28)
        # as required by RNN when batch_first is set True
        input = input.reshape(-1, 28, 28)  # 将输入形状转换为(batch_size, 28, 28)
        output, hidden = self.rnn(input)  # 通过LSTM层进行前向传播

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        output = output[:, -1, :]  # 获取LSTM的最后输出
        output = self.batchnorm(output)  # 应用批归一化
        output = self.dropout1(output)  # 应用第一个丢弃层
        output = self.fc1(output)  # 通过第一个全连接层
        output = F.relu(output)  # 应用ReLU激活函数
        output = self.dropout2(output)  # 应用第二个丢弃层
        output = self.fc2(output)  # 通过第二个全连接层
        output = F.log_softmax(output, dim=1)  # 返回log_softmax输出
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  # 打印训练状态
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:  # 如果是干运行模式
                break  # 退出循环


def test(args, model, device, test_loader):  # 定义测试函数
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
            if args.dry_run:  # 如果是干运行模式
                break  # 退出循环

    test_loss /= len(test_loader.dataset)  # 计算平均测试损失

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(  # 打印测试结果
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():  # 定义主函数
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example using RNN')  # 创建参数解析器
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',  # 输入批量大小
                        help='input batch size for training (default: 64)')  # 帮助信息
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',  # 输入测试批量大小
                        help='input batch size for testing (default: 1000)')  # 帮助信息
    parser.add_argument('--epochs', type=int, default=14, metavar='N',  # 训练周期数
                        help='number of epochs to train (default: 14)')  # 帮助信息
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',  # 学习率
                        help='learning rate (default: 0.1)')  # 帮助信息
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',  # 学习率衰减系数
                        help='learning rate step gamma (default: 0.7)')  # 帮助信息
    parser.add_argument('--cuda', action='store_true', default=False,  # 启用CUDA训练
                        help='enables CUDA training')  # 帮助信息
    parser.add_argument('--mps', action='store_true', default=False,  # 启用macOS GPU训练
                        help='enables MPS training')  # 帮助信息
    parser.add_argument('--dry-run', action='store_true', default=False,  # 干运行模式
                        help='quickly check a single pass')  # 帮助信息
    parser.add_argument('--seed', type=int, default=1, metavar='S',  # 随机种子
                        help='random seed (default: 1)')  # 帮助信息
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',  # 日志输出间隔
                        help='how many batches to wait before logging training status')  # 帮助信息
    parser.add_argument('--save-model', action='store_true', default=False,  # 保存模型
                        help='for Saving the current Model')  # 帮助信息
    args = parser.parse_args()  # 解析命令行参数

    if args.cuda and not args.mps:  # 如果启用CUDA且不启用MPS
        device = "cuda"  # 设置设备为CUDA
    elif args.mps and not args.cuda:  # 如果启用MPS且不启用CUDA
        device = "mps"  # 设置设备为MPS
    else:  # 如果两者都不启用
        device = "cpu"  # 设置设备为CPU

    device = torch.device(device)  # 创建设备对象

    torch.manual_seed(args.seed)  # 设置随机种子

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # 定义数据加载器参数
    train_loader = torch.utils.data.DataLoader(  # 创建训练数据加载器
        datasets.MNIST('../data', train=True, download=True,  # 加载训练集
                       transform=transforms.Compose([  # 应用数据转换
                           transforms.ToTensor(),  # 转换为张量
                           transforms.Normalize((0.1307,), (0.3081,))  # 归一化处理
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)  # 定义批量大小和是否打乱数据
    test_loader = torch.utils.data.DataLoader(  # 创建测试数据加载器
        datasets.MNIST('../data', train=False, transform=transforms.Compose([  # 加载测试集
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.1307,), (0.3081,))  # 归一化处理
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)  # 定义批量大小和是否打乱数据

    model = Net().to(device)  # 创建模型实例并移动到指定设备
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 创建Adadelta优化器

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # 创建学习率调度器
    for epoch in range(1, args.epochs + 1):  # 遍历每个训练周期
        train(args, model, device, train_loader, optimizer, epoch)  # 训练模型
        test(args, model, device, test_loader)  # 测试模型
        scheduler.step()  # 更新学习率

    if args.save_model:  # 如果指定了保存模型
        torch.save(model.state_dict(), "mnist_rnn.pt")  # 保存模型状态字典


if __name__ == '__main__':  # 主程序入口
    main()  # 调用主函数