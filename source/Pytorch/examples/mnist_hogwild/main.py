from __future__ import print_function  # 导入print_function以支持Python 2和3的兼容性
import argparse  # 导入argparse模块，用于处理命令行参数
import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch导入神经网络模块
import torch.nn.functional as F  # 从PyTorch导入功能性模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
from torch.utils.data.sampler import Sampler  # 从PyTorch导入采样器
from torchvision import datasets, transforms  # 从torchvision导入数据集和数据转换模块

from train import train, test  # 从train模块导入训练和测试函数

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')  # 创建参数解析器
parser.add_argument('--batch-size', type=int, default=64, metavar='N',  # 输入批量大小
                    help='input batch size for training (default: 64)')  # 帮助信息
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',  # 输入测试批量大小
                    help='input batch size for testing (default: 1000)')  # 帮助信息
parser.add_argument('--epochs', type=int, default=10, metavar='N',  # 训练周期数
                    help='number of epochs to train (default: 10)')  # 帮助信息
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  # 学习率
                    help='learning rate (default: 0.01)')  # 帮助信息
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',  # SGD动量
                    help='SGD momentum (default: 0.5)')  # 帮助信息
parser.add_argument('--seed', type=int, default=1, metavar='S',  # 随机种子
                    help='random seed (default: 1)')  # 帮助信息
parser.add_argument('--log-interval', type=int, default=10, metavar='N',  # 日志输出间隔
                    help='how many batches to wait before logging training status')  # 帮助信息
parser.add_argument('--num-processes', type=int, default=2, metavar='N',  # 训练进程数量
                    help='how many training processes to use (default: 2)')  # 帮助信息
parser.add_argument('--cuda', action='store_true', default=False,  # 启用CUDA训练
                    help='enables CUDA training')  # 帮助信息
parser.add_argument('--mps', action='store_true', default=False,  # 启用macOS GPU训练
                    help='enables macOS GPU training')  # 帮助信息
parser.add_argument('--save_model', action='store_true', default=False,  # 保存模型
                    help='save the trained model to state_dict')  # 帮助信息
parser.add_argument('--dry-run', action='store_true', default=False,  # 干运行模式
                    help='quickly check a single pass')  # 帮助信息

class Net(nn.Module):  # 定义神经网络类Net，继承自nn.Module
    def __init__(self):  # 初始化函数
        super(Net, self).__init__()  # 调用父类构造函数
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 定义第一个卷积层，输入通道1，输出通道10，卷积核大小5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 定义第二个卷积层，输入通道10，输出通道20，卷积核大小5
        self.conv2_drop = nn.Dropout2d()  # 定义丢弃层
        self.fc1 = nn.Linear(320, 50)  # 定义第一个全连接层，输入320，输出50
        self.fc2 = nn.Linear(50, 10)  # 定义第二个全连接层，输入50，输出10（分类数）

    def forward(self, x):  # 前向传播函数
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 通过第一个卷积层和最大池化层
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # 通过第二个卷积层、丢弃层和最大池化层
        x = x.view(-1, 320)  # 将多维输入展平为一维
        x = F.relu(self.fc1(x))  # 通过第一个全连接层
        x = F.dropout(x, training=self.training)  # 应用丢弃层
        x = self.fc2(x)  # 通过第二个全连接层
        return F.log_softmax(x, dim=1)  # 返回log_softmax输出


if __name__ == '__main__':  # 主程序入口
    args = parser.parse_args()  # 解析命令行参数

    use_cuda = args.cuda and torch.cuda.is_available()  # 检查是否使用CUDA
    use_mps = args.mps and torch.backends.mps.is_available()  # 检查是否使用MPS
    if use_cuda:  # 如果使用CUDA
        device = torch.device("cuda")  # 创建CUDA设备对象
    elif use_mps:  # 如果使用MPS
        device = torch.device("mps")  # 创建MPS设备对象
    else:  # 如果没有可用的GPU或MPS
        device = torch.device("cpu")  # 使用CPU设备

    transform = transforms.Compose([  # 定义数据转换
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化处理
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,  # 加载训练集
                       transform=transform)  # 应用数据转换
    dataset2 = datasets.MNIST('../data', train=False,  # 加载测试集
                       transform=transform)  # 应用数据转换
    kwargs = {'batch_size': args.batch_size,  # 定义数据加载器参数
              'shuffle': True}  # 随机打乱数据
    if use_cuda:  # 如果使用CUDA
        kwargs.update({'num_workers': 1,  # 工作线程数
                       'pin_memory': True,  # 固定内存
                      })

    torch.manual_seed(args.seed)  # 设置随机种子
    mp.set_start_method('spawn', force=True)  # 设置多进程启动方法

    model = Net().to(device)  # 创建模型实例并移动到指定设备
    model.share_memory()  # 共享模型内存，以便多进程访问

    processes = []  # 初始化进程列表
    for rank in range(args.num_processes):  # 遍历每个训练进程
        p = mp.Process(target=train, args=(rank, args, model, device,  # 创建新进程
                                           dataset1, kwargs))
        # We first train the model across `num_processes` processes
        p.start()  # 启动进程
        processes.append(p)  # 将进程添加到列表
    for p in processes:  # 等待所有进程完成
        p.join()

    if args.save_model:  # 如果指定了保存模型
        torch.save(model.state_dict(), "MNIST_hogwild.pt")  # 保存模型状态字典

    # Once training is complete, we can test the model
    test(args, model, device, dataset2, kwargs)  # 测试模型