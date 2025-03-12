from __future__ import print_function  # 为了兼容 Python 2 和 3 的 print 函数
import argparse  # 导入 argparse 库，用于处理命令行参数
from math import log10  # 导入对数函数

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
from torch.utils.data import DataLoader  # 导入数据加载器
from model import Net  # 从自定义模型模块导入 Net 类
from data import get_training_set, get_test_set  # 从数据模块导入获取训练和测试集的函数

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')  # 创建参数解析器，描述为 PyTorch 超分辨率示例
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")  # 添加放大因子参数
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')  # 添加训练批次大小参数，默认值为 64
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')  # 添加测试批次大小参数，默认值为 10
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')  # 添加训练周期数参数，默认值为 2
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')  # 添加学习率参数，默认值为 0.01
parser.add_argument('--cuda', action='store_true', help='use cuda?')  # 添加参数以使用 CUDA
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')  # 添加参数以启用 macOS GPU 训练
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')  # 添加数据加载器使用的线程数参数，默认值为 4
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')  # 添加随机种子参数，默认值为 123
opt = parser.parse_args()  # 解析命令行参数

print(opt)  # 打印解析后的参数

if opt.cuda and not torch.cuda.is_available():  # 如果指定使用 CUDA 但没有可用的 GPU
    raise Exception("No GPU found, please run without --cuda")  # 抛出异常提示没有找到 GPU
if not opt.mps and torch.backends.mps.is_available():  # 如果没有指定使用 MPS 但找到了 MPS 设备
    raise Exception("Found mps device, please run with --mps to enable macOS GPU")  # 抛出异常提示使用 MPS

torch.manual_seed(opt.seed)  # 设置随机种子以确保可重复性
use_mps = opt.mps and torch.backends.mps.is_available()  # 检查是否可以使用 MPS

if opt.cuda:  # 如果 CUDA 可用
    device = torch.device("cuda")  # 设置设备为 CUDA
elif use_mps:  # 如果 MPS 可用
    device = torch.device("mps")  # 设置设备为 MPS
else:  # 否则
    device = torch.device("cpu")  # 设置设备为 CPU

print('===> Loading datasets')  # 打印加载数据集的提示
train_set = get_training_set(opt.upscale_factor)  # 获取训练数据集
test_set = get_test_set(opt.upscale_factor)  # 获取测试数据集
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)  # 创建训练数据加载器
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)  # 创建测试数据加载器

print('===> Building model')  # 打印构建模型的提示
model = Net(upscale_factor=opt.upscale_factor).to(device)  # 实例化模型并移动到指定设备
criterion = nn.MSELoss()  # 定义均方误差损失

optimizer = optim.Adam(model.parameters(), lr=opt.lr)  # 使用 Adam 优化器

def train(epoch):
    """Train the model for one epoch.
    训练模型一个周期。

    Args:
        epoch: 当前周期
    """
    epoch_loss = 0  # 初始化周期损失
    for iteration, batch in enumerate(training_data_loader, 1):  # 遍历训练数据加载器
        input, target = batch[0].to(device), batch[1].to(device)  # 将输入和目标移动到指定设备

        optimizer.zero_grad()  # 清除梯度
        loss = criterion(model(input), target)  # 计算损失
        epoch_loss += loss.item()  # 累加周期损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))  # 打印训练信息

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))  # 打印平均损失


def test():
    """Test the model.
    测试模型。
    """
    avg_psnr = 0  # 初始化平均 PSNR
    with torch.no_grad():  # 在不跟踪历史的情况下执行
        for batch in testing_data_loader:  # 遍历测试数据加载器
            input, target = batch[0].to(device), batch[1].to(device)  # 将输入和目标移动到指定设备

            prediction = model(input)  # 前向传播得到预测结果
            mse = criterion(prediction, target)  # 计算均方误差
            psnr = 10 * log10(1 / mse.item())  # 计算 PSNR
            avg_psnr += psnr  # 累加 PSNR
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))  # 打印平均 PSNR


def checkpoint(epoch):
    """Save the model checkpoint.
    保存模型检查点。

    Args:
        epoch: 当前周期
    """
    model_out_path = "model_epoch_{}.pth".format(epoch)  # 设置模型保存路径
    torch.save(model, model_out_path)  # 保存模型
    print("Checkpoint saved to {}".format(model_out_path))  # 打印保存路径

for epoch in range(1, opt.nEpochs + 1):  # 遍历每个训练周期
    train(epoch)  # 训练模型
    test()  # 测试模型
    checkpoint(epoch)  # 保存模型检查点