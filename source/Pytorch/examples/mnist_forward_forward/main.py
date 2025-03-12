# This code is based on the implementation of Mohammad Pezeshki available at
# https://github.com/mohammadpz/pytorch_forward_forward and licensed under the MIT License.
# Modifications/Improvements to the original code have been made by Vivek V Patel.

import argparse  # 导入argparse模块，用于处理命令行参数
import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch导入神经网络模块
from torchvision.datasets import MNIST  # 从torchvision导入MNIST数据集
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda  # 从torchvision导入数据转换模块
from torch.utils.data import DataLoader  # 从PyTorch导入数据加载器
from torch.optim import Adam  # 从PyTorch导入Adam优化器


def get_y_neg(y):  # 定义获取负样本标签的函数
    y_neg = y.clone()  # 克隆输入标签
    for idx, y_samp in enumerate(y):  # 遍历每个标签
        allowed_indices = list(range(10))  # 允许的标签索引范围
        allowed_indices.remove(y_samp.item())  # 移除当前样本的标签
        y_neg[idx] = torch.tensor(allowed_indices)[  # 随机选择一个不同的标签
            torch.randint(len(allowed_indices), size=(1,))
        ].item()  # 将选择的标签赋值给负样本标签
    return y_neg.to(device)  # 将负样本标签移动到设备


def overlay_y_on_x(x, y, classes=10):  # 定义在输入上叠加标签的函数
    x_ = x.clone()  # 克隆输入数据
    x_[:, :classes] *= 0.0  # 将前classes个通道的值置为0
    x_[range(x.shape[0]), y] = x.max()  # 将最大值放置在对应标签的位置
    return x_  # 返回修改后的输入数据


class Net(torch.nn.Module):  # 定义神经网络类Net，继承自nn.Module
    def __init__(self, dims):  # 初始化函数，接受层的维度
        super().__init__()  # 调用父类构造函数
        self.layers = []  # 初始化层列表
        for d in range(len(dims) - 1):  # 遍历维度列表
            self.layers = self.layers + [Layer(dims[d], dims[d + 1]).to(device)]  # 创建层并移动到设备

    def predict(self, x):  # 定义预测函数
        goodness_per_label = []  # 初始化每个标签的好坏值列表
        for label in range(10):  # 遍历每个标签
            h = overlay_y_on_x(x, label)  # 在输入上叠加当前标签
            goodness = []  # 初始化好坏值列表
            for layer in self.layers:  # 遍历每一层
                h = layer(h)  # 通过层进行前向传播
                goodness = goodness + [h.pow(2).mean(1)]  # 计算当前层的好坏值
            goodness_per_label += [sum(goodness).unsqueeze(1)]  # 将好坏值相加并添加到列表
        goodness_per_label = torch.cat(goodness_per_label, 1)  # 将好坏值合并为一个张量
        return goodness_per_label.argmax(1)  # 返回好坏值最大的标签索引

    def train(self, x_pos, x_neg):  # 定义训练函数
        h_pos, h_neg = x_pos, x_neg  # 初始化正样本和负样本
        for i, layer in enumerate(self.layers):  # 遍历每一层
            print("training layer: ", i)  # 打印当前训练的层
            h_pos, h_neg = layer.train(h_pos, h_neg)  # 训练当前层


class Layer(nn.Linear):  # 定义Layer类，继承自nn.Linear
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):  # 初始化函数
        super().__init__(in_features, out_features, bias, device, dtype)  # 调用父类构造函数
        self.relu = torch.nn.ReLU()  # 定义ReLU激活函数
        self.opt = Adam(self.parameters(), lr=args.lr)  # 创建Adam优化器
        self.threshold = args.threshold  # 设置阈值
        self.num_epochs = args.epochs  # 设置训练周期数

    def forward(self, x):  # 前向传播函数
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)  # 归一化输入
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))  # 计算输出

    def train(self, x_pos, x_neg):  # 定义训练函数
        for i in range(self.num_epochs):  # 遍历每个训练周期
            g_pos = self.forward(x_pos).pow(2).mean(1)  # 计算正样本的好坏值
            g_neg = self.forward(x_neg).pow(2).mean(1)  # 计算负样本的好坏值
            loss = torch.log1p(  # 计算损失
                torch.exp(
                    torch.cat([-g_pos + self.threshold, g_neg - self.threshold])  # 合并好坏值
                )
            ).mean()  # 计算平均损失
            self.opt.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            self.opt.step()  # 更新优化器
            if i % args.log_interval == 0:  # 每log_interval个周期打印一次损失
                print("Loss: ", loss.item())  # 打印损失
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()  # 返回正负样本的输出


if __name__ == "__main__":  # 主程序入口
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(  # 添加训练周期参数
        "--epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of epochs to train (default: 1000)",
    )
    parser.add_argument(  # 添加学习率参数
        "--lr",
        type=float,
        default=0.03,
        metavar="LR",
        help="learning rate (default: 0.03)",
    )
    parser.add_argument(  # 添加禁用CUDA训练参数
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(  # 添加禁用MPS训练参数
        "--no_mps", action="store_true", default=False, help="disables MPS training"
    )
    parser.add_argument(  # 添加随机种子参数
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(  # 添加保存模型参数
        "--save_model",
        action="store_true",
        default=False,
        help="For saving the current Model",
    )
    parser.add_argument(  # 添加训练集大小参数
        "--train_size", type=int, default=50000, help="size of training set"
    )
    parser.add_argument(  # 添加训练阈值参数
        "--threshold", type=float, default=2, help="threshold for training"
    )
    parser.add_argument(  # 添加测试集大小参数
        "--test_size", type=int, default=10000, help="size of test set"
    )
    parser.add_argument(  # 添加保存模型参数
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(  # 添加日志输出间隔参数
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()  # 解析命令行参数
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # 检查是否使用CUDA
    use_mps = not args.no_mps and torch.backends.mps.is_available()  # 检查是否使用MPS
    if use_cuda:  # 如果使用CUDA
        device = torch.device("cuda")  # 创建CUDA设备对象
    elif use_mps:  # 如果使用MPS
        device = torch.device("mps")  # 创建MPS设备对象
    else:  # 如果没有可用的GPU或MPS
        device = torch.device("cpu")  # 使用CPU设备

    train_kwargs = {"batch_size": args.train_size}  # 定义训练参数
    test_kwargs = {"batch_size": args.test_size}  # 定义测试参数

    if use_cuda:  # 如果使用CUDA
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}  # CUDA相关参数
        train_kwargs.update(cuda_kwargs)  # 更新训练参数
        test_kwargs.update(cuda_kwargs)  # 更新测试参数

    transform = Compose(  # 定义数据转换
        [
            ToTensor(),  # 转换为张量
            Normalize((0.1307,), (0.3081,)),  # 归一化处理
            Lambda(lambda x: torch.flatten(x)),  # 展平数据
        ]
    )
    train_loader = DataLoader(  # 创建训练数据加载器
        MNIST("./data/", train=True, download=True, transform=transform), **train_kwargs
    )
    test_loader = DataLoader(  # 创建测试数据加载器
        MNIST("./data/", train=False, download=True, transform=transform), **test_kwargs
    )
    net = Net([784, 500, 500])  # 创建模型实例
    x, y = next(iter(train_loader))  # 获取训练数据
    x, y = x.to(device), y.to(device)  # 将数据移动到设备
    x_pos = overlay_y_on_x(x, y)  # 叠加正样本标签
    y_neg = get_y_neg(y)  # 获取负样本标签
    x_neg = overlay_y_on_x(x, y_neg)  # 叠加负样本标签
    net.train(x_pos, x_neg)  # 训练模型
    print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())  # 打印训练错误率
    x_te, y_te = next(iter(test_loader))  # 获取测试数据
    x_te, y_te = x_te.to(device), y_te.to(device)  # 将测试数据移动到设备
    if args.save_model:  # 如果指定了保存模型
        torch.save(net.state_dict(), "mnist_ff.pt")  # 保存模型状态字典
    print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())  # 打印测试错误率