import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，用于与操作系统交互
from threading import Lock  # 从threading模块导入Lock类，用于线程锁

import torch  # 导入PyTorch库
import torch.distributed.autograd as dist_autograd  # 导入分布式自动求导模块
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
from torch import optim  # 从torch模块导入优化器
from torch.distributed.optim import DistributedOptimizer  # 导入分布式优化器
from torchvision import datasets, transforms  # 从torchvision导入数据集和转换工具

# --------- MNIST Network to train, from pytorch/examples -----

class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()  # 调用父类构造函数
        print(f"Using {num_gpus} GPUs to train")  # 打印使用的GPU数量
        self.num_gpus = num_gpus  # 保存GPU数量
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")  # 设置设备为第一个可用的CUDA设备或CPU
        print(f"Putting first 2 convs on {str(device)}")  # 打印将前两个卷积层放置的设备
        # Put conv layers on the first cuda device
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)  # 在设备上创建第一个卷积层
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)  # 在设备上创建第二个卷积层
        # Put rest of the network on the 2nd cuda device, if there is one
        if "cuda" in str(device) and num_gpus > 1:  # 如果设备是CUDA并且有多个GPU
            device = torch.device("cuda:1")  # 设置设备为第二个CUDA设备

        print(f"Putting rest of layers on {str(device)}")  # 打印将其余层放置的设备
        self.dropout1 = nn.Dropout2d(0.25).to(device)  # 在设备上创建第一个Dropout层
        self.dropout2 = nn.Dropout2d(0.5).to(device)  # 在设备上创建第二个Dropout层
        self.fc1 = nn.Linear(9216, 128).to(device)  # 在设备上创建第一个全连接层
        self.fc2 = nn.Linear(128, 10).to(device)  # 在设备上创建第二个全连接层

    def forward(self, x):
        x = self.conv1(x)  # 通过第一个卷积层
        x = F.relu(x)  # 应用ReLU激活函数
        x = self.conv2(x)  # 通过第二个卷积层
        x = F.max_pool2d(x, 2)  # 进行最大池化

        x = self.dropout1(x)  # 应用第一个Dropout层
        x = torch.flatten(x, 1)  # 将张量展平
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device  # 获取下一个设备
        x = x.to(next_device)  # 将张量移动到下一个设备

        x = self.fc1(x)  # 通过第一个全连接层
        x = F.relu(x)  # 应用ReLU激活函数
        x = self.dropout2(x)  # 应用第二个Dropout层
        x = self.fc2(x)  # 通过第二个全连接层
        output = F.log_softmax(x, dim=1)  # 应用log_softmax激活函数
        return output  # 返回输出


# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)  # 调用方法并传递RRef的本地值和其他参数

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.
def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)  # 将方法和RRef添加到参数列表中
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)  # 在拥有RRef的远程节点上同步调用方法


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()  # 调用父类构造函数
        model = Net(num_gpus=num_gpus)  # 创建网络模型
        self.model = model  # 保存模型
        self.input_device = torch.device(  # 设置输入设备
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    def forward(self, inp):
        inp = inp.to(self.input_device)  # 将输入移动到输入设备
        out = self.model(inp)  # 通过模型进行前向传播
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")  # 将输出移动到CPU
        return out  # 返回输出

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)  # 获取分布式自动求导的梯度
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}  # 初始化CPU梯度字典
        for k, v in grads.items():  # 遍历梯度
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")  # 将梯度移动到CPU
            cpu_grads[k_cpu] = v_cpu  # 保存CPU梯度
        return cpu_grads  # 返回CPU梯度

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]  # 获取模型参数的远程引用
        return param_rrefs  # 返回参数的远程引用


param_server = None  # 初始化参数服务器为None
global_lock = Lock()  # 创建全局锁


def get_parameter_server(num_gpus=0):
    global param_server  # 声明使用全局变量param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:  # 使用全局锁确保线程安全
        if not param_server:  # 如果参数服务器未初始化
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)  # 创建参数服务器
        return param_server  # 返回参数服务器


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")  # 打印参数服务器初始化RPC的信息
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)  # 初始化RPC
    print("RPC initialized! Running parameter server...")  # 打印RPC初始化完成的信息
    rpc.shutdown()  # 关闭RPC
    print("RPC shutdown on parameter server.")  # 打印参数服务器关闭RPC的信息


# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()  # 调用父类构造函数
        self.num_gpus = num_gpus  # 保存GPU数量
        self.param_server_rref = rpc.remote(  # 创建参数服务器的远程引用
            "parameter_server", get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(  # 获取参数服务器的参数远程引用
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params  # 返回远程参数引用

    def forward(self, x):
        model_output = remote_method(  # 通过参数服务器进行前向传播
            ParameterServer.forward, self.param_server_rref, x)
        return model_output  # 返回模型输出


def run_training_loop(rank, num_gpus, train_loader, test_loader):
    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(num_gpus=num_gpus)  # 创建训练网络实例
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()  # 获取全局参数的远程引用
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)  # 创建分布式优化器
    for i, (data, target) in enumerate(train_loader):  # 遍历训练数据
        with dist_autograd.context() as cid:  # 创建分布式自动求导上下文
            model_output = net(data)  # 前向传播
            target = target.to(model_output.device)  # 将目标标签移动到模型输出的设备
            loss = F.nll_loss(model_output, target)  # 计算负对数似然损失
            if i % 5 == 0:  # 每5个批次打印一次损失
                print(f"Rank {rank} training batch {i} loss {loss.item()}")  # 打印当前批次的损失
            dist_autograd.backward(cid, [loss])  # 反向传播

            # Ensure that dist autograd ran successfully and gradients were
            # returned.
            assert remote_method(  # 确保分布式自动求导成功运行并返回梯度
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                cid) != {}
            opt.step(cid)  # 更新优化器

    print("Training complete!")  # 打印训练完成的信息
    print("Getting accuracy....")  # 打印获取准确率的信息
    get_accuracy(test_loader, net)  # 获取模型的准确率


def get_accuracy(test_loader, model):
    model.eval()  # 设置模型为评估模式
    correct_sum = 0  # 初始化正确预测的数量
    # Use GPU to evaluate if possible
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")  # 设置评估设备
    with torch.no_grad():  # 在不计算梯度的情况下进行评估
        for i, (data, target) in enumerate(test_loader):  # 遍历测试数据
            out = model(data)  # 前向传播
            pred = out.argmax(dim=1, keepdim=True)  # 获取预测结果
            pred, target = pred.to(device), target.to(device)  # 将预测和目标移动到设备
            correct = pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测的数量
            correct_sum += correct  # 累加正确预测的数量

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")  # 打印模型的准确率


# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    print(f"Worker rank {rank} initializing RPC")  # 打印工作者初始化RPC的信息
    rpc.init_rpc(  # 初始化RPC
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    print(f"Worker {rank} done initializing RPC")  # 打印工作者初始化完成的信息

    run_training_loop(rank, num_gpus, train_loader, test_loader)  # 运行训练循环
    rpc.shutdown()  # 关闭RPC

# --------- Launcher --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(  # 创建命令行参数解析器
        description="Parameter-Server RPC based training")  # 描述信息
    parser.add_argument(  # 添加参数：世界规模
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")  # 参与进程的总数，应该是主节点和所有训练节点的总和
    parser.add_argument(  # 添加参数：当前进程的排名
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")  # 当前进程的全局排名，主节点传入0
    parser.add_argument(  # 添加参数：使用的GPU数量
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")  # 使用的GPU数量，目前支持0到2个GPU
    parser.add_argument(  # 添加参数：主节点地址
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")  # 主节点的地址，未提供时默认为localhost
    parser.add_argument(  # 添加参数：主节点端口
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")  # 主节点监听的端口，未提供时默认为29500

    args = parser.parse_args()  # 解析命令行参数
    assert args.rank is not None, "must provide rank argument."  # 确保提供了排名参数
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."  # 确保GPU数量在支持范围内
    os.environ['MASTER_ADDR'] = args.master_addr  # 设置主节点地址
    os.environ['MASTER_PORT'] = args.master_port  # 设置主节点端口
    processes = []  # 初始化进程列表
    world_size = args.world_size  # 获取世界规模

    # Note that Linux uses "fork" by default, which may cause deadlock.
    # Besides, cuda doesn't support "fork" and Windows only supports "spawn"
    mp.set_start_method("spawn")  # 设置进程启动方法为spawn

    if args.rank == 0:  # 如果是主节点
        p = mp.Process(target=run_parameter_server, args=(0, world_size))  # 创建参数服务器进程
        p.start()  # 启动参数服务器进程
        processes.append(p)  # 将进程添加到列表中
    else:  # 如果是训练者
        # Get data to train on
        train_loader = torch.utils.data.DataLoader(  # 创建训练数据加载器
            datasets.MNIST('../data', train=True, download=True,  # 下载MNIST训练数据集
                           transform=transforms.Compose([  # 数据转换
                               transforms.ToTensor(),  # 转换为张量
                               transforms.Normalize((0.1307,), (0.3081,))  # 归一化
                           ])),
            batch_size=32, shuffle=True)  # 设置批处理大小和随机打乱
        test_loader = torch.utils.data.DataLoader(  # 创建测试数据加载器
            datasets.MNIST('../data', train=False,  # 下载MNIST测试数据集
                           transform=transforms.Compose([  # 数据转换
                               transforms.ToTensor(),  # 转换为张量
                               transforms.Normalize((0.1307,), (0.3081,))  # 归一化
                           ])),
            batch_size=32, shuffle=True)  # 设置批处理大小和随机打乱
        # start training worker on this node
        p = mp.Process(  # 创建训练者进程
            target=run_worker,
            args=(
                args.rank,
                world_size, args.num_gpus,
                train_loader,
                test_loader))
        p.start()  # 启动训练者进程
        processes.append(p)  # 将进程添加到列表中

    for p in processes:  # 等待所有进程完成
        p.join()  # 等待进程结束