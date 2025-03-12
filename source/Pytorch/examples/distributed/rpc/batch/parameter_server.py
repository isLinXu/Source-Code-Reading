import os  # 导入os模块，用于与操作系统交互
import threading  # 导入threading模块，用于多线程操作
from datetime import datetime  # 从datetime模块导入datetime类，用于处理日期和时间

import torch  # 导入PyTorch库
import torch.distributed.rpc as rpc  # 导入PyTorch的分布式RPC模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch import optim  # 从torch模块导入优化器

import torchvision  # 导入torchvision库，用于计算机视觉任务


batch_size = 20  # 批处理大小
image_w = 64  # 图像宽度
image_h = 64  # 图像高度
num_classes = 30  # 类别数量
batch_update_size = 5  # 批更新大小
num_batches = 6  # 批次数


def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")  # 打印当前时间和文本


class BatchUpdateParameterServer(object):

    def __init__(self, batch_update_size=batch_update_size):
        self.model = torchvision.models.resnet50(num_classes=num_classes)  # 创建ResNet50模型
        self.lock = threading.Lock()  # 创建线程锁
        self.future_model = torch.futures.Future()  # 创建未来模型的占位符
        self.batch_update_size = batch_update_size  # 设置批更新大小
        self.curr_update_size = 0  # 当前更新大小
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)  # 创建SGD优化器
        for p in self.model.parameters():  # 初始化每个参数的梯度
            p.grad = torch.zeros_like(p)  # 将梯度初始化为与参数相同的零张量

    def get_model(self):
        return self.model  # 返回模型

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()  # 获取参数服务器的本地值
        timed_log(f"PS got {self.curr_update_size}/{batch_update_size} updates")  # 打印当前更新数量
        for p, g in zip(self.model.parameters(), grads):  # 遍历模型参数和梯度
            p.grad += g  # 累加梯度
        with self.lock:  # 使用锁确保线程安全
            self.curr_update_size += 1  # 增加当前更新数量
            fut = self.future_model  # 保存当前未来模型

            if self.curr_update_size >= self.batch_update_size:  # 如果达到批更新大小
                for p in self.model.parameters():  # 遍历模型参数
                    p.grad /= self.batch_update_size  # 平均梯度
                self.curr_update_size = 0  # 重置当前更新数量
                self.optimizer.step()  # 更新模型参数
                self.optimizer.zero_grad(set_to_none=False)  # 清零梯度
                fut.set_result(self.model)  # 设置未来模型的结果
                timed_log("PS updated model")  # 打印模型更新信息
                self.future_model = torch.futures.Future()  # 创建新的未来模型占位符

        return fut  # 返回未来模型


class Trainer(object):

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref  # 保存参数服务器的RRef
        self.loss_fn = nn.MSELoss()  # 创建均方误差损失函数
        self.one_hot_indices = torch.LongTensor(batch_size) \
                                    .random_(0, num_classes) \
                                    .view(batch_size, 1)  # 创建随机的one-hot索引

    def get_next_batch(self):
        for _ in range(num_batches):  # 遍历每个批次
            inputs = torch.randn(batch_size, 3, image_w, image_h)  # 生成随机输入
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, self.one_hot_indices, 1)  # 创建one-hot标签
            yield inputs.cuda(), labels.cuda()  # 将输入和标签移动到GPU并返回

    def train(self):
        name = rpc.get_worker_info().name  # 获取当前工作者的名称
        m = self.ps_rref.rpc_sync().get_model().cuda()  # 同步获取模型并移动到GPU
        for inputs, labels in self.get_next_batch():  # 遍历每个批次
            timed_log(f"{name} processing one batch")  # 打印处理批次的信息
            self.loss_fn(m(inputs), labels).backward()  # 计算损失并反向传播
            timed_log(f"{name} reporting grads")  # 打印报告梯度的信息
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),  # 发送参数的梯度
            ).cuda()  # 同步获取更新后的模型并移动到GPU
            timed_log(f"{name} got updated model")  # 打印获取更新模型的信息


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)  # 创建Trainer实例
    trainer.train()  # 开始训练


def run_ps(trainers):
    timed_log("Start training")  # 打印开始训练的信息
    ps_rref = rpc.RRef(BatchUpdateParameterServer())  # 创建参数服务器的RRef
    futs = []  # 初始化未来对象列表
    for trainer in trainers:  # 遍历所有训练者
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))  # 异步调用训练者的训练函数
        )

    torch.futures.wait_all(futs)  # 等待所有未来对象完成
    timed_log("Finish training")  # 打印完成训练的信息


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '29500'  # 设置主节点端口
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,  # 设置工作线程数量
        rpc_timeout=0  # 设置RPC超时为无限
     )
    if rank != 0:  # 如果不是参数服务器
        rpc.init_rpc(
            f"trainer{rank}",  # 初始化训练者的RPC
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
        # 训练者被动等待参数服务器启动训练迭代
    else:  # 如果是参数服务器
        rpc.init_rpc(
            "ps",  # 初始化参数服务器的RPC
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)])  # 启动训练

    # block until all rpcs finish
    rpc.shutdown()  # 关闭RPC


if __name__=="__main__":
    world_size = batch_update_size + 1  # 计算世界规模
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)  # 启动多个进程