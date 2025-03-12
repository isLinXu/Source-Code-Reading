import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # 设置当前进程使用的GPU设备
    init_process_group(backend="nccl")  # 初始化进程组，使用NCCL后端

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])  # 获取当前进程的本地排名
        self.global_rank = int(os.environ["RANK"])  # 获取当前进程的全局排名
        self.model = model.to(self.local_rank)  # 将模型移动到指定的GPU
        self.train_data = train_data  # 训练数据加载器
        self.optimizer = optimizer  # 优化器
        self.save_every = save_every  # 每多少个epoch保存一次快照
        self.epochs_run = 0  # 已运行的epoch数
        self.snapshot_path = snapshot_path  # 快照文件路径
        if os.path.exists(snapshot_path):  # 检查快照文件是否存在
            print("Loading snapshot")  # 打印加载快照信息
            self._load_snapshot(snapshot_path)  # 加载快照

        self.model = DDP(self.model, device_ids=[self.local_rank])  # 使用分布式数据并行包装模型

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"  # 获取当前进程的CUDA设备
        snapshot = torch.load(snapshot_path, map_location=loc)  # 加载快照
        self.model.load_state_dict(snapshot["MODEL_STATE"])  # 加载模型状态
        self.epochs_run = snapshot["EPOCHS_RUN"]  # 获取已运行的epoch数
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")  # 打印恢复训练信息

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()  # 清零梯度
        output = self.model(source)  # 前向传播
        loss = F.cross_entropy(output, targets)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])  # 获取当前批次的大小
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")  # 打印当前epoch信息
        self.train_data.sampler.set_epoch(epoch)  # 设置当前epoch的采样器
        for source, targets in self.train_data:  # 遍历训练数据
            source = source.to(self.local_rank)  # 将输入数据移动到指定的GPU
            targets = targets.to(self.local_rank)  # 将目标数据移动到指定的GPU
            self._run_batch(source, targets)  # 运行一个批次的训练

    def _save_snapshot(self, epoch):
        snapshot = {  # 创建快照字典
            "MODEL_STATE": self.model.module.state_dict(),  # 模型状态
            "EPOCHS_RUN": epoch,  # 当前epoch数
        }
        torch.save(snapshot, self.snapshot_path)  # 保存快照
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")  # 打印保存信息

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):  # 遍历每个epoch
            self._run_epoch(epoch)  # 运行当前epoch的训练
            if self.local_rank == 0 and epoch % self.save_every == 0:  # 如果是主进程并且满足保存条件
                self._save_snapshot(epoch)  # 保存快照

def load_train_objs():
    train_set = MyTrainDataset(2048)  # 加载训练数据集
    model = torch.nn.Linear(20, 1)  # 定义模型
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器
    return train_set, model, optimizer  # 返回训练集、模型和优化器

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,  # 设置批次大小
        pin_memory=True,  # 将数据加载到固定内存中
        shuffle=False,  # 不打乱数据
        sampler=DistributedSampler(dataset)  # 使用分布式采样器
    )

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()  # 设置分布式训练环境
    dataset, model, optimizer = load_train_objs()  # 加载训练对象
    train_data = prepare_dataloader(dataset, batch_size)  # 准备数据加载器
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)  # 创建训练器
    trainer.train(total_epochs)  # 开始训练
    destroy_process_group()  # 销毁进程组

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')  # 创建参数解析器
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')  # 总训练epoch数
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')  # 保存快照的频率
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')  # 每个设备的输入批次大小
    args = parser.parse_args()  # 解析参数

    main(args.save_every, args.total_epochs, args.batch_size)  # 运行主函数