import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from torch.utils.data import Dataset, DataLoader  # 从 torch.utils.data 导入 Dataset 和 DataLoader
from datautils import MyTrainDataset  # 导入自定义的数据集类 MyTrainDataset

import torch.multiprocessing as mp  # 导入多进程模块
from torch.utils.data.distributed import DistributedSampler  # 从 torch.utils.data.distributed 导入分布式采样器
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块
from torch.distributed import init_process_group, destroy_process_group  # 导入初始化和销毁进程组的函数
import os  # 导入 os 库，用于与操作系统交互


def ddp_setup():
    """Set up the distributed data parallel environment.
    设置分布式数据并行环境。
    """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # 设置当前进程使用的 GPU
    init_process_group(backend="nccl")  # 初始化进程组，使用 NCCL 后端


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,  # 模型
        train_data: DataLoader,  # 训练数据加载器
        optimizer: torch.optim.Optimizer,  # 优化器
        save_every: int,  # 保存快照的频率
        snapshot_path: str,  # 快照文件路径
    ) -> None:
        """Initialize the Trainer.
        初始化 Trainer。
        
        Args:
            model: 要训练的模型
            train_data: 训练数据加载器
            optimizer: 优化器
            save_every: 保存快照的频率
            snapshot_path: 快照文件路径
        """
        self.gpu_id = int(os.environ["LOCAL_RANK"])  # 获取当前进程的 GPU ID
        self.model = model.to(self.gpu_id)  # 将模型移动到指定的 GPU
        self.train_data = train_data  # 保存训练数据加载器
        self.optimizer = optimizer  # 保存优化器
        self.save_every = save_every  # 保存快照的频率
        self.epochs_run = 0  # 已运行的 epochs 数
        self.snapshot_path = snapshot_path  # 快照文件路径
        if os.path.exists(snapshot_path):  # 如果快照文件存在
            print("Loading snapshot")  # 打印加载快照信息
            self._load_snapshot(snapshot_path)  # 加载快照

        self.model = DDP(self.model, device_ids=[self.gpu_id])  # 使用 DDP 封装模型以支持分布式训练

    def _load_snapshot(self, snapshot_path):
        """Load the model snapshot.
        加载模型快照。
        
        Args:
            snapshot_path: 快照文件路径
        """
        loc = f"cuda:{self.gpu_id}"  # 设置加载模型的设备
        snapshot = torch.load(snapshot_path, map_location=loc)  # 加载快照
        self.model.load_state_dict(snapshot["MODEL_STATE"])  # 加载模型状态
        self.epochs_run = snapshot["EPOCHS_RUN"]  # 恢复已运行的 epochs 数
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")  # 打印恢复信息

    def _run_batch(self, source, targets):
        """Run a single batch of training.
        运行单个训练批次。
        
        Args:
            source: 输入数据
            targets: 目标标签
        """
        self.optimizer.zero_grad()  # 清除梯度
        output = self.model(source)  # 通过模型计算输出
        loss = F.cross_entropy(output, targets)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新优化器

    def _run_epoch(self, epoch):
        """Run a single epoch of training.
        运行单个训练 epoch。
        
        Args:
            epoch: 当前 epoch 的编号
        """
        b_sz = len(next(iter(self.train_data))[0])  # 获取批次大小
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")  # 打印当前 epoch 信息
        self.train_data.sampler.set_epoch(epoch)  # 设置采样器的 epoch
        for source, targets in self.train_data:  # 遍历训练数据
            source = source.to(self.gpu_id)  # 将输入数据移动到 GPU
            targets = targets.to(self.gpu_id)  # 将目标标签移动到 GPU
            self._run_batch(source, targets)  # 运行批次训练

    def _save_snapshot(self, epoch):
        """Save the model snapshot.
        保存模型快照。
        
        Args:
            epoch: 当前 epoch 的编号
        """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),  # 保存模型状态
            "EPOCHS_RUN": epoch,  # 保存已运行的 epochs 数
        }
        torch.save(snapshot, self.snapshot_path)  # 保存快照
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")  # 打印保存快照信息

    def train(self, max_epochs: int):
        """Train the model for a specified number of epochs.
        训练模型指定的 epoch 数。
        
        Args:
            max_epochs: 最大训练 epochs 数
        """
        for epoch in range(self.epochs_run, max_epochs):  # 从已运行的 epochs 开始训练
            self._run_epoch(epoch)  # 运行当前 epoch
            if self.gpu_id == 0 and epoch % self.save_every == 0:  # 只有 rank 0 进程保存快照
                self._save_snapshot(epoch)  # 保存快照


def load_train_objs():
    """Load the training objects.
    加载训练对象。
    
    Returns:
        训练数据集、模型和优化器
    """
    train_set = MyTrainDataset(2048)  # 加载数据集
    model = torch.nn.Linear(20, 1)  # 加载模型
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器
    return train_set, model, optimizer  # 返回数据集、模型和优化器


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """Prepare the DataLoader for distributed training.
    为分布式训练准备 DataLoader。
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
    
    Returns:
        DataLoader 对象
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,  # 设置批次大小
        pin_memory=True,  # 将数据加载到固定内存
        shuffle=False,  # 不打乱数据
        sampler=DistributedSampler(dataset)  # 使用分布式采样器
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    """Main function to run the training.
    运行训练的主函数。
    
    Args:
        save_every: 保存快照的频率
        total_epochs: 总训练 epochs 数
        batch_size: 批次大小
        snapshot_path: 快照文件路径
    """
    ddp_setup()  # 设置分布式环境
    dataset, model, optimizer = load_train_objs()  # 加载训练对象
    train_data = prepare_dataloader(dataset, batch_size)  # 准备数据加载器
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)  # 创建 Trainer 实例
    trainer.train(total_epochs)  # 开始训练
    destroy_process_group()  # 销毁进程组


if __name__ == "__main__":
    import argparse  # 导入 argparse 库用于解析命令行参数
    parser = argparse.ArgumentParser(description='simple distributed training job')  # 创建参数解析器
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')  # 添加总训练 epochs 参数
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')  # 添加保存快照频率参数
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')  # 添加批次大小参数
    args = parser.parse_args()  # 解析命令行参数

    main(args.save_every, args.total_epochs, args.batch_size)  # 调用主函数