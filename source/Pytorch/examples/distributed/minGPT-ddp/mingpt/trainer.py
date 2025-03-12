"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""  # 简单的训练循环；可以应用于任何任意神经网络的模板，因此这个文件中的内容与GPT没有具体关系。

from dataclasses import dataclass, asdict  # 从dataclasses模块导入dataclass装饰器和asdict函数
from collections import OrderedDict  # 从collections模块导入OrderedDict
from typing import Optional, Any, Dict  # 导入类型提示
import os  # 导入操作系统相关的模块

import torch  # 导入PyTorch库
from torch.utils.data import Dataset, DataLoader  # 从PyTorch的工具库导入Dataset和DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块
from torch.utils.data.distributed import DistributedSampler  # 导入分布式采样器

import boto3  # 导入boto3库，用于与AWS S3交互
from urllib.parse import urlparse  # 导入urlparse函数，用于解析URL
import fsspec  # 导入fsspec库，用于文件系统操作
import io  # 导入io模块，用于处理输入输出

@dataclass
class TrainerConfig:
    max_epochs: int = None  # 最大训练轮数
    batch_size: int = None  # 批量大小
    data_loader_workers: int = None  # 数据加载器的工作线程数
    grad_norm_clip: float = None  # 梯度裁剪的阈值
    snapshot_path: Optional[str] = None  # 快照保存路径
    save_every: int = None  # 每多少轮保存一次快照
    use_amp: bool = None  # 是否使用自动混合精度

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'  # 模型状态
    optimizer_state: Dict[str, Any]  # 优化器状态
    finished_epoch: int  # 完成的轮数

def upload_to_s3(obj, dst):
    buffer = io.BytesIO()  # 创建一个字节流缓冲区
    torch.save(obj, buffer)  # 将对象保存到缓冲区
    buffer.seek(0)  # 将缓冲区指针移动到开始位置
    dst = urlparse(dst, allow_fragments=False)  # 解析目标URL
    boto3.client('s3').upload_fileobj(buffer, dst.netloc, dst.path.lstrip('/'))  # 上传文件到S3

class Trainer:
    def __init__(self, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config  # 保存训练配置
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地进程的rank
        self.global_rank = int(os.environ["RANK"])  # 获取全局进程的rank  
        # data stuff
        self.train_dataset = train_dataset  # 保存训练数据集
        self.train_loader = self._prepare_dataloader(train_dataset)  # 准备训练数据加载器
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None  # 准备测试数据加载器（如果有的话）
        # initialize train states
        self.epochs_run = 0  # 已训练的轮数
        self.model = model.to(self.local_rank)  # 将模型移动到本地设备
        self.optimizer = optimizer  # 保存优化器        
        self.save_every = self.config.save_every  # 保存快照的频率
        if self.config.use_amp:  # 如果使用自动混合精度
            self.scaler = torch.cuda.amp.GradScaler()  # 初始化GradScaler
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:  # 如果没有指定快照路径
            self.config.snapshot_path = "snapshot.pt"  # 默认快照路径
        self._load_snapshot()  # 加载快照
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])  # 使用DDP包装模型以同步所有进程

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(  # 创建数据加载器
            dataset,
            batch_size=self.config.batch_size,  # 设置批量大小
            pin_memory=True,  # 将数据加载到固定内存中
            shuffle=False,  # 不打乱数据
            num_workers=self.config.data_loader_workers,  # 设置工作线程数
            sampler=DistributedSampler(dataset)  # 使用分布式采样器
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)  # 打开快照文件
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")  # 加载快照数据
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")  # 如果找不到快照，打印信息
            return 

        snapshot = Snapshot(**snapshot_data)  # 创建快照实例
        self.model.load_state_dict(snapshot.model_state)  # 加载模型状态
        self.optimizer.load_state_dict(snapshot.optimizer_state)  # 加载优化器状态
        self.epochs_run = snapshot.finished_epoch  # 更新已训练的轮数
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")  # 打印恢复训练的信息

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.config.use_amp)):  # 设置梯度计算和自动混合精度
            _, loss = self.model(source, targets)  # 前向传播，计算损失
        
        if train:  # 如果是训练模式
            self.optimizer.zero_grad(set_to_none=True)  # 清零梯度
            if self.config.use_amp:  # 如果使用自动混合精度 
                self.scaler.scale(loss).backward()  # 缩放损失并反向传播
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)  # 梯度裁剪
                self.scaler.step(self.optimizer)  # 更新优化器
                self.scaler.update()  # 更新缩放器
            else:
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)  # 梯度裁剪
                self.optimizer.step()  # 更新优化器
        
        return loss.item()  # 返回损失值

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)  # 设置采样器的当前轮数
        for iter, (source, targets) in enumerate(dataloader):  # 遍历数据加载器
            step_type = "Train" if train else "Eval"  # 判断当前是训练还是评估
            source = source.to(self.local_rank)  # 将输入数据移动到本地设备
            targets = targets.to(self.local_rank)  # 将目标数据移动到本地设备
            batch_loss = self._run_batch(source, targets, train)  # 运行一个批次的训练或评估
            if iter % 100 == 0:  # 每100个迭代打印一次损失
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss:.5f}")  # 打印损失信息

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model  # 获取模型
        raw_model = model.module if hasattr(model, "module") else model  # 获取原始模型
        snapshot = Snapshot(  # 创建快照实例
            model_state=raw_model.state_dict(),  # 模型状态
            optimizer_state=self.optimizer.state_dict(),  # 优化器状态
            finished_epoch=epoch  # 完成的轮数
        )
        # save snapshot
        snapshot = asdict(snapshot)  # 将快照转换为字典
        if self.config.snapshot_path.startswith("s3://"):  # 如果快照路径是S3
            upload_to_s3(snapshot, self.config.snapshot_path)  # 上传快照到S3
        else:
            torch.save(snapshot, self.config.snapshot_path)  # 保存快照到本地
            
        print(f"Snapshot saved at epoch {epoch}")  # 打印快照保存信息

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):  # 遍历每一轮
            epoch += 1  # 轮数加1
            self._run_epoch(epoch, self.train_loader, train=True)  # 运行训练轮
            if self.local_rank == 0 and epoch % self.save_every == 0:  # 如果是主进程并且达到保存频率
                self._save_snapshot(epoch)  # 保存快照
            # eval run
            if self.test_loader:  # 如果有测试数据加载器
                self._run_epoch(epoch, self.test_loader, train=False)  # 运行评估轮