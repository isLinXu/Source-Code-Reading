import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset  # 导入自定义数据集

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id  # 当前GPU的ID
        self.model = model.to(gpu_id)  # 将模型移动到指定的GPU
        self.train_data = train_data  # 训练数据加载器
        self.optimizer = optimizer  # 优化器
        self.save_every = save_every  # 每多少个epoch保存一次检查点

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()  # 清零梯度
        output = self.model(source)  # 前向传播
        loss = F.cross_entropy(output, targets)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])  # 获取当前批次的大小
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")  # 打印当前epoch信息
        for source, targets in self.train_data:  # 遍历训练数据
            source = source.to(self.gpu_id)  # 将输入数据移动到指定的GPU
            targets = targets.to(self.gpu_id)  # 将目标数据移动到指定的GPU
            self._run_batch(source, targets)  # 运行一个批次的训练

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()  # 获取模型的状态字典
        PATH = "checkpoint.pt"  # 定义保存路径
        torch.save(ckp, PATH)  # 保存模型的状态
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")  # 打印保存信息

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):  # 遍历每个epoch
            self._run_epoch(epoch)  # 运行当前epoch的训练
            if epoch % self.save_every == 0:  # 如果满足保存条件
                self._save_checkpoint(epoch)  # 保存检查点

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
        shuffle=True  # 打乱数据
    )

def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()  # 加载训练对象
    train_data = prepare_dataloader(dataset, batch_size)  # 准备数据加载器
    trainer = Trainer(model, train_data, optimizer, device, save_every)  # 创建训练器
    trainer.train(total_epochs)  # 开始训练

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')  # 创建参数解析器
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')  # 总训练epoch数
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')  # 保存快照的频率
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')  # 每个设备的输入批次大小
    args = parser.parse_args()  # 解析参数
    
    device = 0  # shorthand for cuda:0  # 设置设备为GPU 0
    main(device, args.total_epochs, args.save_every, args.batch_size)  # 运行主函数