import os  # 导入操作系统相关的模块
import torch  # 导入PyTorch库
from torch.utils.data import random_split  # 从PyTorch的工具库导入random_split函数
from torch.distributed import init_process_group, destroy_process_group  # 导入分布式训练相关的函数
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer  # 从model模块导入GPT模型及其配置和优化器创建函数
from trainer import Trainer, TrainerConfig  # 从trainer模块导入训练器及其配置
from char_dataset import CharDataset, DataConfig  # 从char_dataset模块导入字符数据集及其配置
from omegaconf import DictConfig  # 从OmegaConf库导入字典配置类
import hydra  # 导入Hydra库，用于配置管理

def ddp_setup():
    init_process_group(backend="nccl")  # 初始化分布式进程组，使用NCCL后端
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # 设置当前CUDA设备为本地进程的设备

def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    dataset = CharDataset(data_cfg)  # 创建字符数据集实例
    train_len = int(len(dataset) * data_cfg.train_split)  # 计算训练集的长度
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])  # 将数据集随机分割为训练集和测试集

    gpt_cfg.vocab_size = dataset.vocab_size  # 设置GPT配置中的词汇表大小
    gpt_cfg.block_size = dataset.block_size  # 设置GPT配置中的块大小
    model = GPT(gpt_cfg)  # 创建GPT模型实例
    optimizer = create_optimizer(model, opt_cfg)  # 创建优化器实例
    
    return model, optimizer, train_set, test_set  # 返回模型、优化器、训练集和测试集

@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    ddp_setup()  # 设置分布式训练环境

    gpt_cfg = GPTConfig(**cfg['gpt_config'])  # 从配置中加载GPT配置
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])  # 从配置中加载优化器配置
    data_cfg = DataConfig(**cfg['data_config'])  # 从配置中加载数据配置
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])  # 从配置中加载训练器配置

    model, optimizer, train_data, test_data = get_train_objs(gpt_cfg, opt_cfg, data_cfg)  # 获取模型、优化器、训练数据和测试数据
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)  # 创建训练器实例
    trainer.train()  # 开始训练

    destroy_process_group()  # 销毁分布式进程组

if __name__ == "__main__":
    main()  # 如果是主模块，运行main函数