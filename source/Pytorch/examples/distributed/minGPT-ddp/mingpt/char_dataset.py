import torch
from torch.utils.data import Dataset
import fsspec
from dataclasses import dataclass

"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""  # 从 https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py 修改而来

@dataclass
class DataConfig:
    path: str = None  # 数据文件的路径
    block_size: int = None  # 每个数据块的大小
    train_split: float = None  # 训练集的比例
    truncate: float = 1.0  # 数据截断比例，默认值为1.0

class CharDataset(Dataset):
    def __init__(self, data_cfg: DataConfig): #data_path: str, block_size):
        data = fsspec.open(data_cfg.path).open().read().decode('utf-8')  # 打开指定路径的文件并读取数据，解码为UTF-8
        data = data[ : int(len(data) * data_cfg.truncate)]  # 根据截断比例截取数据

        chars = sorted(list(set(data)))  # 获取数据中唯一字符并排序
        data_size, vocab_size = len(data), len(chars)  # 计算数据大小和词汇表大小
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))  # 打印数据字符数和唯一字符数

        self.stoi = {ch: i for i, ch in enumerate(chars)}  # 创建字符到索引的映射（从字符到整数）
        self.itos = {i: ch for i, ch in enumerate(chars)}  # 创建索引到字符的映射（从整数到字符）
        self.block_size = data_cfg.block_size  # 设置块大小
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.data = data  # 存储数据

    def __len__(self):
        return len(self.data) - self.block_size  # 返回数据集的长度，减去块大小

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]  # 从数据中获取一块（块大小 + 1）的字符
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]  # 将每个字符编码为整数
        x = torch.tensor(dix[:-1], dtype=torch.long)  # 创建输入张量，包含块中的所有字符（除最后一个）
        y = torch.tensor(dix[1:], dtype=torch.long)  # 创建目标张量，包含块中的所有字符（除第一个）
        return x, y  # 返回输入和目标张量