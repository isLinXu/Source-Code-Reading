import os  # 导入操作系统库
from io import open  # 从 io 模块导入 open 函数
import torch  # 导入 PyTorch 库

class Dictionary(object):  # 定义 Dictionary 类
    def __init__(self):  # 初始化方法
        self.word2idx = {}  # 创建一个空字典，用于存储单词到索引的映射
        self.idx2word = []  # 创建一个空列表，用于存储索引到单词的映射

    def add_word(self, word):  # 添加单词的方法
        if word not in self.word2idx:  # 如果单词不在字典中
            self.idx2word.append(word)  # 将单词添加到 idx2word 列表中
            self.word2idx[word] = len(self.idx2word) - 1  # 将单词映射到其索引
        return self.word2idx[word]  # 返回单词的索引

    def __len__(self):  # 定义获取字典长度的方法
        return len(self.idx2word)  # 返回单词数量


class Corpus(object):  # 定义 Corpus 类
    def __init__(self, path):  # 初始化方法，接受路径作为参数
        self.dictionary = Dictionary()  # 创建 Dictionary 实例
        self.train = self.tokenize(os.path.join(path, 'train.txt'))  # 对训练集进行分词
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))  # 对验证集进行分词
        self.test = self.tokenize(os.path.join(path, 'test.txt'))  # 对测试集进行分词

    def tokenize(self, path):  # 对文本文件进行分词的方法
        """Tokenizes a text file."""  # 记录文本文件的分词
        assert os.path.exists(path)  # 确保文件存在
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:  # 以 UTF-8 编码打开文件
            for line in f:  # 遍历文件中的每一行
                words = line.split() + ['<eos>']  # 将行分割为单词，并添加结束符
                for word in words:  # 遍历每个单词
                    self.dictionary.add_word(word)  # 将单词添加到字典中

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:  # 以 UTF-8 编码再次打开文件
            idss = []  # 初始化一个空列表，用于存储索引
            for line in f:  # 遍历文件中的每一行
                words = line.split() + ['<eos>']  # 将行分割为单词，并添加结束符
                ids = []  # 初始化一个空列表，用于存储单词的索引
                for word in words:  # 遍历每个单词
                    ids.append(self.dictionary.word2idx[word])  # 将单词的索引添加到列表中
                idss.append(torch.tensor(ids).type(torch.int64))  # 将索引列表转换为 PyTorch 张量并添加到 idss 列表中
            ids = torch.cat(idss)  # 将所有张量连接成一个张量

        return ids  # 返回最终的索引张量