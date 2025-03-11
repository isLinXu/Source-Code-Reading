import torch  # 导入 PyTorch 库
import os  # 导入操作系统库


class Dictionary(object):  # 定义字典类
    def __init__(self):  # 初始化方法
        self.word2idx = {}  # 创建一个空字典，用于存储单词到索引的映射
        self.idx2word = {}  # 创建一个空字典，用于存储索引到单词的映射
        self.idx = 0  # 初始化索引为 0
    
    def add_word(self, word):  # 添加单词的方法
        if not word in self.word2idx:  # 如果单词不在字典中
            self.word2idx[word] = self.idx  # 将单词和当前索引添加到字典
            self.idx2word[self.idx] = word  # 将当前索引和单词添加到反向字典
            self.idx += 1  # 索引加 1
    
    def __len__(self):  # 获取字典长度的方法
        return len(self.word2idx)  # 返回单词到索引映射的长度


class Corpus(object):  # 定义语料库类
    def __init__(self):  # 初始化方法
        self.dictionary = Dictionary()  # 创建字典实例

    def get_data(self, path, batch_size=20):  # 获取数据的方法，path 为文件路径，batch_size 为批次大小
        # Add words to the dictionary
        with open(path, 'r') as f:  # 打开文件进行读取
            tokens = 0  # 初始化 token 数量
            for line in f:  # 遍历文件中的每一行
                words = line.split() + ['<eos>']  # 将行分割成单词，并在末尾添加结束符
                tokens += len(words)  # 更新 token 数量
                for word in words:  # 遍历每个单词
                    self.dictionary.add_word(word)  # 将单词添加到字典中
        
        # Tokenize the file content
        ids = torch.LongTensor(tokens)  # 创建一个长整型张量，用于存储 token 的索引
        token = 0  # 初始化 token 索引
        with open(path, 'r') as f:  # 再次打开文件进行读取
            for line in f:  # 遍历文件中的每一行
                words = line.split() + ['<eos>']  # 将行分割成单词，并在末尾添加结束符
                for word in words:  # 遍历每个单词
                    ids[token] = self.dictionary.word2idx[word]  # 将单词的索引存入 ids 张量
                    token += 1  # 更新 token 索引
        num_batches = ids.size(0) // batch_size  # 计算批次数
        ids = ids[:num_batches*batch_size]  # 只保留完整批次的数据
        return ids.view(batch_size, -1)  # 将 ids 张量重塑为 (batch_size, -1) 的形状