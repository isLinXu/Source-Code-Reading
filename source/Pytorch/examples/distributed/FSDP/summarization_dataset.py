import argparse  # 导入命令行参数解析模块
import glob  # 导入用于文件路径操作的模块
import os  # 导入操作系统模块
import json  # 导入JSON处理模块
import time  # 导入时间模块
import logging  # 导入日志模块
import random  # 导入随机数模块
import re  # 导入正则表达式模块
from itertools import chain  # 从itertools模块导入链式操作
from string import punctuation  # 从字符串模块导入标点符号

import pandas as pd  # 导入Pandas库
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
from torch.utils.data import Dataset, DataLoader  # 从PyTorch导入数据集和数据加载器

from nlp import load_dataset  # 从nlp库导入加载数据集的函数

from transformers import (  # 从transformers库导入所需组件
    AdamW,  # 导入AdamW优化器
    T5ForConditionalGeneration,  # 导入T5条件生成模型
    T5Tokenizer,  # 导入T5分词器
    get_linear_schedule_with_warmup  # 导入线性学习率调度器
)

class wikihow(Dataset):  # 定义wikihow类，继承自Dataset
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):  # 初始化函数
        self.dataset = load_dataset('wikihow', 'all', data_dir='data/', split=type_path)  # 加载wikihow数据集
        if num_samples:  # 如果指定了样本数量
            self.dataset = self.dataset.select(list(range(0, num_samples)))  # 选择指定数量的样本
        self.input_length = input_length  # 设置输入长度
        self.tokenizer = tokenizer  # 设置分词器
        self.output_length = output_length  # 设置输出长度
        self.print_text = print_text  # 设置是否打印文本

    def __len__(self):  # 定义获取数据集长度的方法
        return self.dataset.shape[0]  # 返回数据集的样本数量

    def clean_text(self, text):  # 定义清理文本的方法
        text = text.replace('Example of text:', '')  # 移除特定文本
        text = text.replace('Example of Summary:', '')  # 移除特定文本
        text = text.replace('\n', '')  # 移除换行符
        text = text.replace('``', '')  # 移除反引号
        text = text.replace('"', '')  # 移除双引号

        return text  # 返回清理后的文本

    def convert_to_features(self, example_batch):  # 定义将示例批次转换为特征的方法
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:  # 如果需要打印文本
            print("Input Text: ", self.clean_text(example_batch['text']))  # 打印输入文本
#         input_ = self.clean_text(example_batch['text']) + " </s>"  # 清理输入文本并添加结束符
#         target_ = self.clean_text(example_batch['headline']) + " </s>"  # 清理目标文本并添加结束符

        input_ = self.clean_text(example_batch['text'])  # 清理输入文本
        target_ = self.clean_text(example_batch['headline'])  # 清理目标文本

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,  # 对输入文本进行分词
                                                     padding='max_length', truncation=True, return_tensors="pt")  # 填充到最大长度，截断，并返回PyTorch张量

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,  # 对目标文本进行分词
                                                     padding='max_length', truncation=True, return_tensors="pt")  # 填充到最大长度，截断，并返回PyTorch张量

        return source, targets  # 返回分词后的输入和目标

    def __getitem__(self, index):  # 定义获取单个样本的方法
        source, targets = self.convert_to_features(self.dataset[index])  # 获取样本的特征

        source_ids = source["input_ids"].squeeze()  # 获取输入ID并去除多余维度
        target_ids = targets["input_ids"].squeeze()  # 获取目标ID并去除多余维度

        src_mask = source["attention_mask"].squeeze()  # 获取输入的注意力掩码并去除多余维度
        target_mask = targets["attention_mask"].squeeze()  # 获取目标的注意力掩码并去除多余维度

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}  # 返回输入和目标的ID及掩码

def get_dataset(tokenizer, type_path, num_samples, args):  # 定义获取数据集的函数
      return wikihow(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples, input_length=max_input_length,  # 返回wikihow数据集实例
                        output_length=max_output_length)