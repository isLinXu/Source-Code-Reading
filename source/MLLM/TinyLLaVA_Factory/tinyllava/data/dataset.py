import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset



ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    """一个懒加载的监督数据集类，用于处理文本和图像数据。"""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        """
        初始化 LazySupervisedDataset 实例。

        :param data_path: 包含数据的 JSON 文件路径。
        :param tokenizer: 用于文本分词的预训练分词器。
        :param data_args: 包含数据处理参数的 DataArguments 实例。
        """
        super(LazySupervisedDataset, self).__init__()
        # 从 JSON 文件加载数据列表
        list_data_dict = json.load(open(data_path, "r"))

        # 初始化类的属性
        self.tokenizer = tokenizer                                                      # 分词器
        self.list_data_dict = list_data_dict                                            # 数据列表
        self.data_args = data_args                                                      # 数据处理参数
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)        # 文本预处理实例
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)   # 图像预处理实例

    def __len__(self):
        """
        重写 len() 方法，用于获取列表数据字典的长度。

        返回:
            int: 列表数据字典的长度。
        """
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """
        计算并返回每个多模态样本的长度列表。

        遍历内部的多模态对话样本，计算每个样本的长度，包括文本对话和图像的token数量。
        对于每个样本，其长度由文本对话的单词数量和图像token数量（如果存在图像）组成。

        Returns:
            length_list (List[int]): 包含每个样本长度的列表。
        """
        length_list = []
        for sample in self.list_data_dict:
            # 如果样本包含图像信息，初始化图像token数量为128，否则为0
            img_tokens = 128 if 'image' in sample else 0
            # 计算样本中所有对话的单词数量总和，并加上图像token数量（如果有的话），然后添加到长度列表中
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        """
        计算并返回每个样本的长度，区分文本和含图像的样本。

        遍历self.list_data_dict中的每个样本，计算样本中所有对话的词数总和。
        如果样本包含'image'字段，则将其长度记为负数，以区分含有图像的样本。

        :return: 包含所有样本长度的列表，长度可能为负数表示含有图像。
        """
        # 初始化长度列表，用于存储每个样本的长度
        length_list = []
        # 遍历数据集中的每个样本
        for sample in self.list_data_dict:
            # 计算当前样本中所有对话的词数总和
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # 如果样本包含'image'字段，将其长度记为负数
            cur_len = cur_len if 'image' in sample else -cur_len
            # 将当前样本的长度添加到列表
            length_list.append(cur_len)
        # 返回包含所有样本长度的列表
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        重载索引操作，以获取数据集中特定位置的数据。

        参数:
        i (int): 数据的索引位置。

        返回:
        Dict[str, torch.Tensor]: 一个字典，包含数据样本的不同模态内容，如文本和图像。
        """
        # 获取给定索引的数据源字典
        sources = self.list_data_dict[i]
        # 对对话文本进行文本预处理
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        # 如果数据源中包含图像信息，则加载并预处理图像
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            # 从指定路径打开图像，并转换为RGB格式
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            # 对图像进行预处理
            image = self.image_preprocess(image)
            # 将预处理后的图像添加到数据字典
            data_dict['image'] = image
        # 如果数据源中不包含图像信息，但模型是多模态的
        elif self.data_args.is_multimodal:
            # 打印调试信息，显示缺少图像数据的索引和数据源
            # print(f'{i}:{sources}')
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            # 获取图像裁剪大小
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            # 创建一个零张量，用于表示缺失的图像数据，并赋予正确的尺寸
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        # 返回包含文本和/或图像数据的数据字典
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    """
    用于监督训练数据集的批量处理类。

    该类主要负责对训练样本进行批量处理，以便进行监督微调。它的工作包括：
    - 接收一系列实例并将其整理为字典形式的批量数据。
    - 对输入的实例进行填充，以确保所有实例具有相同的长度。
    - 根据需要处理特殊的填充和结束标记。

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): 用于文本编码和解码的预训练分词器。
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        对实例进行批量处理。

        该方法接收一系列字典形式的实例，每个实例包含'input_ids'和'labels'，
        并返回一个字典，包含转换后的'input_ids'、'labels'和'attention_mask'，
        以及在特定条件下的图像数据。

        Parameters:
            instances (Sequence[Dict]): 包含'input_ids'和'labels'的实例序列。

        Returns:
            Dict[str, torch.Tensor]: 包含批量数据的字典，包括'input_ids'、'labels'、
            'attention_mask'，以及可能的图像数据。
        """
        # 提取每个实例的'input_ids'和'labels'
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # 处理特殊情况，当填充标记和结束标记相同时，临时替换结束标记
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        # 对输入序列进行填充
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        # 对标签序列进行填充
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        # 修剪输入序列和标签序列，确保它们不超过模型的最大长度
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        # 反向处理特殊情况，将临时替换的结束标记改回原样
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        # 构建并返回包含所有必要信息的字典
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        # 如果实例包含图像数据，则处理图像数据
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # 如果所有图像的形状相同，则将它们堆叠成一个张量
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                # 如果图像形状不同，则保持它们为原始形式
                batch['images'] = images
        # 返回包含所有数据的字典
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    """
    准备监督微调的数据模块.

    该函数创建一个监督学习的数据集和数据整合器，用于模型的训练.

    参数:
    - tokenizer: 预训练的分词器，用于文本的编码和解码.
    - data_args: 数据相关的参数，包括数据路径等配置.

    返回:
    一个字典，包含训练数据集、评估数据集（此处为None）和数据整合器.
    """
    # 创建训练数据集，使用LazySupervisedDataset以便于大规模数据的高效加载
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    # 创建数据整合器，用于将数据集中的样本整合成批次
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # 返回包含训练数据集、评估数据集和数据整合器的字典
    # 评估数据集此处设置为None，因为该函数主要针对的是监督学习的场景
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
