from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy

from .formatter import EmptyFormatter, StringFormatter
from .formatter import Formatter
from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
    

# 定义了五个Formatter类型的属性，分别用于格式化图片token、用户消息、助手消息、系统和分隔符
@dataclass
class Template:
    format_image_token: "Formatter"
    format_user: "Formatter"
    format_assistant: "Formatter"
    system: "Formatter"
    separator: "Formatter"
    
    def encode(self, messages, tokenizer, mode='train'):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target

        1. 将messages（对话列表：[{from:human, value:message}, {from:gpt, value:message}]）转换为human_list和value_list
        2. 构造两个列表作为提示
        3. 对提示进行tokenize
        4. 制作目标标签

        参数:
        messages -- 对话列表，包含人类和GPT的消息
        tokenizer -- 用于tokenize的工具
        mode -- 模式，'train'或非'train'

        返回:
        根据模式返回不同的字典，包含input_ids和可能的labels
        """
        # 从messages中提取human_list和value_list
        question_list, answer_list = self.get_list_from_message(messages)
        # 构造提示
        prompt = self.prompt(question_list, answer_list)
        # 对提示进行tokenize并返回input_ids
        input_ids = self.tokenizer_image_token(prompt, tokenizer, return_tensors='pt')

        # 如果是训练模式，制作labels并返回包含input_ids和labels的字典
        if mode == 'train':
            labels = self.make_labels(input_ids, prompt, tokenizer)
            return dict(
                input_ids=input_ids,
                labels=labels
            )
        else:
            # 如果不是训练模式，返回包含input_ids和prompt的字典
            return dict(input_ids=input_ids, prompt=prompt)
        
    
    def get_list_from_message(self, messages):
        """
        处理消息列表，调用私有方法_get_list_from_message来获取问题和答案的列表。

        :param messages: 包含消息的列表，每个消息是一个字典，包含发送者(from)和消息内容(value)
        :return: 返回一个元组，包含两个列表，第一个是问题列表，第二个是答案列表
        """
        return self._get_list_from_message(messages)
    
    def _get_list_from_message(self, messages):
        """
        messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
        私有方法，用于从消息列表中提取问题和答案。

        :param messages: 包含消息的列表，格式为[{from:human, value:message}, {from:gpt, value:message}]
        :return: 返回一个元组，包含两个列表，第一个是问题列表，第二个是答案列表
        """
        question_list = []                              # 存储问题的列表
        answer_list = []                                # 存储答案的列表
        first_is_not_question = 0                       # 标记第一条消息是否不是问题

        # 遍历消息列表
        for i, message in enumerate(messages):
            # 如果是第一条消息且不是人类发送的，则标记first_is_not_question为1，并跳过本次循环
            if i == 0 and message['from'] != 'human':
                first_is_not_question = 1
                continue
            # 根据first_is_not_question的值决定当前消息是问题还是答案
            if i % 2 == first_is_not_question:
                question_list.append(message['value'])  # 添加到问题列表
            else:
                answer_list.append(message['value'])    # 添加到答案列表

        # 断言问题和答案的数量相等
        assert len(question_list) == len(answer_list) , \
            f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list               # 返回问题和答案列表
    

    def prompt(
        self,
        question_list,                                  # 问题列表，可以是单个字符串或字符串列表
        answer_list                                     # 答案列表，可以是单个字符串或字符串列表
    ):
        """
        根据问题列表和答案列表生成提示信息。

        参数:
        question_list -- 问题列表，可以是单个字符串或字符串列表
        answer_list -- 答案列表，可以是单个字符串或字符串列表

        返回:
        msg -- 生成的提示信息
        """
        # 如果question_list是单个字符串，则将其转换为列表
        if type(question_list) is str:
            question_list = [question_list]
        # 如果answer_list是单个字符串，则将其转换为列表
        if type(answer_list) is str:
            answer_list = [answer_list]
        # 调用内部方法_prompt生成提示信息
        msg = self._prompt(question_list, answer_list)
        # 返回生成的提示信息
        return msg

    def _prompt(
        self,
        question_list,  # 问题列表
        answer_list,    # 答案列表
    ):
        """
        _prompt方法用于将问题和答案组合成提示信息。

        :param question_list: 包含问题的列表
        :param answer_list: 包含答案的列表
        :return: 组合后的提示信息字符串
        """
        msg = ""        # 初始化提示信息字符串
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()                                              # 如果是第一个问题，添加系统信息
            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()            # 移除默认图片标记
                question = self.format_image_token.apply(content=question).strip()      # 格式化图片标记
            msg += self.format_user.apply(content=question)                             # 格式化用户问题
            msg += self.format_assistant.apply(content=answer)                          # 格式化助手答案
        return msg                                                                      # 返回组合后的提示信息
    
    def make_labels(self, input_ids, prompt, tokenizer):
        """
        根据输入的ids、提示和分词器生成标签。

        :param input_ids: 输入的token ids列表
        :param prompt: 提示字符串
        :param tokenizer: 分词器对象
        :return: 生成的标签列表
        """
        labels = copy.deepcopy(input_ids)                                                       # 深拷贝输入的ids，避免原数据被修改
        sep, eos_token = self.separator.apply()                                                 # 获取分隔符和结束标记
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())                                # 计算非填充token的总长度

        # 如果填充token和结束标记相同，需要额外计算提示中结束标记的数量
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)                                                        # 根据结束标记分割提示字符串
        eos_token_length = len(tokenizer.encode(eos_token))                                     # 获取结束标记的编码长度
        labels, cur_len = self._make_masks(labels, tokenizer, sep, eos_token_length, rounds)    # 生成标签掩码

        # 如果当前长度小于模型最大长度，检查tokenization是否匹配
        if cur_len < tokenizer.model_max_length:
            import time
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("prompt: ", prompt)
                print(labels)
                print(input_ids)
                time.sleep(5)                                                                    # 暂停5秒，以便观察警告信息
                labels[:] = IGNORE_INDEX                                                         # 将标签全部设置为忽略索引
        return labels
        
        
        
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        """
        根据给定的rounds生成labels的掩码。

        Args:
            labels (List[int]): 需要生成掩码的标签列表。
            tokenizer: 用于分词的工具。
            sep (str): 分隔符，用于分割rounds中的指令和图像部分。
            eos_token_length (int): 结束标记的长度。
            rounds (List[str]): 包含指令和图像部分的字符串列表。

        Returns:
            Tuple[List[int], int]: 返回两个值，第一个是更新后的labels列表，第二个是当前处理的总长度。
        """
        cur_len = 0                                                                                 # 当前处理的总长度
        for rou in rounds:                                                                          # 遍历rounds中的每个元素
            if rou == "":                                                                           # 如果元素为空字符串，则跳出循环
                break
            parts = rou.split(sep)                                                                  # 使用分隔符分割指令和图像部分
            if len(parts) != 2:                                                                     # 如果分割后的部分不是两部分，则跳出循环
                break
            parts[0] += sep                                                                         # 在指令部分后面添加分隔符
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length          # 计算当前round的总长度
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1              # 计算指令部分的长度
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX                              # 更新labels的掩码
            cur_len += round_len                                                                    # 更新当前处理的总长度
        labels[cur_len:] = IGNORE_INDEX                                                             # 将剩余的labels设置为IGNORE_INDEX
        return labels, cur_len                                                                      # 返回更新后的labels和当前处理的总长度
        
    @classmethod    
    def tokenizer_image_token(cls, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        """
        将包含图像标记的提示字符串转换为模型输入的ID列表或张量。

        :param prompt: 包含'<image>'标记的字符串，用于指示图像的位置。
        :param tokenizer: 用于将文本转换为模型输入ID的分词器对象。
        :param image_token_index: 图像标记在词汇表中的索引，默认为预定义的IMAGE_TOKEN_INDEX。
        :param return_tensors: 指定返回的张量类型，'pt'表示返回PyTorch张量，None表示返回ID列表。
        :return: 转换后的模型输入ID列表或张量。
        """

        # 定义一个内部函数，用于在列表X的每个元素之间插入分隔符sep
        def _insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        # 将prompt字符串按'<image>'分割成多个块，并对每个块进行分词
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        input_ids = []
        offset = 0
        # 如果第一个块的第一个ID是bos_token_id，则将其添加到input_ids，并设置offset为1
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        # 在prompt_chunks的每个块之间插入图像标记索引，并将结果合并到input_ids列表中
        for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        # 根据return_tensors参数返回相应的结果
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids





