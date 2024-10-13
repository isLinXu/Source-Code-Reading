from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from packaging import version

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template
from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
import tokenizers

# 系统描述，用于模板中的系统消息部分
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

# 定义一个名为GemmaTemplate的模板类，继承自Template类
@register_template('gemma')
@dataclass
class GemmaTemplate(Template):
    # 定义模板中的各种格式化器，用于格式化输出内容
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<eos>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<eos>'])

    # 定义一个私有方法，用于生成标签掩码，用于标记哪些部分需要被忽略
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        """
        Generate masks for the labels based on the tokenizer and rounds of dialogue.

        Args:
            labels (list): List of labels to be masked.
            tokenizer (PreTrainedTokenizer): Tokenizer used to tokenize the input.
            sep (str): Separator used to split the dialogue rounds.
            eos_token_length (int): Length of the end-of-sentence token.
            rounds (list): List of dialogue rounds.

        Returns:
            tuple: A tuple containing the masked labels and the current length.
        """
        cur_len = 1 # bos                   # 初始化当前长度为1，代表bos（beginning of sentence）标记
        eos_token_length = 1                # 初始化eos（end of sentence）标记的长度为1
        bos_token_length = 1                # 初始化bos标记的长度为1
        labels[:cur_len] = IGNORE_INDEX     # 将标签列表的前cur_len个元素设置为IGNORE_INDEX，表示这些部分不需要被处理

        # 遍历对话轮次
        for i, rou in enumerate(rounds):
            if rou == "":
                break                       # 如果轮次为空字符串，则跳出循环
            parts = rou.split(sep)          # 使用分隔符分割轮次
            if len(parts) != 2:
                break                       # 如果分割后的部分不是两部分，则跳出循环
            parts[0] += sep                 # 在第一部分的末尾添加分隔符
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length - bos_token_length # 计算轮次的长度
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1 - bos_token_length     # 计算指令的长度
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # 将标签列表中对应的部分设置为IGNORE_INDEX
            cur_len += round_len            # 更新当前长度
        
        labels[cur_len:] = IGNORE_INDEX     # 将标签列表剩余部分设置为IGNORE_INDEX
        return labels, cur_len              # 返回掩码后的标签列表和当前长度
