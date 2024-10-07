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

# 检查tokenizers库的版本是否大于等于0.14
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

# 系统消息模板
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

# 注册llama模板
@register_template('llama')
@dataclass
class LlamaTemplate(Template):
    # 定义不同部分的格式化器
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "</s>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '</s>'])
    
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        """
        创建掩码以忽略某些标签。

        :param labels: 标签列表
        :param tokenizer: 用于分词的tokenizer对象
        :param sep: 分隔符字符串
        :param eos_token_length: 结束标记的长度
        :param rounds: 轮次信息的列表
        :return: 更新后的标签列表和当前长度
        """
        cur_len = 1 # bos                       # bos标记的长度
        eos_token_length = 1                    # eos标记的长度
        bos_token_length = 1                    # 开始标记的长度
        labels[:cur_len] = IGNORE_INDEX         # 忽略开始标记

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)              # 分割轮次信息
            if len(parts) != 2:
                break
            parts[0] += sep                     # 确保分隔符存在
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length - bos_token_length   # 计算轮次长度
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1 - bos_token_length       # 计算指令长度
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1                                                                                  # 如果不是第一个轮次，则减去1
                instruction_len -= 1                                                                            # 同上
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX                                          # 忽略指令部分
            cur_len += round_len                                                                                # 更新当前长度
        
        labels[cur_len:] = IGNORE_INDEX                                                                         # 忽略剩余部分
        return labels, cur_len







