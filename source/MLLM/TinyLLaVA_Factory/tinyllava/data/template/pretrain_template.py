from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from ...utils.constants import *
from . import register_template

from transformers import PreTrainedTokenizer
import torch
    

# 注册名为 'pretrain' 的模板
@register_template('pretrain')
# 使用 dataclass 装饰器定义 PretrainTemplate 类，继承自 Template 类
@dataclass
class PretrainTemplate(Template):
    # 定义格式化图片 token 的 Formatter，默认使用 EmptyFormatter
    format_image_token: "Formatter" = EmptyFormatter(slot="")
    # 定义格式化用户输入的 Formatter，默认使用 EmptyFormatter，并指定 slot 为 <image>
    format_user: "Formatter" = EmptyFormatter(slot="<image>")
    # 定义格式化助手回复的 Formatter，默认使用 StringFormatter，并指定 slot 为 {{content}}\n
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}\n")
    # 定义系统的 Formatter，默认使用 EmptyFormatter
    system: "Formatter" = EmptyFormatter(slot="")
    # 定义分隔符的 Formatter，默认使用 EmptyFormatter，并指定 slot 为空字符串列表
    separator: "Formatter" = EmptyFormatter(slot=['', ''])

    # 定义 make_labels 方法，用于生成标签
    def make_labels(self, input_ids, prompt, tokenizer):
        """
        根据输入的 token ids 和 prompt 生成标签。

        :param input_ids: 输入的 token ids 列表
        :param prompt: 提示字符串
        :param tokenizer: 分词器对象
        :return: 生成的标签列表
        """
        # 深拷贝 input_ids 到 labels
        labels = copy.deepcopy(input_ids)
        # 获取 <image> 对应的 token 数量
        mask_len = len(self.tokenizer_image_token("<image>", tokenizer))
        # 将 labels 中对应 <image> 的 token ids 设置为 IGNORE_INDEX
        labels[:mask_len] = IGNORE_INDEX
        # 返回生成的标签列表
        return labels







