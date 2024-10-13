from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from transformers import PreTrainedTokenizer
import torch

# 定义一个系统消息字符串，描述了用户和AI助手之间的对话场景
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

# 使用@register_template装饰器注册一个名为'qwen2_base'的模板类
@register_template('qwen2_instruct')
# 使用dataclass装饰器定义一个数据类Qwen2BaseTemplate，继承自Template类
@dataclass
class Qwen2InstructTemplate(Template):
    # 定义format_image_token属性，使用StringFormatter格式化图片内容
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    # 定义format_user属性，用于格式化用户发言
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    # 定义format_assistant属性，用于格式化AI助手的回复
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<|im_end|>")
    # 定义system属性，使用EmptyFormatter格式化系统消息
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    # 定义separator属性，用于分隔用户发言和AI助手回复
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<|im_end|>'])







