from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from transformers import PreTrainedTokenizer
import torch

# 系统描述，用于在对话中显示
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

# 使用装饰器注册一个名为'phi'的模板
@register_template('phi')
# 定义一个继承自Template的数据类PhiTemplate
@dataclass
class PhiTemplate(Template):
    # 定义一个格式化图片标记的Formatter，使用StringFormatter并指定slot
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    # 定义一个格式化用户消息的Formatter，使用StringFormatter并指定slot
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    # 定义一个格式化助手消息的Formatter，使用StringFormatter并指定slot
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>")
    # 定义一个系统消息的Formatter，使用EmptyFormatter并指定slot为系统描述
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    # 定义一个分隔符的Formatter，使用EmptyFormatter并指定slot为特定的字符串列表
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<|endoftext|>'])







