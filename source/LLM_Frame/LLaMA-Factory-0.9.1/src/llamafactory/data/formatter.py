# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from typing_extensions import override

from .data_utils import SLOTS  # 从当前目录导入SLOTS类型
from .tool_utils import get_tool_utils  # 工具处理工具函数


if TYPE_CHECKING:
    from .tool_utils import FunctionCall  # 类型检查时导入FunctionCall类型


@dataclass
class Formatter(ABC):
    r"""
    格式化器的抽象基类
    """
    slots: SLOTS = field(default_factory=list)  # 槽位定义，默认为空列表
    tool_format: Optional[str] = None  # 工具格式类型

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        r"""
        Forms a list of slots according to the inputs to encode.
        根据输入参数生成槽位列表
        """
        ...

    def extract(self, content: str) -> Union[str, List["FunctionCall"]]:
        r"""
        Extract a list of tuples from the response message if using tools.
        从响应消息中提取函数调用列表（当使用工具时）

        Each tuple consists of function name and function arguments.
        每个元组包含函数名称和参数
        """
        raise NotImplementedError  # 子类必须实现


@dataclass
class EmptyFormatter(Formatter):
    r"""
    空格式化器，直接返回预定义槽位
    """
    def __post_init__(self):
        # 检查是否包含占位符
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):  # 过滤字符串类型的槽位
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):  # 正则匹配{{variable}}格式
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")  # 空格式化器不应包含占位符

    @override
    def apply(self, **kwargs) -> SLOTS:
        return self.slots  # 直接返回预定义槽位


@dataclass
class StringFormatter(Formatter):
    r"""
    字符串格式化器，支持占位符替换
    """
    def __post_init__(self):
        # 检查是否包含至少一个占位符
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")  # 必须包含至少一个占位符

    @override
    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                # 替换所有占位符
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(f"Expected a string, got {value}")  # 值必须是字符串

                    slot = slot.replace("{{" + name + "}}", value, 1)  # 逐个替换占位符
                elements.append(slot)
            elif isinstance(slot, (dict, set)):  # 处理特殊标记（如停止标记）
                elements.append(slot)
            else:
                raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}")

        return elements


@dataclass
class FunctionFormatter(Formatter):
    r"""
    函数调用格式化器，处理工具调用格式
    """
    def __post_init__(self):
        # 初始化时添加工具相关的槽位
        self.slots = get_tool_utils(self.tool_format).get_function_slots() + self.slots

    @override
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")  # 获取函数调用内容
        functions: List[Tuple[str, str]] = []  # 存储解析后的函数调用
        try:
            tool_calls = json.loads(content)  # 解析JSON内容
            if not isinstance(tool_calls, list):  # 处理单个函数调用情况
                tool_calls = [tool_calls]

            # 提取函数名和参数
            for tool_call in tool_calls:
                functions.append((
                    tool_call["name"],  # 函数名
                    json.dumps(tool_call["arguments"], ensure_ascii=False)  # 参数转为JSON字符串
                ))

        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in function message: {str([content])}")  # JSON解析失败

        elements = []
        for name, arguments in functions:
            for slot in self.slots:
                if isinstance(slot, str):
                    # 替换函数名和参数占位符
                    slot = slot.replace("{{name}}", name).replace("{{arguments}}", arguments)
                    elements.append(slot)
                elif isinstance(slot, (dict, set)):  # 处理特殊标记
                    elements.append(slot)
                else:
                    raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}")

        return elements


@dataclass
class ToolFormatter(Formatter):
    r"""
    工具描述格式化器，处理工具定义格式
    """
    def __post_init__(self):
        # 初始化工具处理工具
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")  # 获取工具描述内容
        try:
            tools = json.loads(content)  # 解析JSON
            return [self.tool_utils.tool_formatter(tools) if len(tools) != 0 else ""]  # 格式化工具描述
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in tool description: {str([content])}")  # JSON解析失败

    @override
    def extract(self, content: str) -> Union[str, List["FunctionCall"]]:
        # 从内容中提取函数调用
        return self.tool_utils.tool_extractor(content)
