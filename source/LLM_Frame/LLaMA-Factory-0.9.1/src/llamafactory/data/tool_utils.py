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
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from typing_extensions import override

from .data_utils import SLOTS


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)


GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具{tool_text}"
)


# 工具调用相关命名元组
FunctionCall = namedtuple("FunctionCall", ["name", "arguments"])  # 函数调用结构：名称+参数


# 工具工具类基类（抽象类）
@dataclass
class ToolUtils(ABC):
    """
    Base class for tool utilities.
    工具处理工具基类，定义工具处理的通用接口
    """
    @staticmethod
    @abstractmethod
    def get_function_slots() -> SLOTS:
        r"""
        Gets a list of slots corresponding to a single function call.
        获取单次函数调用对应的模板插槽
        """
        ...

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        r"""
        Generates the system message describing all the available tools.
        生成描述可用工具的系统消息
        """
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        r"""
        Extracts all the function calls from the response message.
        从响应内容中提取函数调用
        """
        ...


# 默认工具处理实现
class DefaultToolUtils(ToolUtils):
    @override
    @staticmethod
    def get_function_slots() -> SLOTS:
        return ["Action: {{name}}\nAction Input: {{arguments}}\n"]  # 默认工具调用格式模板

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:  # 遍历每个工具定义
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():  # 处理每个参数
                # 构建参数描述
                required = ", required" if name in tool["parameters"].get("required", []) else ""
                enum = ", should be one of [{}]".format(", ".join(param["enum"])) if param.get("enum") else ""
                items = ", where each item should be {}".format(param["items"].get("type", "")) if param.get("items") else ""
                
                # 拼接参数描述
                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            # 构建工具描述
            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        # 使用正则表达式匹配工具调用模式
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: List[Tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content  # 没有匹配到工具调用则返回原始内容

        results = []
        for match in action_match:  # 处理每个匹配结果
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")  # 清理输入格式
            try:
                arguments = json.loads(tool_input)  # 解析JSON参数
                results.append((tool_name, json.dumps(arguments, ensure_ascii=False)))  # 重新序列化保证格式
            except json.JSONDecodeError:
                return content  # 解析失败返回原始内容

        return results


# GLM4专用工具处理实现
class GLM4ToolUtils(ToolUtils):
    @override
    @staticmethod
    def get_function_slots() -> SLOTS:
        return ["{{name}}\n{{arguments}}"]  # GLM4专用工具调用格式

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            # 生成GLM4格式的工具描述（直接展示JSON结构）
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        if "\n" not in content:  # GLM4格式要求换行分隔工具名和参数
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)  # 分割工具名和输入
        try:
            arguments = json.loads(tool_input)  # 解析JSON参数
            return [(tool_name, json.dumps(arguments, ensure_ascii=False))]
        except json.JSONDecodeError:
            return content


# 工具类注册表
TOOLS = {
    "default": DefaultToolUtils(),  # 默认工具处理
    "glm4": GLM4ToolUtils(),         # GLM4专用处理
}

def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")  # 工具类不存在时报错

    return tool_utils
