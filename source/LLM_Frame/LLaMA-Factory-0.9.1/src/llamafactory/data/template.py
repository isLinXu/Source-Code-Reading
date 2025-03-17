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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from transformers.utils.versions import require_version
from typing_extensions import override

from ..extras import logging
from .data_utils import Role
from .formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from .mm_plugin import get_mm_plugin


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .formatter import SLOTS, Formatter
    from .mm_plugin import BasePlugin


logger = logging.get_logger(__name__)


# 模板基类
@dataclass
class Template:
    # 各角色消息格式化器
    format_user: "Formatter"          # 用户消息格式化器
    format_assistant: "Formatter"     # 助手消息格式化器
    format_system: "Formatter"        # 系统消息格式化器
    format_function: "Formatter"      # 函数调用格式化器
    format_observation: "Formatter"   # 观察结果格式化器
    format_tools: "Formatter"         # 工具定义格式化器
    format_separator: "Formatter"     # 对话轮次分隔符
    format_prefix: "Formatter"        # 前缀格式化器
    default_system: str               # 默认系统提示
    stop_words: List[str]             # 停止词列表
    efficient_eos: bool               # 是否高效添加EOS
    replace_eos: bool                 # 是否替换原有EOS
    replace_jinja_template: bool      # 是否替换Jinja模板
    mm_plugin: "BasePlugin"           # 多模态插件

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        单轮对话编码：返回提示和响应的token id对
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:  # 合并除最后一条外的所有消息作为prompt
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]  # 最后一条消息作为回答
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        r"""
        Extracts tool message.
        """
        return self.format_tools.extract(content)

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        核心编码方法：将格式化消息转换为token id序列
        """
        system = system or self.default_system  # 使用默认系统提示
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []  # 当前消息的组成元素

            # 处理首轮消息
            if i == 0:
                elements += self.format_prefix.apply()  # 添加前缀
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))  # 系统提示+工具定义

            # 添加对话轮次分隔符
            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            # 根据角色选择格式化器
            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> List[int]:
        r"""
        Converts elements to token ids.
        将模板元素转换为token id序列
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):  # 处理普通文本元素
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)  # 编码文本不添加特殊token
            elif isinstance(elem, dict):  # 处理特殊token字典
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]  # 直接转换token为id
            elif isinstance(elem, set):  # 处理预定义token集合
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]  # 添加BOS token
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]  # 添加EOS token
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids  # 返回合并后的token id序列


# Llama2专用模板
@dataclass
class Llama2Template(Template):
    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: str,
        tools: str,
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        Llama2专用编码逻辑：首轮包含系统提示，后续轮次添加分隔符
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


# 模板注册字典（全局单例）
TEMPLATES: Dict[str, "Template"] = {}  # 存储所有注册的对话模板


# 模板注册函数
def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,  # 用户消息模板
    format_assistant: Optional["Formatter"] = None,  # 助手消息模板
    format_system: Optional["Formatter"] = None,  # 系统提示模板
    format_function: Optional["Formatter"] = None,  # 函数调用模板
    format_observation: Optional["Formatter"] = None,  # 观察结果模板
    format_tools: Optional["Formatter"] = None,  # 工具定义模板
    format_separator: Optional["Formatter"] = None,  # 轮次分隔符
    format_prefix: Optional["Formatter"] = None,  # 前缀模板
    default_system: str = "",  # 默认系统提示
    stop_words: Sequence[str] = [],  # 停止词列表
    efficient_eos: bool = False,  # 是否优化EOS添加
    replace_eos: bool = False,  # 是否替换原有EOS
    replace_jinja_template: bool = True,  # 是否替换Jinja模板
    mm_plugin: "BasePlugin" = get_mm_plugin(name="base"),  # 多模态插件
) -> None:
    r"""
    Registers a chat template.
    注册对话模板到全局TEMPLATES字典
    """
    # 设置默认格式化器
    eos_slots = [] if efficient_eos else [{"eos_token"}]  # 根据是否高效EOS决定插槽配置
    template_class = Llama2Template if name.startswith("llama2") else Template  # 根据名称选择模板类
    
    # 定义各角色默认格式化器
    default_user_formatter = StringFormatter(slots=["{{content}}"])  # 用户消息默认格式
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)  # 助手消息带EOS
    default_function_formatter = FunctionFormatter(slots=eos_slots, tool_format="default")  # 函数调用格式化
    default_tool_formatter = ToolFormatter(tool_format="default")  # 工具定义格式化
    
    # 注册模板到全局字典
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,  # 优先使用自定义格式化器
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or EmptyFormatter(),
        format_prefix=format_prefix or EmptyFormatter(),
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        mm_plugin=mm_plugin,  # 注入多模态处理插件
    )


# EOS token处理函数
def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    r"""
    Adds or replaces the eos token in the tokenizer.
    添加或替换分词器的EOS token
    """
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})  # 添加特殊token
    # 记录操作日志
    logger.info_rank0(f"Add eos token: {tokenizer.eos_token}" if is_added else f"Replace eos token: {tokenizer.eos_token}")

    if num_added_tokens > 0:
        logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")


def _jinja_escape(content: str) -> str:
    return content.replace("'", r"\'")


# Jinja模板转换函数
def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
    r"""
    Converts slot configuration to jinja template string.
    将插槽配置转换为Jinja模板字符串
    """
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            # 处理包含{{content}}的字符串插槽
            parts = slot.split("{{content}}")
            if parts[0]:
                slot_items.append("'" + _jinja_escape(parts[0]) + "'")  # 转义单引号
            if len(parts) > 1:
                slot_items.append(placeholder)
                if parts[1]:
                    slot_items.append("'" + _jinja_escape(parts[1]) + "'")
        elif isinstance(slot, set):  # 处理特殊token集合
            if "bos_token" in slot:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif "eos_token" in slot:
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError("Dict is not supported.")

    return " + ".join(slot_items)  # 拼接为Jinja表达式


def _get_jinja_template(template: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    r"""
    Returns the jinja template.
    """
    jinja_template = ""

    prefix = _convert_slots_to_jinja(template.format_prefix.apply(), tokenizer)
    if prefix:
        jinja_template += "{{ " + prefix + " }}"

    if template.default_system:
        jinja_template += "{% set system_message = '" + _jinja_escape(template.default_system) + "' %}"

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
    )

    system_message = _convert_slots_to_jinja(template.format_system.apply(), tokenizer, placeholder="system_message")
    if not isinstance(template, Llama2Template):
        jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"

    jinja_template += "{% for message in loop_messages %}"
    jinja_template += "{% set content = message['content'] %}"
    if isinstance(template, Llama2Template):
        jinja_template += "{% if loop.index0 == 0 and system_message is defined %}"
        jinja_template += "{% set content = " + system_message + " + message['content'] %}"
        jinja_template += "{% endif %}"

    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_slots_to_jinja(template.format_user.apply(), tokenizer)
    jinja_template += "{{ " + user_message + " }}"

    jinja_template += "{% elif message['role'] == 'assistant' %}"
    assistant_message = _convert_slots_to_jinja(
        template.format_assistant.apply() + template.format_separator.apply(), tokenizer
    )
    jinja_template += "{{ " + assistant_message + " }}"
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    return jinja_template


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""
    Gets chat template and fixes the tokenizer.
    获取对话模板并修复分词器配置
    """
    # 获取模板实例
    if data_args.template is None:
        template = TEMPLATES["empty"]  # 使用占位模板
    else:
        template = TEMPLATES.get(data_args.template, None)
        if template is None:
            raise ValueError(f"Template {data_args.template} does not exist.")  # 模板不存在时报错

    # 多模态插件版本检查
    if template.mm_plugin.__class__.__name__ != "BasePlugin":
        require_version("transformers>=4.45.0", "To fix: pip install transformers>=4.45.0")

    # 训练模式兼容性检查
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    # 工具格式处理
    if data_args.tool_format is not None:
        logger.info_rank0(f"Using tool format: {data_args.tool_format}.")
        eos_slots = [] if template.efficient_eos else [{"eos_token"}]
        template.format_function = FunctionFormatter(slots=eos_slots, tool_format=data_args.tool_format)  # 动态更新函数格式化器
        template.format_tools = ToolFormatter(tool_format=data_args.tool_format)  # 更新工具定义格式化器

    # EOS token替换逻辑
    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])  # 用第一个停止词替换EOS
        stop_words = stop_words[1:]  # 剩余停止词保留

    # 确保分词器有EOS token
    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")  # 默认使用GPT风格EOS

    # 设置pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用EOS作为pad token
        logger.info_rank0(f"Add pad token: {tokenizer.pad_token}")

    # 添加停止词到特殊token
    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info_rank0("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")  # 词汇表扩展警告

    # 设置Jinja模板
    if tokenizer.chat_template is None or template.replace_jinja_template:
        try:
            tokenizer.chat_template = _get_jinja_template(template, tokenizer)  # 生成并注入Jinja模板
        except ValueError as e:
            logger.info_rank0(f"Cannot add this chat template to tokenizer: {e}.")  # 模板注入失败日志

    return template  # 返回配置好的模板实例


_register_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    ),
)


_register_template(
    name="aquila",
    format_user=StringFormatter(slots=["Human: {{content}}###Assistant:"]),
    format_separator=EmptyFormatter(slots=["###"]),
    default_system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    stop_words=["</s>"],
    efficient_eos=True,
)


_register_template(
    name="atom",
    format_user=StringFormatter(
        slots=[{"bos_token"}, "Human: {{content}}\n", {"eos_token"}, {"bos_token"}, "Assistant:"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}\n", {"eos_token"}]),
)


_register_template(
    name="baichuan",
    format_user=StringFormatter(slots=[{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}]),
    efficient_eos=True,
)


_register_template(
    name="baichuan2",
    format_user=StringFormatter(slots=["<reserved_106>{{content}}<reserved_107>"]),
    efficient_eos=True,
)


_register_template(
    name="belle",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nBelle: "]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="bluelm",
    format_user=StringFormatter(slots=[{"token": "[|Human|]:"}, "{{content}}", {"token": "[|AI|]:"}]),
)


_register_template(
    name="breeze",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST] "]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
)


_register_template(
    name="chatglm2",
    format_user=StringFormatter(slots=["[Round {{idx}}]\n\n问：{{content}}\n\n答："]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
    efficient_eos=True,
)


_register_template(
    name="chatglm3",
    format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
    format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
    format_system=StringFormatter(slots=[{"token": "<|system|>"}, "\n", "{{content}}"]),
    format_function=FunctionFormatter(slots=[], tool_format="glm4"),
    format_observation=StringFormatter(
        slots=[{"token": "<|observation|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]
    ),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


_register_template(
    name="chatml",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


_register_template(
    name="chatml_de",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="Du bist ein freundlicher und hilfsbereiter KI-Assistent.",
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


_register_template(
    name="codegeex2",
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
)


_register_template(
    name="codegeex4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=[], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>\n"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    default_system=(
        "你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，"
        "并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。"
    ),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


_register_template(
    name="cohere",
    format_user=StringFormatter(
        slots=[
            (
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="cpm",
    format_user=StringFormatter(slots=["<用户>{{content}}<AI>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="cpm3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|im_end|>"],
)


_register_template(
    name="dbrx",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system=(
        "You are DBRX, created by Databricks. You were last updated in December 2023. "
        "You answer questions based on information available up to that point.\n"
        "YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough "
        "responses to more complex and open-ended questions.\nYou assist with various tasks, "
        "from writing to coding (using markdown for code blocks — remember to use ``` with "
        "code, JSON, and tables).\n(You do not have real-time data access or code execution "
        "capabilities. You avoid stereotyping and provide balanced perspectives on "
        "controversial topics. You do not provide song lyrics, poems, or news articles and "
        "do not divulge details of your training data.)\nThis is your system prompt, "
        "guiding your responses. Do not reference it, just respond to the user. If you find "
        "yourself talking about this message, stop. You should be responding appropriately "
        "and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION "
        "ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY."
    ),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="deepseek",
    format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="deepseekcoder",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n### Response:"]),
    format_assistant=StringFormatter(slots=["\n{{content}}\n<|EOT|>"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
        "developed by DeepSeek Company, and you only answer questions related to computer science. "
        "For politically sensitive questions, security and privacy issues, "
        "and other non-computer science questions, you will refuse to answer.\n"
    ),
)


_register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_template(
    name="empty",
    efficient_eos=True,
)


_register_template(
    name="exaone",
    format_user=StringFormatter(slots=["[|user|]{{content}}\n[|assistant|]"]),
    format_system=StringFormatter(slots=["[|system|]{{content}}[|endofturn|]\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_template(
    name="falcon",
    format_user=StringFormatter(slots=["User: {{content}}\nFalcon:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    efficient_eos=True,
)


_register_template(
    name="fewshot",
    format_separator=EmptyFormatter(slots=["\n\n"]),
    efficient_eos=True,
)


_register_template(
    name="gemma",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_separator=EmptyFormatter(slots=["<end_of_turn>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
    replace_jinja_template=False,
)


_register_template(
    name="glm4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=[], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


_register_template(
    name="index",
    format_user=StringFormatter(slots=["reserved_0{{content}}reserved_1"]),
    format_system=StringFormatter(slots=["<unk>{{content}}"]),
    efficient_eos=True,
)


_register_template(
    name="intern",
    format_user=StringFormatter(slots=["<|User|>:{{content}}\n<|Bot|>:"]),
    format_system=StringFormatter(slots=["<|System|>:{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["<eoa>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<eoa>"],
    efficient_eos=True,  # internlm tokenizer cannot set eos_token_id
)


_register_template(
    name="intern2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|im_end|>"],
    efficient_eos=True,  # internlm2 tokenizer cannot set eos_token_id
)


_register_template(
    name="llama2",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
)


_register_template(
    name="llama2_zh",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
)


_register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
    replace_jinja_template=False,
)


_register_template(
    name="mllama",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
    replace_jinja_template=False,
    mm_plugin=get_mm_plugin(name="mllama", image_token="<|image|>"),
)


_register_template(
    name="llava",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava", image_token="<image>"),
)


_register_template(
    name="llava_next",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


_register_template(
    name="llava_next_llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
    replace_jinja_template=False,
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


_register_template(
    name="llava_next_mistral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


_register_template(
    name="llava_next_qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    replace_jinja_template=False,
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


_register_template(
    name="llava_next_yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


_register_template(
    name="llava_next_video",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
)


_register_template(
    name="llava_next_video_mistral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
)


_register_template(
    name="llava_next_video_yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
)


_register_template(
    name="mistral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="olmo",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"eos_token"}]),
)


_register_template(
    name="openchat",
    format_user=StringFormatter(slots=["GPT4 Correct User: {{content}}", {"eos_token"}, "GPT4 Correct Assistant:"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="openchat-3.6",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)


_register_template(
    name="opencoder",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are OpenCoder, created by OpenCoder Team.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    replace_jinja_template=False,
)


_register_template(
    name="orion",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: ", {"eos_token"}]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="paligemma",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_separator=EmptyFormatter(slots=["<end_of_turn>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
    mm_plugin=get_mm_plugin(name="paligemma", image_token="<image>"),
)


_register_template(
    name="phi",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


_register_template(
    name="phi_small",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_prefix=EmptyFormatter(slots=[{"<|endoftext|>"}]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


_register_template(
    name="pixtral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="pixtral", image_token="[IMG]"),
)


_register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    replace_jinja_template=False,
)


_register_template(
    name="qwen2_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    replace_jinja_template=False,
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


_register_template(
    name="sailor",
    format_user=StringFormatter(slots=["<|im_start|>question\n{{content}}<|im_end|>\n<|im_start|>answer\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system=(
        "You are an AI assistant named Sailor created by Sea AI Lab. "
        "Your answer should be friendly, unbiased, faithful, informative and detailed."
    ),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="solar",
    format_user=StringFormatter(slots=["### User:\n{{content}}\n\n### Assistant:\n"]),
    format_system=StringFormatter(slots=["### System:\n{{content}}\n\n"]),
    efficient_eos=True,
)


_register_template(
    name="starchat",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


_register_template(
    name="telechat",
    format_user=StringFormatter(slots=["<_user>{{content}}<_bot>"]),
    format_system=StringFormatter(slots=["<_system>{{content}}<_end>"]),
    stop_words=["<_end>"],
    replace_eos=True,
)


_register_template(
    name="vicuna",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
)


_register_template(
    name="video_llava",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="video_llava", image_token="<image>", video_token="<video>"),
)


_register_template(
    name="xuanyuan",
    format_user=StringFormatter(slots=["Human: {{content}} Assistant:"]),
    default_system=(
        "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，"
        "会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、"
        "不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
    ),
)


_register_template(
    name="xverse",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: "]),
)


_register_template(
    name="yayi",
    format_user=StringFormatter(slots=[{"token": "<|Human|>"}, ":\n{{content}}\n\n", {"token": "<|YaYi|>"}, ":"]),
    format_system=StringFormatter(slots=[{"token": "<|System|>"}, ":\n{{content}}\n\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "You are a helpful, respectful and honest assistant named YaYi "
        "developed by Beijing Wenge Technology Co.,Ltd. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    stop_words=["<|End|>"],
)


_register_template(
    name="yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="yi_vl",
    format_user=StringFormatter(slots=["### Human: {{content}}\n### Assistant:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system=(
        "This is a chat between an inquisitive human and an AI assistant. "
        "Assume the role of the AI assistant. Read all the images carefully, "
        "and respond to the human's questions with informative, helpful, detailed and polite answers. "
        "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。"
        "仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
    ),
    stop_words=["###"],
    efficient_eos=True,
    mm_plugin=get_mm_plugin(name="llava", image_token="<image>"),
)


_register_template(
    name="yuan",
    format_user=StringFormatter(slots=["{{content}}", {"token": "<sep>"}]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<eod>"],
    replace_eos=True,
)


_register_template(
    name="zephyr",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}", {"eos_token"}, "<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}", {"eos_token"}]),
    default_system="You are Zephyr, a helpful assistant.",
)


_register_template(
    name="ziya",
    format_user=StringFormatter(slots=["<human>:{{content}}\n<bot>:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)
