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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Literal, Optional, Sequence, Union


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from vllm import AsyncLLMEngine

    from ..data import Template
    from ..data.mm_plugin import ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


# 响应数据结构类
@dataclass
class Response:
    response_text: str  # 生成的响应文本
    response_length: int  # 响应token长度
    prompt_length: int  # 提示token长度
    finish_reason: Literal["stop", "length"]  # 停止原因：自然结束/长度限制


# 推理引擎基类（抽象类）
class BaseEngine(ABC):
    r"""
    Base class for inference engine of chat models.
    聊天模型推理引擎基类，必须实现异步方法：chat(), stream_chat() 和 get_scores()
    """
    model: Union["PreTrainedModel", "AsyncLLMEngine"]  # 模型实例（支持HuggingFace和vLLM）
    tokenizer: "PreTrainedTokenizer"  # 分词器实例
    can_generate: bool  # 是否支持生成模式
    template: "Template"  # 对话模板处理器
    generating_args: Dict[str, Any]  # 生成参数配置

    @abstractmethod
    def __init__(
        self,
        model_args: "ModelArguments",  # 模型参数（路径、精度等）
        data_args: "DataArguments",  # 数据参数（模板、工具格式等）
        finetuning_args: "FinetuningArguments",  # 微调参数（适配器路径等）
        generating_args: "GeneratingArguments",  # 生成参数（温度、top_p等）
    ) -> None:
        r"""
        Initializes an inference engine.
        初始化推理引擎，加载模型和配置
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Dict[str, str]],  # 对话历史消息列表
        system: Optional[str] = None,  # 系统提示
        tools: Optional[str] = None,  # 工具定义
        images: Optional[Sequence["ImageInput"]] = None,  # 图像输入
        videos: Optional[Sequence["VideoInput"]] = None,  # 视频输入
        **input_kwargs,  # 其他输入参数
    ) -> List["Response"]:
        r"""
        Gets a list of responses of the chat model.
        获取聊天模型的批量响应（同步生成）
        """
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],  # 对话历史消息列表
        system: Optional[str] = None,  # 系统提示
        tools: Optional[str] = None,  # 工具定义
        images: Optional[Sequence["ImageInput"]] = None,  # 图像输入
        videos: Optional[Sequence["VideoInput"]] = None,  # 视频输入
        **input_kwargs,  # 其他输入参数
    ) -> AsyncGenerator[str, None]:
        r"""
        Gets the response token-by-token of the chat model.
        流式获取聊天模型的逐token响应（异步生成）
        """
        ...

    @abstractmethod
    async def get_scores(
        self,
        batch_input: List[str],  # 输入文本列表（用于奖励模型评分）
        **input_kwargs,  # 其他输入参数
    ) -> List[float]:
        r"""
        Gets a list of scores of the reward model.
        获取奖励模型对输入文本的评分列表
        """
        ...
