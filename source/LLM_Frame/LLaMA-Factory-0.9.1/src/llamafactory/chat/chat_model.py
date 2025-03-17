# Copyright 2024 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
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

import asyncio
import os
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from ..extras.misc import torch_gc
from ..hparams import get_infer_args
from .hf_engine import HuggingfaceEngine
from .vllm_engine import VllmEngine


if TYPE_CHECKING:
    from ..data.mm_plugin import ImageInput, VideoInput
    from .base_engine import BaseEngine, Response


# 启动后台事件循环的辅助函数
def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)  # 设置当前线程的事件循环
    loop.run_forever()  # 启动事件循环的无限运行


# 聊天模型主类
class ChatModel:
    r"""
    General class for chat models. Backed by huggingface or vllm engines.
    聊天模型主类，支持HuggingFace和vLLM两种后端引擎
    """

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        # 解析推理参数
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        # 根据后端类型初始化引擎
        self.engine_type = model_args.infer_backend
        if model_args.infer_backend == "huggingface":
            self.engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == "vllm":
            self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")

        # 创建异步事件循环和后台线程
        self._loop = asyncio.new_event_loop()  # 创建新的事件循环
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)  # 后台守护线程
        self._thread.start()  # 启动线程运行事件循环

    # 同步聊天方法
    def chat(
        self,
        messages: Sequence[Dict[str, str]],  # 对话历史
        system: Optional[str] = None,  # 系统提示
        tools: Optional[str] = None,  # 工具定义
        images: Optional[Sequence["ImageInput"]] = None,  # 图像输入
        videos: Optional[Sequence["VideoInput"]] = None,  # 视频输入
        **input_kwargs,  # 其他生成参数
    ) -> List["Response"]:
        r"""
        Gets a list of responses of the chat model.
        同步获取聊天模型的响应列表
        """
        # 将异步方法包装为同步调用
        task = asyncio.run_coroutine_threadsafe(
            self.achat(messages, system, tools, images, videos, **input_kwargs), self._loop
        )
        return task.result()  # 等待并返回结果

    # 异步聊天方法
    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> List["Response"]:
        r"""
        Asynchronously gets a list of responses of the chat model.
        异步获取聊天模型的响应列表
        """
        return await self.engine.chat(messages, system, tools, images, videos, **input_kwargs)  # 调用引擎的异步方法

    # 同步流式聊天方法
    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        r"""
        Gets the response token-by-token of the chat model.
        同步流式获取逐token响应
        """
        generator = self.astream_chat(messages, system, tools, images, videos, **input_kwargs)  # 获取异步生成器
        while True:
            try:
                # 将异步生成转换为同步生成
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()  # 逐token生成结果
            except StopAsyncIteration:  # 生成结束
                break

    # 异步流式聊天方法
    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r"""
        Asynchronously gets the response token-by-token of the chat model.
        异步流式生成逐token响应
        """
        async for new_token in self.engine.stream_chat(messages, system, tools, images, videos, **input_kwargs):
            yield new_token  # 逐token生成

    # 同步评分方法
    def get_scores(
        self,
        batch_input: List[str],  # 输入文本列表
        **input_kwargs,
    ) -> List[float]:
        r"""
        Gets a list of scores of the reward model.
        同步获取奖励模型评分
        """
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    # 异步评分方法
    async def aget_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        r"""
        Asynchronously gets a list of scores of the reward model.
        异步获取奖励模型评分
        """
        return await self.engine.get_scores(batch_input, **input_kwargs)  # 调用引擎评分方法


# 命令行交互入口函数
def run_chat() -> None:
    # 非Windows系统尝试加载readline提升输入体验
    if os.name != "nt":
        try:
            import readline  # 提供命令行历史记录等功能
        except ImportError:
            print("Install `readline` for a better experience.")

    chat_model = ChatModel()  # 初始化聊天模型
    messages = []  # 对话历史存储
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:  # 交互主循环
        try:
            query = input("\nUser: ")  # 获取用户输入
        except UnicodeDecodeError:  # 处理编码错误
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue

        if query.strip() == "exit":  # 退出命令
            break

        if query.strip() == "clear":  # 清空历史
            messages = []
            torch_gc()  # 释放显存
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})  # 添加用户消息到历史
        print("Assistant: ", end="", flush=True)  # 打印助手前缀

        response = ""
        # 流式生成响应
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)  # 逐token打印
            response += new_text  # 累积完整响应
        print()
        messages.append({"role": "assistant", "content": response})  # 添加助手响应到历史
