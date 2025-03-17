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

import asyncio
import concurrent.futures
import os
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import GenerationConfig, TextIteratorStreamer
from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.misc import get_logits_processor
from ..model import load_model, load_tokenizer
from .base_engine import BaseEngine, Response


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from trl import PreTrainedModelWrapper

    from ..data import Template
    from ..data.mm_plugin import ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class HuggingfaceEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",  # 模型参数（路径、精度等）
        data_args: "DataArguments",  # 数据参数（模板、工具格式等）
        finetuning_args: "FinetuningArguments",  # 微调参数（适配器路径等）
        generating_args: "GeneratingArguments",  # 生成参数（温度、top_p等）
    ) -> None:
        self.can_generate = finetuning_args.stage == "sft"  # 判断是否支持生成模式（仅SFT阶段）
        tokenizer_module = load_tokenizer(model_args)  # 加载分词器
        self.tokenizer = tokenizer_module["tokenizer"]  # 文本分词器
        self.processor = tokenizer_module["processor"]  # 多模态处理器（图像/视频）
        self.tokenizer.padding_side = "left" if self.can_generate else "right"  # 设置填充方向（生成模式左填充）
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)  # 获取对话模板并修复分词器
        self.model = load_model(
            self.tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=(not self.can_generate)
        )  # 加载模型（非训练模式，奖励模型需要添加value head）
        self.generating_args = generating_args.to_dict()  # 转换生成参数为字典格式
        
        # 确保事件循环存在
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            logger.warning_once("There is no current event loop, creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "1")))  # 并发控制信号量

    @staticmethod
    def _process_args(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],  # 原始对话消息
        system: Optional[str] = None,  # 系统提示
        tools: Optional[str] = None,  # 工具定义
        images: Optional[Sequence["ImageInput"]] = None,  # 图像输入
        videos: Optional[Sequence["VideoInput"]] = None,  # 视频输入
        input_kwargs: Optional[Dict[str, Any]] = {},  # 用户输入参数
    ) -> Tuple[Dict[str, Any], int]:
        # 处理多模态输入
        mm_input_dict = {"images": [], "videos": [], "imglens": [0], "vidlens": [0]}
        if images is not None:
            mm_input_dict.update({"images": images, "imglens": [len(images)]})
            if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]  # 自动添加图像占位符

        if videos is not None:
            mm_input_dict.update({"videos": videos, "vidlens": [len(videos)]})
            if not any(VIDEO_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = VIDEO_PLACEHOLDER * len(videos) + messages[0]["content"]  # 自动添加视频占位符

        # 处理多模态消息
        messages = template.mm_plugin.process_messages(
            messages, mm_input_dict["images"], mm_input_dict["videos"], processor
        )
        paired_messages = messages + [{"role": "assistant", "content": ""}]  # 添加空助手消息用于生成
        system = system or generating_args["default_system"]  # 使用默认系统提示
        
        # 编码对话为token id
        prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system, tools)
        prompt_ids, _ = template.mm_plugin.process_token_ids(  # 处理多模态token id
            prompt_ids, None, mm_input_dict["images"], mm_input_dict["videos"], tokenizer, processor
        )
        prompt_length = len(prompt_ids)  # 记录prompt长度
        
        # 准备模型输入
        inputs = torch.tensor([prompt_ids], device=model.device)  # 转换为张量
        attention_mask = torch.ones_like(inputs, dtype=torch.bool)  # 创建注意力掩码

        # 处理生成参数
        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        num_return_sequences = input_kwargs.pop("num_return_sequences", 1)  # 生成序列数
        
        # 合并用户参数和默认参数
        generating_args = generating_args.copy()
        generating_args.update(
            dict(
                do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
                temperature=temperature if temperature is not None else generating_args["temperature"],
                top_p=top_p if top_p is not None else generating_args["top_p"],
                top_k=top_k if top_k is not None else generating_args["top_k"],
                num_return_sequences=num_return_sequences,
                eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,  # 停止token列表
                pad_token_id=tokenizer.pad_token_id,
            )
        )

        # 处理生成模式逻辑
        if num_return_sequences > 1:  # 多序列生成需要采样
            generating_args["do_sample"] = True
            generating_args["temperature"] = generating_args["temperature"] or 1.0  # 确保温度有效

        if not generating_args["temperature"]:  # 温度=0时关闭采样
            generating_args["do_sample"] = False

        # 准备最终生成参数
        gen_kwargs = dict(
            inputs=inputs,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(**generating_args),  # 转换为生成配置
            logits_processor=get_logits_processor(),  # 添加logits处理器
        )

        # 处理多模态输入
        mm_inputs = template.mm_plugin.get_mm_inputs(**mm_input_dict, batch_ids=[prompt_ids], processor=processor)
        for key, value in mm_inputs.items():
            if isinstance(value, list) and all(isinstance(v, torch.Tensor) for v in value):  # 处理多模态张量列表
                value = torch.stack(value)  # 堆叠为批次张量
            elif not isinstance(value, torch.Tensor):
                value = torch.tensor(value)  # 转换为张量

            gen_kwargs[key] = value.to(model.device)  # 移动到模型设备

        return gen_kwargs, prompt_length  # 返回生成参数和prompt长度

    @staticmethod
    @torch.inference_mode()
    def _chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> List["Response"]:
        gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
            model,
            tokenizer,
            processor,
            template,
            generating_args,
            messages,
            system,
            tools,
            images,
            videos,
            input_kwargs,
        )
        generate_output = model.generate(**gen_kwargs)
        response_ids = generate_output[:, prompt_length:]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results = []
        for i in range(len(response)):
            eos_index = (response_ids[i] == tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[i])
            results.append(
                Response(
                    response_text=response[i],
                    response_length=response_length,
                    prompt_length=prompt_length,
                    finish_reason="stop" if len(eos_index) else "length",
                )
            )

        return results

    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Callable[[], str]:
        # 准备生成参数
        gen_kwargs, _ = HuggingfaceEngine._process_args(
            model, tokenizer, processor, template, generating_args, messages, system, tools, images, videos, input_kwargs
        )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 创建流式生成器
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)  # 后台生成线程
        thread.start()

        def stream():
            try:
                return streamer.__next__()  # 获取下一个token
            except StopIteration:
                raise StopAsyncIteration()

        return stream  # 返回流式生成函数

    @staticmethod
    @torch.inference_mode()
    def _get_scores(
        model: "PreTrainedModelWrapper",
        tokenizer: "PreTrainedTokenizer",
        batch_input: List[str],
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> List[float]:
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        device = getattr(model.pretrained_model, "device", "cuda")
        inputs: Dict[str, "torch.Tensor"] = tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            max_length=max_length or getattr(model.config, "max_position_embeddings", 1024),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        values: "torch.Tensor" = model(**inputs, return_dict=True, use_cache=False)[-1]
        scores = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return scores

    @override
    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> List["Response"]:
        if not self.can_generate:
            raise ValueError("The current model does not support `chat`.")  # 检查生成能力

        loop = asyncio.get_running_loop()  # 获取当前事件循环
        async with self.semaphore:  # 并发控制
            with concurrent.futures.ThreadPoolExecutor() as pool:  # 线程池执行同步方法
                return await loop.run_in_executor(
                    pool, 
                    self._chat,  # 实际调用同步生成方法
                    self.model, self.tokenizer, self.processor, self.template, 
                    self.generating_args, messages, system, tools, images, videos, input_kwargs
                )

    @override
    async def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        if not self.can_generate:
            raise ValueError("The current model does not support `stream_chat`.")

        loop = asyncio.get_running_loop()
        async with self.semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                stream = self._stream_chat(  # 获取流式生成函数
                    self.model, self.tokenizer, self.processor, self.template, 
                    self.generating_args, messages, system, tools, images, videos, input_kwargs
                )
                while True:
                    try:
                        yield await loop.run_in_executor(pool, stream)  # 异步获取每个token
                    except StopAsyncIteration:
                        break

    @override
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        if self.can_generate:
            raise ValueError("Cannot get scores using an auto-regressive model.")

        loop = asyncio.get_running_loop()
        input_args = (self.model, self.tokenizer, batch_input, input_kwargs)
        async with self.semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, self._get_scores, *input_args)
