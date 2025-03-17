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

import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Sequence, Union

from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import IMAGE_PLACEHOLDER
from ..extras.misc import get_device_count
from ..extras.packages import is_pillow_available, is_vllm_available
from ..model import load_config, load_tokenizer
from ..model.model_utils.quantization import QuantizationMethod
from ..model.model_utils.visual import LlavaMultiModalProjectorForYiVLForVLLM
from .base_engine import BaseEngine, Response


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_vllm_available():
    from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest


if TYPE_CHECKING:
    from ..data.mm_plugin import ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class VllmEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",  # 模型参数（路径、精度等）
        data_args: "DataArguments",  # 数据参数（模板、工具格式等）
        finetuning_args: "FinetuningArguments",  # 微调参数（适配器路径等）
        generating_args: "GeneratingArguments",  # 生成参数（温度、top_p等）
    ) -> None:
        config = load_config(model_args)  # 加载模型配置（可能从魔搭社区下载）
        # 处理GPTQ量化模型的精度设置
        if getattr(config, "quantization_config", None):  
            quantization_config: Dict[str, Any] = getattr(config, "quantization_config", None)
            quant_method = quantization_config.get("quant_method", "")
            if quant_method == QuantizationMethod.GPTQ and model_args.infer_dtype == "auto":
                model_args.infer_dtype = "float16"  # GPTQ模型默认使用float16

        self.can_generate = finetuning_args.stage == "sft"  # 判断是否支持生成模式
        tokenizer_module = load_tokenizer(model_args)  # 加载分词器
        self.tokenizer = tokenizer_module["tokenizer"]  # 文本分词器
        self.processor = tokenizer_module["processor"]  # 多模态处理器
        self.tokenizer.padding_side = "left"  # vLLM强制左填充
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)  # 获取对话模板并修复分词器
        self.generating_args = generating_args.to_dict()  # 转换生成参数为字典格式

        # 配置vLLM引擎参数
        engine_args = {
            "model": model_args.model_name_or_path,  # 模型路径
            "trust_remote_code": True,  # 信任远程代码
            "download_dir": model_args.cache_dir,  # 缓存目录
            "dtype": model_args.infer_dtype,  # 推理精度
            "max_model_len": model_args.vllm_maxlen,  # 最大模型长度
            "tensor_parallel_size": get_device_count() or 1,  # 张量并行度
            "gpu_memory_utilization": model_args.vllm_gpu_util,  # GPU显存利用率
            "disable_log_stats": True,  # 禁用统计日志
            "disable_log_requests": True,  # 禁用请求日志
            "enforce_eager": model_args.vllm_enforce_eager,  # 强制eager模式
            "enable_lora": model_args.adapter_name_or_path is not None,  # 启用LoRA
            "max_lora_rank": model_args.vllm_max_lora_rank,  # 最大LoRA秩
        }
        if isinstance(model_args.vllm_config, dict):  # 合并自定义vLLM配置
            engine_args.update(model_args.vllm_config)

        # 处理Yi-VL视觉语言模型的特殊投影层
        if getattr(config, "is_yi_vl_derived_model", None):
            import vllm.model_executor.models.llava
            logger.info_rank0("Detected Yi-VL model, applying projector patch.")
            vllm.model_executor.models.llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVLForVLLM

        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))  # 创建异步引擎
        # 配置LoRA适配器
        if model_args.adapter_name_or_path is not None:
            self.lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])  # LoRA请求参数
        else:
            self.lora_request = None

    async def _generate(
        self,
        messages: Sequence[Dict[str, str]],  # 对话历史
        system: Optional[str] = None,  # 系统提示
        tools: Optional[str] = None,  # 工具定义
        images: Optional[Sequence["ImageInput"]] = None,  # 图像输入
        videos: Optional[Sequence["VideoInput"]] = None,  # 视频输入
        **input_kwargs,
    ) -> AsyncIterator["RequestOutput"]:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"  # 生成唯一请求ID
        # 处理图像占位符
        if images is not None:
            if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]  # 自动添加占位符

        # 处理不同多模态模板的特殊标记
        if self.template.mm_plugin.__class__.__name__ == "Qwen2vlPlugin":  # 通义千问视觉插件特殊处理
            image_str = f"<|vision_start|>{self.template.mm_plugin.image_token}<|vision_end|>"
        else:
            image_str = self.template.mm_plugin.image_token or ""

        # 构建对话结构并编码
        paired_messages = [
            {"role": message["role"], "content": message["content"].replace(IMAGE_PLACEHOLDER, image_str)}
            for message in messages
        ] + [{"role": "assistant", "content": ""}]  # 添加空助手消息
        system = system or self.generating_args["default_system"]  # 使用默认系统提示
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages, system, tools)  # 编码对话
        prompt_length = len(prompt_ids)  # 记录prompt长度

        # 解析生成参数
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        num_return_sequences = input_kwargs.pop("num_return_sequences", 1)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)
        stop = input_kwargs.pop("stop", None)  # 停止条件

        # 计算最大生成长度
        max_tokens = self.generating_args.get("max_new_tokens") or 1
        if max_length:
            max_tokens = max_length - prompt_length if max_length > prompt_length else 1
        if max_new_tokens:
            max_tokens = max_new_tokens

        # 配置采样参数
        sampling_params = SamplingParams(
            n=num_return_sequences,  # 生成序列数
            repetition_penalty=repetition_penalty or self.generating_args["repetition_penalty"] or 1.0,  # 重复惩罚
            temperature=temperature or self.generating_args["temperature"],  # 温度参数
            top_p=(top_p or self.generating_args["top_p"]) or 1.0,  # 核心采样参数
            top_k=top_k or self.generating_args["top_k"],  # 前k采样
            stop=stop,  # 停止条件
            stop_token_ids=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,  # 停止token
            max_tokens=max_tokens,  # 最大生成token数
            skip_special_tokens=True,  # 跳过特殊token
        )

        # 处理图像输入
        multi_modal_data = None
        if images is not None:
            image_data = []
            for image in images:
                if isinstance(image, str):  # 图像路径转换为PIL.Image
                    image = Image.open(image).convert("RGB")
                image_data.append(image)
            multi_modal_data = {"image": image_data}  # 多模态数据格式

        # 启动异步生成
        result_generator = self.model.generate(
            {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data},  # 输入数据
            sampling_params=sampling_params,  # 采样参数
            request_id=request_id,  # 请求ID
            lora_request=self.lora_request,  # LoRA适配器请求
        )
        return result_generator  # 返回生成器

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
        final_output = None
        generator = await self._generate(messages, system, tools, images, videos, **input_kwargs)
        async for request_output in generator:  # 等待生成完成
            final_output = request_output  # 获取最终输出

        results = []
        for output in final_output.outputs:  # 遍历所有生成结果
            results.append(
                Response(
                    response_text=output.text,  # 生成文本
                    response_length=len(output.token_ids),  # 响应长度
                    prompt_length=len(final_output.prompt_token_ids),  # prompt长度
                    finish_reason=output.finish_reason,  # 停止原因
                )
            )
        return results

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
        generated_text = ""  # 已生成文本缓存
        generator = await self._generate(messages, system, tools, images, videos, **input_kwargs)
        async for result in generator:  # 实时获取生成结果
            delta_text = result.outputs[0].text[len(generated_text) :]  # 计算增量文本
            generated_text = result.outputs[0].text  # 更新缓存
            yield delta_text  # 流式返回增量内容

    @override
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        raise NotImplementedError("vLLM engine does not support get_scores.")  # 明确不支持评分功能
