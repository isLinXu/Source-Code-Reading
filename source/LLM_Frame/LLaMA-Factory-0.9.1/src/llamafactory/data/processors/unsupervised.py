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

from collections import defaultdict  # 导入默认字典
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple  # 导入类型提示

from ...extras import logging  # 导入日志模块
from ..data_utils import Role  # 导入角色枚举
from .processor_utils import infer_seqlen  # 导入序列长度推断工具


if TYPE_CHECKING:  # 类型检查时导入
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)  # 获取日志记录器


def _encode_unsupervised_example(
    prompt: Sequence[Dict[str, str]],  # 提示序列
    response: Sequence[Dict[str, str]],  # 回复序列
    system: Optional[str],  # 系统提示
    tools: Optional[str],  # 工具描述
    images: Sequence["ImageInput"],  # 图像输入
    videos: Sequence["VideoInput"],  # 视频输入
    template: "Template",  # 模板
    tokenizer: "PreTrainedTokenizer",  # 分词器
    processor: Optional["ProcessorMixin"],  # 处理器
    cutoff_len: int,  # 截断长度
) -> Tuple[List[int], List[int]]:
    """编码无监督学习示例"""
    # 处理消息内容
    if len(response) == 1:  # 如果有回复内容
        messages = prompt + response
    else:  # 如果无回复内容
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]  # 添加空回复

    # 处理多模态消息
    messages = template.mm_plugin.process_messages(messages, images, videos, processor)
    # 编码单轮对话
    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    
    # 添加结束符（如果需要）
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    # 处理多模态token ID
    input_ids, _ = template.mm_plugin.process_token_ids(input_ids, None, images, videos, tokenizer, processor)
    # 推断序列长度
    source_len, target_len = infer_seqlen(len(input_ids), len(labels), cutoff_len)
    # 截断处理
    input_ids = input_ids[:source_len]
    labels = labels[:target_len]
    return input_ids, labels


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    # 构建格式为 `<bos> X` 的输入和格式为 `Y <eos>` 的标签
    model_inputs = defaultdict(list)  # 初始化模型输入字典

    # 处理每个示例
    for i in range(len(examples["_prompt"])):
        # 验证示例格式有效性
        if len(examples["_prompt"][i]) % 2 != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        # 编码无监督示例
        input_ids, labels = _encode_unsupervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )

        # 添加到模型输入
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    """打印无监督数据集示例"""
    print("input_ids:\n{}".format(example["input_ids"]))  # 打印输入ID
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))  # 解码输入文本
