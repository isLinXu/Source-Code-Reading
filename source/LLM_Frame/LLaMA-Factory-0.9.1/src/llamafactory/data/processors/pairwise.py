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
from ...extras.constants import IGNORE_INDEX  # 导入忽略索引常量
from .processor_utils import infer_seqlen  # 导入序列长度推断工具


if TYPE_CHECKING:  # 类型检查时导入
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)  # 获取日志记录器


def _encode_pairwise_example(
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
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """编码成对示例的私有函数"""
    # 处理选中和被拒绝的消息
    chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    # 编码一轮对话
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    # 如果启用了高效EOS，添加结束符
    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    # 处理多模态token ID
    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
    # 推断序列长度（优先保证响应部分）
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]  # 截断提示部分
    chosen_ids = chosen_ids[:target_len]  # 截断选中回复
    rejected_ids = rejected_ids[:target_len]  # 截断被拒绝回复

    # 构建输入ID和标签
    chosen_input_ids = prompt_ids + chosen_ids  # 选中样本的输入ID
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids  # 选中样本的标签
    rejected_input_ids = prompt_ids + rejected_ids  # 被拒绝样本的输入ID
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids  # 被拒绝样本的标签
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    """预处理成对数据集"""
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    # 构建输入对，格式为 `<bos> X`, `Y1 <eos>` 和 `Y2 <eos>`
    model_inputs = defaultdict(list)  # 使用默认字典存储模型输入
    for i in range(len(examples["_prompt"])):  # 遍历所有示例
        # 检查示例格式是否有效
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        # 编码成对示例
        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
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
        # 添加到模型输入字典
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    """打印成对数据集示例，用于调试"""
    # 过滤出有效的标签（非IGNORE_INDEX）
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    # 打印各种信息
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print(f"chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print(f"rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
