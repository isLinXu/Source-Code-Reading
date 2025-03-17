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
from .processor_utils import greedy_knapsack, infer_seqlen  # 导入工具函数


if TYPE_CHECKING:  # 类型检查时导入
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)  # 获取日志记录器


def _encode_supervised_example(
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
    train_on_prompt: bool,  # 是否在提示上训练
    mask_history: bool,  # 是否遮蔽历史
) -> Tuple[List[int], List[int]]:
    """编码监督学习示例"""
    # 处理消息
    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    # 处理多模态token
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    # 编码多轮对话
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    
    if mask_history:  # 如果需要遮蔽历史
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns
                                           # 反转序列，优先处理最后几轮对话

    # 处理每一轮对话
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:  # 如果已达到截断长度则停止
            break

        # 推断当前轮的序列长度
        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]  # 截断源序列
        target_ids = target_ids[:target_len]  # 截断目标序列
        total_length += source_len + target_len  # 更新总长度

        # 根据训练设置处理源序列标签
        if train_on_prompt:  # 如果在提示上训练
            source_label = source_ids
        elif template.efficient_eos:  # 如果使用高效EOS
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        # 处理目标序列标签
        if mask_history and turn_idx != 0:  # train on the last turn only
                                           # 只在最后一轮训练
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        # 根据历史遮蔽设置组合序列
        if mask_history:  # reversed sequences
                         # 反转序列的情况
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    # 添加结束符（如果需要）
    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    # 构建格式为 `<bos> X Y <eos>` 的输入和格式为 `<ignore> ... <ignore> Y <eos>` 的标签
    # 对于多轮对话示例，我们只遮蔽每对提示-回复中的提示部分
    model_inputs = defaultdict(list)  # 初始化模型输入字典

    # 处理每个示例
    for i in range(len(examples["_prompt"])):
        # 验证示例格式
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        # 编码监督学习示例
        input_ids, labels = _encode_supervised_example(
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
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )

        # 添加到模型输入
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    # 构建格式为 `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>` 的输入
    # 和格式为 `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>` 的标签
    
    valid_num = 0  # 有效示例计数器
    batch_input_ids, batch_labels, batch_images, batch_videos = [], [], [], []  # 批量存储容器
    lengths = []  # 存储每个示例的长度
    length2indexes = defaultdict(list)  # 按长度分类存储索引的字典

    # 第一遍处理：收集有效示例
    for i in range(len(examples["_prompt"])):
        # 验证示例格式有效性
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        # 编码单个示例
        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len - 1,  # 为填充token预留空间
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        length = len(input_ids)
        # 长度验证
        if length > data_args.cutoff_len:
            logger.warning_rank0(f"Dropped lengthy example with length {length} > {data_args.cutoff_len}.")
        else:
            lengths.append(length)  # 记录长度
            length2indexes[length].append(valid_num)  # 按长度分类存储索引
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            valid_num += 1

    # 第二遍处理：打包示例
    model_inputs = defaultdict(list)
    # 使用贪心算法进行打包
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # 为填充token预留空间
    
    for knapsack in knapsacks:  # 处理每个打包块
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        packed_images, packed_videos = [], []
        
        # 组装打包块
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()  # 取出对应长度的示例索引
            packed_input_ids += batch_input_ids[index]  # 拼接输入ID
            packed_labels += batch_labels[index]  # 拼接标签
            packed_images += batch_images[index]  # 拼接图像
            packed_videos += batch_videos[index]  # 拼接视频
            
            # 生成注意力掩码
            if data_args.neat_packing:  # 整洁打包模式
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # 从1开始编号
            else:  # 普通模式
                packed_attention_masks += [1] * len(batch_input_ids[index])

        # 填充处理
        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length  # 填充输入ID
            packed_labels += [IGNORE_INDEX] * pad_length  # 填充标签
            # 填充注意力掩码
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length  # 整洁模式用0填充
            else:
                packed_attention_masks += [1] * pad_length  # 普通模式用1填充（优化注意力计算）

        # 长度验证
        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        # 存储最终结果
        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)
        model_inputs["images"].append(packed_images or None)  # 空列表转为None
        model_inputs["videos"].append(packed_videos or None)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    """打印监督学习数据集示例"""
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))  # 过滤有效标签
    print("input_ids:\n{}".format(example["input_ids"]))  # 打印原始输入ID
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))  # 解码输入文本
    print("label_ids:\n{}".format(example["labels"]))  # 打印原始标签ID
    print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")  # 解码有效标签文本
