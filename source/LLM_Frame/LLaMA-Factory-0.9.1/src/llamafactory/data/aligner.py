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

import os
from functools import partial  # 导入偏函数工具
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union  # 导入类型提示

from ..extras import logging  # 导入日志模块
from .data_utils import Role  # 导入角色枚举


if TYPE_CHECKING:  # 类型检查时导入
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import ImageInput, VideoInput
    from .parser import DatasetAttr


logger = logging.get_logger(__name__)  # 获取日志记录器


def _convert_images(
    images: Union["ImageInput", Sequence["ImageInput"]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Optional[List["ImageInput"]]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    当从本地磁盘加载时，可选地将图像路径连接到数据集目录
    """
    # 处理单个图像输入的情况，转换为列表
    if not isinstance(images, list):
        images = [images]
    elif len(images) == 0:  # 空列表直接返回None
        return None
    else:  # 创建列表的浅拷贝以避免修改原始数据
        images = images[:]

    # 当从脚本或文件加载时，处理本地文件路径
    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(images)):
            # 检查是否是字符串路径且文件存在
            if isinstance(images[i], str) and os.path.isfile(os.path.join(data_args.image_dir, images[i])):
                images[i] = os.path.join(data_args.image_dir, images[i])  # 拼接完整路径

    return images


def _convert_videos(
    videos: Union["VideoInput", Sequence["VideoInput"]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Optional[List["VideoInput"]]:
    r"""
    Optionally concatenates video path to dataset dir when loading from local disk.
    当从本地磁盘加载时，可选地将视频路径连接到数据集目录
    """
    # 处理逻辑与图像转换相同，保持结构一致性
    if not isinstance(videos, list):
        videos = [videos]
    elif len(videos) == 0:
        return None
    else:
        videos = videos[:]

    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(videos)):
            if isinstance(videos[i], str) and os.path.isfile(os.path.join(data_args.image_dir, videos[i])):
                videos[i] = os.path.join(data_args.image_dir, videos[i])  # 注意：视频使用image_dir可能需要调整

    return videos


def convert_alpaca(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts alpaca format dataset to the standard format.
    将Alpaca格式数据集转换为标准格式
    """
    prompt = []
    # 处理历史对话记录
    if dataset_attr.history and isinstance(example[dataset_attr.history], list):
        for old_prompt, old_response in example[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})  # 用户历史消息
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})  # 助手历史回复

    # 构建当前查询
    query = []
    if dataset_attr.prompt and example[dataset_attr.prompt]:  # 基础提示词
        query.append(example[dataset_attr.prompt])
    if dataset_attr.query and example[dataset_attr.query]:  # 具体查询内容
        query.append(example[dataset_attr.query])
    prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # 合并为单个用户消息

    # 处理不同响应类型
    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # KTO对比数据
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
        # 根据KTO标签调整响应顺序
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]  # 正例在后
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response  # 负例在前
    elif (  # 排序任务数据
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], str)
        and isinstance(example[dataset_attr.rejected], str)
    ):
        response = [
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.chosen]},  # 优选回复
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.rejected]},  # 次选回复
        ]
    elif dataset_attr.response and isinstance(example[dataset_attr.response], str):  # 普通单响应
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
    else:  # 无监督学习情况
        response = []

    # 使用偏函数预先绑定参数
    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    
    # 构建最终输出结构
    output = {
        "_prompt": prompt,  # 完整对话历史
        "_response": response,  # 响应内容
        "_system": example[dataset_attr.system] if dataset_attr.system else "",  # 系统指令
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",  # 工具定义
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,  # 图像数据
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,  # 视频数据
    }
    return output


def convert_sharegpt(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    将ShareGPT格式数据集转换为标准格式
    """
    # 角色标签映射表
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    # 定义对话轮次的合法角色顺序
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)  # 奇数轮次允许的角色
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)  # 偶数轮次允许的角色
    accept_tags = (odd_tags, even_tags)  # 组合成检查元组
    
    messages = example[dataset_attr.messages]  # 原始消息列表
    
    # 提取系统消息（如果存在）
    if (
        dataset_attr.system_tag
        and len(messages) != 0
        and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]  # 提取系统消息内容
        messages = messages[1:]  # 剩余消息
    else:
        system = example[dataset_attr.system] if dataset_attr.system else ""  # 备用系统消息字段

    aligned_messages = []  # 对齐后的消息列表
    broken_data = False  # 数据异常标志
    # 遍历并验证每条消息
    for turn_idx, message in enumerate(messages):
        # 检查角色顺序是否合法
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning_rank0(f"Invalid role tag in {messages}.")
            broken_data = True

        # 转换角色标签并保留内容
        aligned_messages.append({
            "role": tag_mapping[message[dataset_attr.role_tag]],
            "content": message[dataset_attr.content_tag]
        })

    # 验证消息数量是否符合要求
    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        logger.warning_rank0(f"Invalid message count in {messages}.")
        broken_data = True

    # 处理不同任务类型
    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # KTO数据
        prompt = aligned_messages[:-1]  # 历史对话作为提示
        response = aligned_messages[-1:]  # 最后一条作为响应
        # 根据KTO标签添加空响应
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (  # 排序任务数据
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], dict)
        and isinstance(example[dataset_attr.rejected], dict)
    ):
        chosen = example[dataset_attr.chosen]  # 优选回复
        rejected = example[dataset_attr.rejected]  # 次选回复
        # 验证角色标签合法性
        if (
            chosen[dataset_attr.role_tag] not in accept_tags[-1]
            or rejected[dataset_attr.role_tag] not in accept_tags[-1]
        ):
            logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
            broken_data = True

        prompt = aligned_messages  # 全部消息作为提示
        response = [  # 构建对比响应
            {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
        ]
    else:  # 普通对话数据
        prompt = aligned_messages[:-1]  # 历史对话作为提示
        response = aligned_messages[-1:]  # 最后一条作为响应

    # 处理异常数据
    if broken_data:
        logger.warning_rank0("Skipping this abnormal example.")
        prompt, response = [], []  # 清空异常数据

    # 多媒体数据处理（同alpaca格式）
    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    
    # 构建输出结构
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
    """
    # 根据数据格式选择转换函数
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

    # 获取数据集列名用于后续删除
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    # 非流式处理时的参数配置
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,  # 并行进程数
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),  # 缓存策略
            desc="Converting format of dataset",  # 进度条描述
        )

    # 执行数据集转换
    return dataset.map(
        convert_func,
        batched=False,  # 单样本处理模式
        remove_columns=column_names,  # 移除原始列
        **kwargs,
    )
