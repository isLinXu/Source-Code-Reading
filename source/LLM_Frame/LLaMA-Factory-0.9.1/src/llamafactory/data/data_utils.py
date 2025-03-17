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

from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, TypedDict, Union

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from ..extras import logging  # 从上级目录导入日志模块


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # 类型检查时导入数据集类型
    from ..hparams import DataArguments  # 类型检查时导入数据参数


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]  # 定义槽位类型别名


@unique
class Role(str, Enum):
    USER = "user"  # 用户角色
    ASSISTANT = "assistant"  # 助手角色
    SYSTEM = "system"  # 系统角色
    FUNCTION = "function"  # 函数角色
    OBSERVATION = "observation"  # 观察结果角色


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]  # 训练数据集
    eval_dataset: Optional[Union["Dataset", "IterableDataset"]]  # 评估数据集


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    合并多个数据集为统一数据集
    """
    if len(all_datasets) == 1:  # 单个数据集直接返回
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":  # 拼接策略
        if data_args.streaming:  # 流式模式下警告样本不会混合
            logger.warning_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)  # 使用Hugging Face的拼接方法
    elif data_args.mix_strategy.startswith("interleave"):  # 交错混合策略
        if not data_args.streaming:  # 非流式模式建议使用concat
            logger.warning_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,  # 各数据集的采样概率
            seed=seed,  # 随机种子
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
            # 停止策略：以"under"结尾时任一数据集耗尽即停止，否则全部耗尽
        )
    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")  # 未知策略报错


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.
    划分数据集为训练集和验证集，返回DatasetDict对象

    Supports both map dataset and iterable dataset.
    支持普通数据集和流式数据集
    """
    if data_args.streaming:  # 流式数据集处理方式
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)  # 先进行shuffle
        val_set = dataset.take(int(data_args.val_size))  # 取前val_size条作为验证集
        train_set = dataset.skip(int(data_args.val_size))  # 跳过前val_size条作为训练集
        return DatasetDict({"train": train_set, "validation": val_set})
    else:  # 普通数据集处理方式
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size  # 确定验证集大小
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)  # 使用Hugging Face的划分方法
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})  # 返回划分后的数据集字典
