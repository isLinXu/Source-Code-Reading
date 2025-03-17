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
import sys
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers.utils.versions import require_version

from ..extras import logging  # 日志模块
from ..extras.constants import FILEEXT2TYPE  # 文件扩展名到类型的映射
from ..extras.misc import has_tokenized_data  # 检查是否存在预处理数据
from .aligner import align_dataset  # 数据对齐模块
from .data_utils import merge_dataset, split_dataset  # 数据集合并和分割工具
from .parser import get_dataset_list  # 数据集列表解析器
from .preprocess import get_preprocess_and_print_func  # 预处理和打印函数


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # 类型提示
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments  # 类型提示
    from ..hparams import DataArguments, ModelArguments  # 配置参数
    from .data_utils import DatasetModule  # 数据集模块类型
    from .parser import DatasetAttr  # 数据集属性
    from .template import Template  # 模板类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    加载单个数据集并转换为标准格式
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    # 根据数据来源设置路径参数
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:  # 从模型平台加载
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":  # 从脚本加载
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":  # 从本地文件加载
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # 处理目录
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # 处理单个文件
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        # 根据文件扩展名确定类型
        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        # 检查所有文件类型一致性
        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    # 处理不同平台的加载方式
    if dataset_attr.load_from == "ms_hub":  # ModelScope平台
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()  # 转换为HuggingFace格式

    elif dataset_attr.load_from == "om_hub":  # OpenMind平台
        require_version("openmind>=0.8.0", "To fix: pip install openmind>=0.8.0")
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    else:  # HuggingFace平台
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            trust_remote_code=True,
        )

    # 处理采样数量
    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # 随机采样
        target_num -= len(indexes)
        if target_num > 0:  # 补充采样
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    # 截断数据集
    if data_args.max_samples is not None:  
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)  # 对齐数据格式


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    获取合并后的标准格式数据集
    """
    if dataset_names is None:
        return None

    datasets = []
    # 遍历数据集列表并加载
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        # 检查数据集与训练阶段的兼容性
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))

    return merge_dataset(datasets, data_args, seed=training_args.seed)  # 合并数据集


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    预处理数据集，包括格式检查和tokenization
    """
    if dataset is None:
        return None

    # 获取预处理和打印函数
    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())  # 获取数据集列名
    kwargs = {}
    if not data_args.streaming:  # 非流式处理参数
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,  # 并行进程数
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),  # 缓存策略
            desc="Running tokenizer on dataset",  # 进度描述
        )

    # 应用预处理函数
    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,  # 批处理大小
        remove_columns=column_names,  # 移除原始列
        **kwargs,
    )

    # 打印样本示例
    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":  # 预训练阶段数据不足
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:  # 其他阶段数据格式错误
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    获取训练数据集和可选的验证数据集
    """
    # 加载已预处理的数据集
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            dataset_dict: "DatasetDict" = load_from_disk(data_args.tokenized_path)
            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]

            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if data_args.streaming:  # 转换为流式数据集
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # 加载和预处理数据集
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

    # 数据预处理流程
    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
        )

        # 划分验证集
        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        # 保存预处理后的数据集
        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please restart the training with `tokenized_path: {data_args.tokenized_path}`.")

            sys.exit(0)  # 保存后退出

        # 构建最终数据集模块
        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module
