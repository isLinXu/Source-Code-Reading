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

from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple

from .processors.feedback import preprocess_feedback_dataset
from .processors.pairwise import preprocess_pairwise_dataset, print_pairwise_dataset_example
from .processors.pretrain import preprocess_pretrain_dataset
from .processors.supervised import (
    preprocess_packed_supervised_dataset,
    preprocess_supervised_dataset,
    print_supervised_dataset_example,
)
from .processors.unsupervised import preprocess_unsupervised_dataset, print_unsupervised_dataset_example


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import DataArguments
    from .template import Template


def get_preprocess_and_print_func(
    data_args: "DataArguments",          # 数据参数配置对象
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],  # 训练阶段：预训练/监督微调/奖励模型/PPO/KTO
    template: "Template",                # 模板处理器
    tokenizer: "PreTrainedTokenizer",    # 文本分词器
    processor: Optional["ProcessorMixin"],  # 多模态处理器（图像/视频）
    do_generate: bool = False            # 是否处于生成模式
) -> Tuple[Callable, Callable]:          # 返回预处理函数和示例打印函数
    # 根据训练阶段选择预处理函数
    if stage == "pt":  # 预训练阶段
        preprocess_func = partial(  # 部分应用参数，创建预处理函数
            preprocess_pretrain_dataset,  # 使用预训练数据集预处理
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)  # 无监督数据示例打印

    elif stage == "sft" and not do_generate:  # 监督微调且非生成模式
        if data_args.packing:  # 是否启用数据打包
            if data_args.neat_packing:  # 优化打包模式（hack数据集注意力掩码类型）
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                # 重写TypedSequence初始化方法，优化int类型存储
                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__  # 应用猴子补丁

            # 使用打包的监督学习预处理
            preprocess_func = partial(
                preprocess_packed_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        else:  # 普通监督学习预处理
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)  # 监督数据示例打印

    elif stage == "rm":  # 奖励模型阶段
        preprocess_func = partial(
            preprocess_pairwise_dataset,  # 成对数据预处理
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)  # 成对数据示例打印

    elif stage == "kto":  # KTO反馈学习阶段
        preprocess_func = partial(
            preprocess_feedback_dataset,  # 反馈数据预处理
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)  # 使用监督示例打印

    else:  # 其他情况（如生成模式）
        preprocess_func = partial(
            preprocess_unsupervised_dataset,  # 无监督数据预处理
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)  # 无监督示例打印

    return preprocess_func, print_function  # 返回预处理和打印函数
