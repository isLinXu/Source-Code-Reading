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
from dataclasses import dataclass, field
from typing import Literal, Optional

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""
    Arguments pertaining to specify the evaluation parameters.
    用于指定评估参数的参数类。
    """

    task: str = field(
        metadata={"help": "Name of the evaluation task. (评估任务的名称)"},
    )
    task_dir: str = field(
        default="evaluation",
        metadata={"help": "Path to the folder containing the evaluation datasets. (包含评估数据集的文件夹路径)"},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU for evaluation. (评估时每个GPU的批处理大小)"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders. (用于数据加载器的随机种子)"},
    )
    lang: Literal["en", "zh"] = field(
        default="en",
        metadata={"help": "Language used at evaluation. (评估使用的语言)"},
    )
    n_shot: int = field(
        default=5,
        metadata={"help": "Number of examplars for few-shot learning. (小样本学习的样例数量)"},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the evaluation results. (保存评估结果的路径)"},
    )
    download_mode: DownloadMode = field(
        default=DownloadMode.REUSE_DATASET_IF_EXISTS,
        metadata={"help": "Download mode used for the evaluation datasets. (用于评估数据集的下载模式)"},
    )

    def __post_init__(self):
        # 检查保存目录是否已存在，如果存在则抛出错误
        if self.save_dir is not None and os.path.exists(self.save_dir):
            raise ValueError("`save_dir` already exists, use another one. (保存目录已存在，请使用另一个目录)")
