# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    用于定义模型训练和评估所需数据输入的参数。
    """

    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference. (用于在训练和推理中构建提示的模板)"},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets. (用于训练的数据集名称，使用逗号分隔多个数据集)"},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets. (用于评估的数据集名称，使用逗号分隔多个数据集)"},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets. (包含数据集的文件夹路径)"},
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing the images or videos. Defaults to `dataset_dir`. (包含图像或视频的文件夹路径，默认与dataset_dir相同)"},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset. (数据集中标记化输入的截断长度)"},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt. (是否禁用提示部分的掩码)"},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only. (是否掩盖历史对话并仅在最后一轮上训练)"},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming. (启用数据集流式处理)"},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming. (数据集流式处理中随机采样示例的缓冲区大小)"},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling). (数据集混合策略：连接/交错，欠采样/过采样)"},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets. (从数据集采样数据的概率，使用逗号分隔多个数据集)"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets. (是否覆盖缓存的训练和评估集)"},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing. (预处理中一组中的示例数量)"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing. (用于预处理的进程数)"},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset. (用于调试目的，截断每个数据集的示例数量)"},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate` (评估时使用的beam数量，此参数将传递给`model.generate`)"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation. (在损失计算中是否忽略与填充标签对应的token)"},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`. (开发集大小，应为整数或[0,1)范围内的浮点数)"},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training. (在训练中启用序列打包，在预训练中将自动启用)"},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention. (启用无交叉注意力的序列打包)"},
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples. (用于构建函数调用示例的工具格式)"},
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
                "（标记化数据集的保存或加载路径。如果路径不存在，将保存标记化数据集；如果路径存在，将加载标记化数据集。）"
            )
        },
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

        if self.image_dir is None:
            self.image_dir = self.dataset_dir

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None. (如果`dataset`为None，则不能指定`val_size`)")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None. (如果`eval_dataset`不为None，则不能指定`val_size`)")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing. (`interleave_probs`仅对交错混合有效)")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical. (数据集长度和交错概率长度应相同)")

            if self.eval_dataset is not None and len(self.eval_dataset) != len(self.interleave_probs):
                raise ValueError("The length of eval dataset and interleave probs should be identical. (评估数据集长度和交错概率长度应相同)")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size. (流式模式下val_size应为整数)")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`. (`max_samples`与`streaming`不兼容)")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`. (`mask_history`与`train_on_prompt`不兼容)")
