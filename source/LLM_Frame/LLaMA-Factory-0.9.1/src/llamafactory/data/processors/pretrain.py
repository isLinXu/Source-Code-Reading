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

from itertools import chain  # 导入chain用于连接迭代器
from typing import TYPE_CHECKING, Any, Dict, List  # 导入类型提示

# 类型检查时导入的类型
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ...hparams import DataArguments


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]],  # 输入示例字典
    tokenizer: "PreTrainedTokenizer",  # 分词器
    data_args: "DataArguments"  # 数据参数
) -> Dict[str, List[Any]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    # 如果启用了打包功能，则构建格式为 `X1 X2 X3 ...` 的分组文本
    
    # 根据模板类型选择结束符
    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    # 为每个示例添加结束符
    text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

    if not data_args.packing:  # 如果不启用打包
        if data_args.template == "gemma":  # 如果是Gemma模板
            # 为每个示例添加开始符
            text_examples = [tokenizer.bos_token + example for example in text_examples]

        # 对文本进行分词，不添加特殊标记，并进行长度截断
        result = tokenizer(text_examples, add_special_tokens=False, truncation=True, max_length=data_args.cutoff_len)
    else:  # 如果启用打包
        # 对文本进行分词，不添加特殊标记
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        # 将所有示例连接成一个长序列
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        # 获取总长度
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len  # 获取块大小
        # 调整总长度为块大小的整数倍
        total_length = (total_length // block_size) * block_size
        # 将连接后的序列切分成固定大小的块
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if data_args.template == "gemma":  # 如果是Gemma模板
            # 为每个块的开始添加BOS标记
            for i in range(len(result["input_ids"])):
                result["input_ids"][i][0] = tokenizer.bos_token_id

    return result
