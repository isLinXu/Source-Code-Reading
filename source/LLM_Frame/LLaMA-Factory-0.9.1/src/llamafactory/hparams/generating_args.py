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

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GeneratingArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    用于指定解码参数的参数类。
    """

    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise. (是否使用采样，否则使用贪婪解码)"},
    )
    temperature: float = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities. (用于调节下一个token概率的温度值)"},
    )
    top_p: float = field(
        default=0.7,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept. (保留累积概率达到top_p或更高的最小可能token集合)"
        },
    )
    top_k: int = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering. (用于top-k过滤的保留的最高概率词汇token数量)"},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search. (束搜索的束数量，1表示不使用束搜索)"},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens. (生成token的最大长度，可被max_new_tokens覆盖)"},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. (要生成的最大token数量，忽略提示中的token数量)"},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty. (重复惩罚参数，1.0表示无惩罚)"},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation. (用于基于束的生成的长度指数惩罚)"},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Default system message to use in chat completion. (聊天完成中使用的默认系统消息)"},
    )

    def to_dict(self) -> Dict[str, Any]:
        # 将参数转换为字典，并处理max_length和max_new_tokens之间的关系
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)  # 如果设置了max_new_tokens，则移除max_length
        else:
            args.pop("max_new_tokens", None)  # 否则移除max_new_tokens
        return args
