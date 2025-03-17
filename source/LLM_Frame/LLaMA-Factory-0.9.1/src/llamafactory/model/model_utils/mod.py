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

from typing import TYPE_CHECKING  # 导入类型检查相关的模块

from ...extras.constants import MOD_SUPPORTED_MODELS  # 从 extras.constants 导入支持的模型常量


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig, PreTrainedModel  # 导入预训练配置和模型的类型

    from ...hparams import ModelArguments  # 导入模型参数的类型


def load_mod_pretrained_model(**init_kwargs) -> "PreTrainedModel":
    from MoD import AutoMoDModelForCausalLM  # 从 MoD 导入自动模型类

    return AutoMoDModelForCausalLM.from_pretrained(**init_kwargs)  # 使用提供的初始化参数加载预训练模型


def convert_pretrained_model_to_mod(
    model: "PreTrainedModel", config: "PretrainedConfig", model_args: "ModelArguments"
) -> "PreTrainedModel":
    from MoD import apply_mod_to_hf  # 从 MoD 导入应用模型到 Hugging Face 的函数

    if getattr(config, "model_type", None) not in MOD_SUPPORTED_MODELS:  # 检查模型类型是否在支持的模型列表中
        raise ValueError("Current model is not supported by mixture-of-depth.")  # 如果不支持，抛出错误

    model = apply_mod_to_hf(model)  # 应用模型转换
    model = model.to(model_args.compute_dtype)  # 将模型转换为指定的计算数据类型
    return model  # 返回转换后的模型
