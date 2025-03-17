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

import inspect  # 导入 inspect 模块，用于获取对象的信息
from typing import TYPE_CHECKING  # 导入类型检查相关的模块

from ...extras import logging  # 从 extras 模块导入 logging


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig  # 导入预训练配置的类型
    from ...hparams import ModelArguments  # 导入模型参数的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def apply_liger_kernel(
    config: "PretrainedConfig",  # 预训练配置
    model_args: "ModelArguments",  # 模型参数
    is_trainable: bool,  # 是否可训练的标志
    require_logits: bool,  # 是否需要 logits 的标志
) -> None:
    if not is_trainable or not model_args.enable_liger_kernel:  # 如果不可训练或未启用 liger kernel
        return  # 直接返回

    model_type = getattr(config, "model_type", None)  # 获取模型类型
    if model_type == "gemma":  # 如果模型类型为 gemma
        from liger_kernel.transformers import apply_liger_kernel_to_gemma as apply_liger_kernel  # 导入对应的函数
    elif model_type == "gemma2":  # 如果模型类型为 gemma2
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2 as apply_liger_kernel  # 导入对应的函数
    elif model_type == "llama":  # 如果模型类型为 llama
        from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_liger_kernel  # 导入对应的函数
    elif model_type == "mistral":  # 如果模型类型为 mistral
        from liger_kernel.transformers import apply_liger_kernel_to_mistral as apply_liger_kernel  # 导入对应的函数
    elif model_type == "mixtral":  # 如果模型类型为 mixtral
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral as apply_liger_kernel  # 导入对应的函数
    elif model_type == "phi3":  # 如果模型类型为 phi3
        from liger_kernel.transformers import apply_liger_kernel_to_phi3 as apply_liger_kernel  # 导入对应的函数
    elif model_type == "qwen2":  # 如果模型类型为 qwen2
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as apply_liger_kernel  # 导入对应的函数
    elif model_type == "qwen2_vl":  # 如果模型类型为 qwen2_vl
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl as apply_liger_kernel  # 导入对应的函数
    else:  # 如果模型类型不支持
        logger.warning_rank0("Current model does not support liger kernel.")  # 记录警告：当前模型不支持 liger kernel。
        return  # 直接返回

    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_liger_kernel).parameters:
        logger.info_rank0("Current training stage does not support chunked cross entropy.")  # 记录信息：当前训练阶段不支持分块交叉熵。
        kwargs = {"fused_linear_cross_entropy": False}  # 设置参数
    else:
        kwargs = {}  # 其他情况下，参数为空字典

    apply_liger_kernel(**kwargs)  # 应用 liger kernel
    logger.info_rank0("Liger kernel has been applied to the model.")  # 记录信息：已将 liger kernel 应用到模型。