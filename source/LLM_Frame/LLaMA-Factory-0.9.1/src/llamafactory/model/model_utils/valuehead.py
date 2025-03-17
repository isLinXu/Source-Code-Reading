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

from typing import TYPE_CHECKING, Dict  # 导入类型检查和字典类型

import torch  # 导入 PyTorch 库
from transformers.utils import cached_file  # 从 transformers 导入缓存文件的函数

from ...extras import logging  # 从 extras 导入日志模块
from ...extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME  # 从 extras.constants 导入值头安全权重和权重名称


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PreTrainedModel  # 导入预训练模型的类型

    from ...hparams import ModelArguments  # 导入模型参数的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def load_valuehead_params(path_or_repo_id: str, model_args: "ModelArguments") -> Dict[str, torch.Tensor]:
    r"""
    Loads value head parameters from Hugging Face Hub or local disk.  # 从 Hugging Face Hub 或本地磁盘加载值头参数

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.  # 返回包含 `v_head.summary.weight` 和 `v_head.summary.bias` 的字典
    """
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}  # 设置加载参数
    err_text = ""  # 初始化错误文本

    try:
        from safetensors import safe_open  # 从 safetensors 导入安全打开的函数

        vhead_file = cached_file(filename=V_HEAD_SAFE_WEIGHTS_NAME, **kwargs)  # 获取安全权重文件
        with safe_open(vhead_file, framework="pt", device="cpu") as f:  # 安全打开权重文件
            return {key: f.get_tensor(key) for key in f.keys()}  # 返回权重张量
    except Exception as err:  # 捕获异常
        err_text = str(err)  # 将错误信息转换为字符串

    try:
        vhead_file = cached_file(filename=V_HEAD_WEIGHTS_NAME, **kwargs)  # 获取普通权重文件
        return torch.load(vhead_file, map_location="cpu")  # 返回加载的权重
    except Exception as err:  # 捕获异常
        err_text = str(err)  # 将错误信息转换为字符串

    logger.info_rank0(f"Provided path ({path_or_repo_id}) does not contain value head weights: {err_text}.")  # 记录未找到权重的信息
    logger.info_rank0("Ignore the above message if you are not resuming the training of a value head model.")  # 如果不需要恢复训练，可以忽略上述消息
    return None  # 返回 None


def prepare_valuehead_model(model: "PreTrainedModel") -> None:
    if getattr(model.config, "model_type", None) == "llava":  # 如果模型类型为 llava
        setattr(model, "lm_head", model.language_model.get_output_embeddings())  # 设置语言模型的输出嵌入
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])  # 设置保存时忽略的键

    if getattr(model.config, "model_type", None) == "chatglm":  # 如果模型类型为 chatglm
        setattr(model, "lm_head", model.transformer.output_layer)  # 设置变换器的输出层
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])  # 设置保存时忽略的键

    if getattr(model.config, "model_type", None) == "internlm2":  # 如果模型类型为 internlm2
        setattr(model, "lm_head", model.output)  # 设置模型的输出
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])  # 设置保存时忽略的键