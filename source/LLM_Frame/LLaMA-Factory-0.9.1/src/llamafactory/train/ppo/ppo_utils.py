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

import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

# 检查并导入requests库（用于HTTP请求）
from ...extras.packages import is_requests_available  # Check requests availability


if is_requests_available():
    import requests  # 仅当requests可用时导入


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead  # 类型提示专用导入


def get_rewards_from_server(server_url: str, messages: List[str]) -> List["torch.Tensor"]:
    r"""
    Gets reward scores from the API server.
    从API服务器获取奖励分数
    """
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)  # 发送POST请求
    rewards = json.loads(response.text)["scores"]  # 解析响应中的分数
    return torch.Tensor(rewards)  # 转换为PyTorch张量


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    r"""
    Replaces the default/reward modules in the model. The model is already unwrapped.
    替换模型中的默认/奖励模块（模型已解封装）
    """
    v_head_layer = model.v_head.summary  # 获取价值头层
    if is_deepspeed_zero3_enabled():  # 处理DeepSpeed Zero3模式
        import deepspeed  # type: ignore // 类型检查忽略
        
        # 收集分布在多个GPU上的参数
        params = [v_head_layer.weight, v_head_layer.bias]
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()  # 普通模式使用空上下文

    model.pretrained_model.set_adapter(target)  # 设置活动适配器 // 切换LoRA适配器
    with context_maybe_zero3:
        if target == "reward":  # 保存默认头参数
            # 暂存默认头权重和偏置
            setattr(model, "default_head_weight", v_head_layer.weight.data.detach().clone())
            setattr(model, "default_head_bias", v_head_layer.bias.data.detach().clone())

        device = v_head_layer.weight.device
        # 加载目标头参数（从模型缓冲区）
        v_head_layer.weight.data = model.get_buffer(f"{target}_head_weight").detach().clone().to(device)
        v_head_layer.bias.data = model.get_buffer(f"{target}_head_bias").detach().clone().to(device)


def dump_layernorm(model: "PreTrainedModel") -> Dict[str, "torch.Tensor"]:
    r"""
    Dumps the layernorm parameters in the model. The model is already unwrapped (and gathered).
    转储模型中的LayerNorm参数（模型已解封装且参数已收集）
    """
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:  # 仅处理float32类型的参数
            layer_norm_params[name] = param.data.detach().clone()  # 保存副本
            param.data = param.data.to(model.config.torch_dtype)  # 转换回模型原始精度
    return layer_norm_params  # 返回参数字典


def restore_layernorm(model: "PreTrainedModel", layernorm_params: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
    r"""
    Restores the layernorm parameters in the model. The model is already unwrapped (and gathered).
    恢复模型中的LayerNorm参数（模型已解封装且参数已收集）
    """
    for name, param in model.named_parameters():
        if name in layernorm_params:  # 仅恢复有记录的参数
            param.data = layernorm_params[name]  # 从字典恢复数据
