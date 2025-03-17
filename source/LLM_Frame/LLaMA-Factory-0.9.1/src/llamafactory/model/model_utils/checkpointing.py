# Copyright 2024 HuggingFace Inc., Daniel Han-Chen & the Unsloth team and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and PEFT library,
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py
# and the Unsloth library.
# https://github.com/unslothai/unsloth/blob/July-2024/unsloth/models/_utils.py
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

import inspect
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch

from ...extras import logging
from ...extras.constants import LAYERNORM_NAMES


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)  # 获取日志记录器


def get_unsloth_gradient_checkpointing_func() -> Callable:
    class UnslothGradientCheckpointing(torch.autograd.Function):
        r"""
        Saves VRAM by smartly offloading to RAM.
        """  # 通过智能地卸载到 RAM 来节省 VRAM

        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx: "torch.autograd.Function",
            forward_function: "torch.Module",
            hidden_states: "torch.Tensor",
            *args: Union["torch.Tensor", Any],
        ) -> "torch.Tensor":
            # 将隐藏状态转移到 CPU，非阻塞
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            with torch.no_grad():
                output = forward_function(hidden_states, *args)  # 执行前向传播

            ctx.save_for_backward(saved_hidden_states)  # 保存隐藏状态
            ctx.forward_function = forward_function  # 保存前向函数
            ctx.args = args  # 保存其他参数
            return output  # 返回输出

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx: "torch.autograd.Function", grad_output: "torch.Tensor") -> "torch.Tensor":
            (hidden_states,) = ctx.saved_tensors  # 获取保存的隐藏状态
            hidden_states = hidden_states.to("cuda", non_blocking=True).detach()  # 转移到 CUDA 并分离
            hidden_states.requires_grad_(True)  # 需要梯度
            with torch.enable_grad():
                (output,) = ctx.forward_function(hidden_states, *ctx.args)  # 执行前向传播以计算输出

            torch.autograd.backward(output, grad_output)  # 反向传播
            return (None, hidden_states.grad) + (None,) * len(ctx.args)  # 返回梯度

    return UnslothGradientCheckpointing.apply  # 返回自定义的梯度检查点函数


def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    r"""
    Only applies gradient checkpointing to trainable layers.
    """  # 仅对可训练层应用梯度检查点

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        module: "torch.nn.Module" = func.__self__  # 获取模块实例

        if any(param.requires_grad for param in module.parameters()):  # 检查是否有参数需要梯度
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):  # 如果参数是浮点张量
                    arg.requires_grad_(True)  # 设置需要梯度

        return gradient_checkpointing_func(func, *args, **kwargs)  # 调用原始梯度检查点函数

    return custom_gradient_checkpointing_func  # 返回自定义的梯度检查点函数


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    use_unsloth_gc: bool = False,
) -> None:
    r"""
    Activates gradient checkpointing for the current model.
    # 激活当前模型的梯度检查点
    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    """  # 修改原始方法以启用块级优化器的梯度检查点
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:  # 检查模型是否支持梯度检查点
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")  # 抛出错误

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}  # 默认参数

    if use_unsloth_gc:  # 如果使用 unsloth 梯度检查点
        gradient_checkpointing_func = get_unsloth_gradient_checkpointing_func()  # 获取 unsloth 函数
    else:
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)  # 使用部分应用的检查点函数

    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)  # 获取自定义函数
    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # 旧的 GC 格式
        self.apply(partial(self._set_gradient_checkpointing, value=True))  # 应用设置
        self.enable_input_require_grads()  # 启用输入需要梯度
        logger.warning_once("You are using the old GC format, some features (e.g. BAdam) will be invalid.")  # 记录警告
    else:  # 已经启用输入需要梯度
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)  # 启用梯度检查点


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)  # 转换输出为 float32


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""
    Includes:
        (1) cast the layernorm in fp32  # 包括：将 layernorm 转换为 float32
        (2) make output embedding layer require grads  # 使输出嵌入层需要梯度
        (3) add the upcasting of the lm_head in fp32  # 添加将 lm_head 转换为 float32 的功能
    """
    # 如果 model_args 中的 upcast_layernorm 为 True
    if model_args.upcast_layernorm:
        logger.info_rank0("Upcasting layernorm weights in float32.")  # 记录信息：将 layernorm 权重转换为 float32。
        for name, param in model.named_parameters():  # 遍历模型的所有参数
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)  # 将符合条件的参数转换为 float32

    # 如果没有禁用梯度检查点
    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning_rank0("Current model does not support gradient checkpointing.")  # 记录警告：当前模型不支持梯度检查点。
        else:
            # use_reentrant=False 可能会增加 VRAM 使用量（尚未经过实证验证）
            # 根据：https://github.com/huggingface/transformers/issues/28339
            gradient_checkpointing_enable = partial(
                _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
            )  # 部分应用 _gradient_checkpointing_enable 函数
            model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)  # 将方法绑定到模型
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})  # 启用梯度检查点
            setattr(model.config, "use_cache", False)  # 在启用梯度检查点时关闭缓存
            logger.info_rank0("Gradient checkpointing enabled.")  # 记录信息：梯度检查点已启用。

    # 如果 model_args 中的 upcast_lmhead_output 为 True
    if model_args.upcast_lmhead_output:
        output_layer = model.get_output_embeddings()  # 获取模型的输出嵌入层
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            logger.info_rank0("Upcasting lm_head outputs in float32.")  # 记录信息：将 lm_head 输出转换为 float32。
            output_layer.register_forward_hook(_fp32_forward_post_hook)  # 注册前向钩子以处理输出
