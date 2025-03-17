# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/modeling_llava.py
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

from typing import TYPE_CHECKING, List, Sequence, Set, Tuple, Union  # 导入类型检查、列表、序列、集合、元组和联合类型

import torch  # 导入 PyTorch 库
import transformers  # 导入 transformers 库
import transformers.models  # 导入 transformers 的模型模块
from transformers.activations import ACT2FN  # 从 transformers 导入激活函数映射

from ...extras import logging  # 从 extras 导入日志模块


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import LlavaConfig, PretrainedConfig, PreTrainedModel, ProcessorMixin  # 导入相关的 transformers 类型

    from ...hparams import FinetuningArguments, ModelArguments  # 导入微调参数和模型参数的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
transformers_logger = transformers.utils.logging.get_logger(__name__)  # 获取 transformers 的日志记录器


class LlavaMultiModalProjectorForYiVL(torch.nn.Module):  # 定义 Llava 多模态投影器类
    def __init__(self, config: "LlavaConfig") -> None:  # 初始化方法
        super().__init__()  # 调用父类初始化方法

        self.config = config  # 保存配置
        if config is None:  # 如果配置为 None
            return  # 直接返回

        self.linear_1 = torch.nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)  # 定义线性层
        self.linear_2 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)  # 定义层归一化
        self.linear_3 = torch.nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)  # 定义线性层
        self.linear_4 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)  # 定义层归一化
        self.act = ACT2FN[config.projector_hidden_act]  # 获取激活函数

    def forward(self, image_features: "torch.Tensor") -> "torch.Tensor":  # 前向传播方法
        hidden_states = self.linear_1(image_features)  # 通过线性层处理图像特征
        hidden_states = self.linear_2(hidden_states)  # 进行层归一化
        hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states = self.linear_3(hidden_states)  # 通过另一个线性层处理
        hidden_states = self.linear_4(hidden_states)  # 再次进行层归一化
        if hidden_states.dtype == torch.float32:  # 如果数据类型为 float32
            if torch.is_autocast_enabled():  # 如果启用了自动混合精度
                target_dtype = torch.get_autocast_gpu_dtype()  # 获取目标数据类型
            elif hasattr(self.config, "_pre_quantization_dtype"):  # 如果配置中有前量化数据类型
                target_dtype = self.config._pre_quantization_dtype  # 获取前量化数据类型
            else:  # 否则
                target_dtype = self.linear_1.weight.dtype  # 获取线性层权重的数据类型

            transformers_logger.warning_once("The hidden states seems to be silently casted in float32.")  # 记录警告信息
            hidden_states = hidden_states.to(target_dtype)  # 转换数据类型

        return hidden_states  # 返回处理后的隐藏状态


class LlavaMultiModalProjectorForYiVLForVLLM(LlavaMultiModalProjectorForYiVL):  # 定义 VLLM 的多模态投影器类
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, projector_hidden_act: str) -> None:  # 初始化方法
        super().__init__(config=None)  # 调用父类初始化方法

        self.linear_1 = torch.nn.Linear(vision_hidden_size, text_hidden_size, bias=True)  # 定义线性层
        self.linear_2 = torch.nn.LayerNorm(text_hidden_size, bias=True)  # 定义层归一化
        self.linear_3 = torch.nn.Linear(text_hidden_size, text_hidden_size, bias=True)  # 定义线性层
        self.linear_4 = torch.nn.LayerNorm(text_hidden_size, bias=True)  # 定义层归一化
        self.act = ACT2FN[projector_hidden_act]  # 获取激活函数


def autocast_projector_dtype(model: "PreTrainedModel", model_args: "ModelArguments") -> None:  # 自动转换投影器数据类型
    r"""
    Casts projector output to half precision for fine-tuning quantized VLMs.  # 将投影器输出转换为半精度，以便微调量化的 VLM。
    """

    def _mm_projector_forward_post_hook(  # 定义前向钩子
        module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)  # 返回转换后的输出

    if getattr(model, "quantization_method", None):  # 如果模型有量化方法
        model_type = getattr(model.config, "model_type", None)  # 获取模型类型
        if model_type in ["llava", "llava_next", "llava_next_video", "paligemma", "pixtral", "video_llava"]:  # 如果模型类型在指定列表中
            mm_projector: "torch.nn.Module" = getattr(model, "multi_modal_projector")  # 获取多模态投影器
        elif model_type == "qwen2_vl":  # 如果模型类型为 qwen2_vl
            mm_projector: "torch.nn.Module" = getattr(getattr(model, "visual"), "merger")  # 获取视觉合并模块
        else:
            return  # 如果不匹配，直接返回

        logger.info_rank0(f"Casting multimodal projector outputs in {model_args.compute_dtype}.")  # 记录投影器输出的数据类型
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)  # 注册前向钩子


def configure_visual_model(config: "PretrainedConfig") -> None:  # 配置视觉模型
    r"""
    Patches VLMs before loading them.  # 加载视觉语言模型之前进行补丁处理。
    """
    model_type = getattr(config, "model_type", None)  # 获取模型类型
    if model_type in [  # 如果模型类型在指定列表中
        "llava",
        "llava_next",
        "llava_next_video",
        "paligemma",
        "pixtral",
        "video_llava",
    ]:  # required for ds zero3 and valuehead models
        setattr(config, "hidden_size", getattr(config.text_config, "hidden_size", None))  # 更新配置中的隐藏大小

    if getattr(config, "is_yi_vl_derived_model", None):  # 如果模型是 Yi-VL 派生模型
        logger.info_rank0("Detected Yi-VL model, applying projector patch.")  # 记录检测到 Yi-VL 模型的信息
        transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVL  # 应用投影器补丁


def get_forbidden_modules(config: "PretrainedConfig", finetuning_args: "FinetuningArguments") -> Set[str]:  # 获取禁止的模块
    r"""
    Freezes vision tower and language model for VLM full/freeze tuning.  # 冻结视觉塔和语言模型以进行 VLM 完全/冻结微调。
    """
    model_type = getattr(config, "model_type", None)  # 获取模型类型
    forbidden_modules = set()  # 初始化禁止模块集合
    if model_type in ["llava", "llava_next", "llava_next_video", "paligemma", "pixtral", "video_llava"]:  # 如果模型类型在指定列表中
        if finetuning_args.freeze_vision_tower:  # 如果冻结视觉塔
            forbidden_modules.add("vision_tower")  # 添加视觉塔到禁止模块

        if finetuning_args.train_mm_proj_only:  # 如果只训练多模态投影器
            forbidden_modules.add("language_model")  # 添加语言模型到禁止模块

    elif model_type == "qwen2_vl":  # 如果模型类型为 qwen2_vl
        if finetuning_args.freeze_vision_tower:  # 如果冻结视觉塔
            forbidden_modules.add("visual")  # 添加视觉到禁止模块

        if finetuning_args.train_mm_proj_only:  # 如果只训练多模态投影器
            raise ValueError("Qwen2-VL models do not support `train_mm_proj_only`.")  # 抛出错误

    return forbidden_modules  # 返回禁止模块集合


def get_image_seqlen(config: "PretrainedConfig") -> int:  # 获取图像序列长度
    r"""
    Computes the number of special tokens per image.  # 计算每个图像的特殊令牌数量。
    """
    model_type = getattr(config, "model_type", None)  # 获取模型类型
    if model_type == "llava":  # 如果模型类型为 llava
        image_seqlen = (config.vision_config.image_size // config.vision_config.patch_size) ** 2  # 计算图像序列长度
        if getattr(config, "vision_feature_select_strategy", "default") == "full":  # 如果选择的视觉特征选择策略为 full
            image_seqlen += 1  # 添加 [CLS] 令牌
    elif model_type == "paligemma":  # 如果模型类型为 paligemma
        image_seqlen = config.vision_config.num_image_tokens  # 获取图像令牌数量
    else:
        image_seqlen = -1  # 设置为 -1 表示不支持

    return image_seqlen  # 返回图像序列长度


def get_patch_size(config: "PretrainedConfig", processor: "ProcessorMixin") -> int:  # 获取补丁大小
    r"""
    Computes the patch size of the vit.  # 计算视觉变换器的补丁大小。
    """
    patch_size = getattr(config.vision_config, "patch_size", getattr(processor, "patch_size", -1))  # 获取补丁大小
    return patch_size  # 返回补丁大小


def get_vision_feature_select_strategy(config: "PretrainedConfig", processor: "ProcessorMixin") -> int:  # 获取视觉特征选择策略
    r"""
    Get the vision_feature_select_strategy.  # 获取视觉特征选择策略。
    """
    vision_feature_select_strategy = getattr(  # 获取视觉特征选择策略
        config, "vision_feature_select_strategy", getattr(processor, "vision_feature_select_strategy", "default")
    )
    return vision_feature_select_strategy  # 返回视觉特征选择策略


def patch_target_modules(  # 补丁目标模块
    config: "PretrainedConfig", finetuning_args: "FinetuningArguments", target_modules: Sequence[str]
) -> Union[str, List[str]]:
    r"""
    Freezes vision tower for VLM LoRA tuning.  # 冻结视觉塔以进行 VLM LoRA 微调。
    """
    model_type = getattr(config, "model_type", None)  # 获取模型类型
    if finetuning_args.freeze_vision_tower:  # 如果冻结视觉塔
        if model_type in ["llava", "llava_next", "llava_next_video", "paligemma", "pixtral", "video_llava"]:  # 如果模型类型在指定列表中
            return "^(?!.*vision_tower).*(?:{}).*".format("|".join(target_modules))  # 返回正则表达式
        elif model_type == "mllama":  # 如果模型类型为 mllama
            return "^(?!.*vision_model).*(?:{}).*".format("|".join(target_modules))  # 返回正则表达式
        elif model_type == "qwen2_vl":  # 如果模型类型为 qwen2_vl
            return "^(?!.*visual).*(?:{}).*".format("|".join(target_modules))  # 返回正则表达式
        else:
            return target_modules  # 返回目标模块

    else:  # 如果不冻结视觉塔
        if model_type == "qwen2_vl":  # 如果模型类型为 qwen2_vl
            return "^(?!.*patch_embed).*(?:{}).*".format("|".join(target_modules))  # 返回正则表达式
        elif model_type == "pixtral":  # 如果模型类型为 pixtral
            return "^(?!.*patch_conv).*(?:{}).*".format("|".join(target_modules))  # 返回正则表达式
        else:
            return target_modules  # 返回目标模块