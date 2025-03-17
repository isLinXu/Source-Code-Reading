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

from typing import TYPE_CHECKING, List  # 导入类型检查、列表类型

from ...extras import logging  # 从 extras 模块导入 logging


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer  # 导入预训练配置、模型和分词器的类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.  # 查找所有可用于应用 lora 或 galore 的模块
    """
    model_type = getattr(model.config, "model_type", None)  # 获取模型类型
    forbidden_modules = {"lm_head"}  # 定义禁止的模块集合
    if model_type == "chatglm":  # 如果模型类型为 chatglm
        forbidden_modules.add("output_layer")  # 添加禁止的输出层
    elif model_type == "internlm2":  # 如果模型类型为 internlm2
        forbidden_modules.add("output")  # 添加禁止的输出模块
    elif model_type in ["llava", "llava_next", "llava_next_video", "mllama", "paligemma", "video_llava"]:  # 如果模型类型在指定列表中
        forbidden_modules.add("multi_modal_projector")  # 添加禁止的多模态投影器
    elif model_type == "qwen2_vl":  # 如果模型类型为 qwen2_vl
        forbidden_modules.add("merger")  # 添加禁止的合并模块

    if freeze_vision_tower:  # 如果冻结视觉塔
        if model_type == "mllama":  # 如果模型类型为 mllama
            forbidden_modules.add("vision_model")  # 添加禁止的视觉模型
        elif model_type == "qwen2_vl":  # 如果模型类型为 qwen2_vl
            forbidden_modules.add("visual")  # 添加禁止的视觉模块
        else:
            forbidden_modules.add("vision_tower")  # 添加禁止的视觉塔

    module_names = set()  # 创建一个集合以存储模块名称
    for name, module in model.named_modules():  # 遍历模型的所有命名模块
        if any(forbidden_module in name for forbidden_module in forbidden_modules):  # 如果模块名称包含禁止的模块
            continue  # 跳过该模块

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:  # 如果模块是线性层且不是嵌入层
            module_names.add(name.split(".")[-1])  # 添加模块名称的最后一部分

    logger.info_rank0("Found linear modules: {}".format(",".join(module_names)))  # 记录找到的线性模块
    return list(module_names)  # 返回线性模块名称的列表


def find_expanded_modules(model: "PreTrainedModel", target_modules: List[str], num_layer_trainable: int) -> List[str]:
    r"""
    Finds the modules in the expanded blocks to apply lora.  # 查找扩展块中的模块以应用 lora
    """
    num_layers = getattr(model.config, "num_hidden_layers", None)  # 获取模型的隐藏层数量
    if not num_layers:  # 如果没有隐藏层数量
        raise ValueError("Model was not supported.")  # 抛出错误：模型不受支持

    if num_layers % num_layer_trainable != 0:  # 如果隐藏层数量不能被可训练层数量整除
        raise ValueError(
            f"`num_layers` {num_layers} should be divisible by `num_layer_trainable` {num_layer_trainable}."  # 抛出错误：隐藏层数量应能被可训练层数量整除
        )

    stride = num_layers // num_layer_trainable  # 计算步幅
    trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)  # 计算可训练层的 ID
    trainable_layers = [f".{idx:d}." for idx in trainable_layer_ids]  # 创建可训练层的名称列表
    module_names = []  # 创建一个列表以存储模块名称
    for name, _ in model.named_modules():  # 遍历模型的所有命名模块
        if any(target_module in name for target_module in target_modules) and any(  # 如果模块名称包含目标模块并且包含可训练层
            trainable_layer in name for trainable_layer in trainable_layers
        ):
            module_names.append(name)  # 添加模块名称

    logger.info_rank0("Apply lora to layers: {}".format(",".join(map(str, trainable_layer_ids))))  # 记录应用 lora 的层
    return module_names  # 返回模块名称


def register_autoclass(config: "PretrainedConfig", model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    if "AutoConfig" in getattr(config, "auto_map", {}):  # 如果配置的 auto_map 中包含 AutoConfig
        config.__class__.register_for_auto_class()  # 注册自动配置类
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):  # 如果配置的 auto_map 中包含 AutoModelForCausalLM
        model.__class__.register_for_auto_class()  # 注册自动模型类
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):  # 如果分词器的 auto_map 中包含 AutoTokenizer
        tokenizer.__class__.register_for_auto_class()  # 注册自动分词器类
