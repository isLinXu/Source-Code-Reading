# Copyright 2024 the LlamaFactory team.
# 版权所有 2024 LlamaFactory 团队。

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守该许可证，否则您不得使用此文件。
# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”提供的，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证以了解有关权限和限制的具体语言。

import re  # 导入正则表达式模块
from typing import TYPE_CHECKING  # 导入类型检查模块

import torch  # 导入 PyTorch 库
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model  # 从 peft 模块导入相关类和函数
from transformers.integrations import is_deepspeed_zero3_enabled  # 导入检查 DeepSpeed ZeRO-3 是否启用的函数
from transformers.modeling_utils import is_fsdp_enabled  # 导入检查 FSDP 是否启用的函数

from ..extras import logging  # 从 extras 模块导入日志记录
from .model_utils.misc import find_all_linear_modules, find_expanded_modules  # 导入查找线性模块和扩展模块的函数
from .model_utils.quantization import QuantizationMethod  # 导入量化方法
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model  # 导入 unsloth 模块的函数
from .model_utils.visual import get_forbidden_modules, patch_target_modules  # 导入获取禁止模块和修补目标模块的函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig, PreTrainedModel  # 导入预训练配置和模型

    from ..hparams import FinetuningArguments, ModelArguments  # 从 hparams 模块导入微调参数和模型参数


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def _setup_full_tuning(  # 定义设置完整微调的函数
    model: "PreTrainedModel",  # 模型参数，类型为预训练模型
    finetuning_args: "FinetuningArguments",  # 微调参数
    is_trainable: bool,  # 是否可训练的布尔值
    cast_trainable_params_to_fp32: bool,  # 是否将可训练参数转换为 FP32 的布尔值
) -> None:  # 返回类型为 None
    if not is_trainable:  # 如果不可训练
        return  # 直接返回

    logger.info_rank0("Fine-tuning method: Full")  # 记录微调方法为完整微调
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)  # 获取禁止的模块
    for name, param in model.named_parameters():  # 遍历模型的命名参数
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):  # 如果参数名不在禁止模块中
            if cast_trainable_params_to_fp32:  # 如果需要将可训练参数转换为 FP32
                param.data = param.data.to(torch.float32)  # 将参数数据转换为 FP32
        else:  # 如果参数名在禁止模块中
            param.requires_grad_(False)  # 不需要计算梯度


def _setup_freeze_tuning(  # 定义设置冻结微调的函数
    model: "PreTrainedModel",  # 模型参数，类型为预训练模型
    finetuning_args: "FinetuningArguments",  # 微调参数
    is_trainable: bool,  # 是否可训练的布尔值
    cast_trainable_params_to_fp32: bool,  # 是否将可训练参数转换为 FP32 的布尔值
) -> None:  # 返回类型为 None
    if not is_trainable:  # 如果不可训练
        return  # 直接返回

    logger.info_rank0("Fine-tuning method: Freeze")  # 记录微调方法为冻结微调
    if hasattr(model.config, "text_config"):  # 如果模型配置有文本配置属性
        config = getattr(model.config, "text_config")  # 获取文本配置
    else:  # 否则
        config = model.config  # 使用模型配置

    num_layers = (  # 获取模型的层数
        getattr(config, "num_hidden_layers", None)  # 从配置中获取隐藏层数
        or getattr(config, "num_layers", None)  # 或获取层数
        or getattr(config, "n_layer", None)  # 或获取 n_layer
    )
    if not num_layers:  # 如果没有层数
        raise ValueError("Current model does not support freeze tuning.")  # 抛出错误，当前模型不支持冻结微调

    if finetuning_args.use_llama_pro:  # 如果使用 Llama Pro
        if num_layers % finetuning_args.freeze_trainable_layers != 0:  # 如果层数不能被可训练层数整除
            raise ValueError(
                "`num_layers` {} should be divisible by `num_layer_trainable` {}.".format(
                    num_layers, finetuning_args.freeze_trainable_layers  # 抛出错误，层数应能被可训练层数整除
                )
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers  # 计算步幅
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)  # 获取可训练层的索引
    elif finetuning_args.freeze_trainable_layers > 0:  # 如果可训练层数大于 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)  # 获取最后 n 层的可训练层索引
    else:  # 如果可训练层数小于 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))  # 获取前 n 层的可训练层索引

    hidden_modules = set()  # 定义隐藏模块集合
    non_hidden_modules = set()  # 定义非隐藏模块集合
    for name, _ in model.named_parameters():  # 遍历模型的命名参数
        if ".0." in name:  # 如果参数名包含 ".0."
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])  # 添加到隐藏模块集合
        elif ".1." in name:  # 如果参数名包含 ".1."
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])  # 添加到隐藏模块集合

        if re.search(r"\.\d+\.", name) is None:  # 如果参数名中不包含数字
            non_hidden_modules.add(name.split(".")[-2])  # 添加到非隐藏模块集合

    trainable_layers = []  # 定义可训练层列表
    for module_name in finetuning_args.freeze_trainable_modules:  # 遍历冻结可训练模块
        if module_name != "all" and module_name not in hidden_modules:  # 如果模块名不是 "all" 且不在隐藏模块中
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))  # 抛出错误，模块未找到
            )

        for idx in trainable_layer_ids:  # 遍历可训练层索引
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))  # 添加到可训练层列表

    if finetuning_args.freeze_extra_modules:  # 如果有冻结额外模块
        for module_name in finetuning_args.freeze_extra_modules:  # 遍历冻结的额外模块
            if module_name not in non_hidden_modules:  # 如果模块不在非隐藏模块中
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))  # 抛出错误，模块未找到
                )

            trainable_layers.append(module_name)  # 添加到可训练层列表

    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)  # 获取禁止的模块
    for name, param in model.named_parameters():  # 遍历模型的命名参数
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(  # 如果参数名在可训练层中且不在禁止模块中
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:  # 如果需要将可训练参数转换为 FP32
                param.data = param.data.to(torch.float32)  # 将参数数据转换为 FP32
        else:  # 如果参数名在禁止模块中
            param.requires_grad_(False)  # 不需要计算梯度

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))  # 记录已设置的可训练层


def _setup_lora_tuning(  # 定义设置 LoRA 微调的函数
    config: "PretrainedConfig",  # 配置参数，类型为预训练配置
    model: "PreTrainedModel",  # 模型参数，类型为预训练模型
    model_args: "ModelArguments",  # 模型参数
    finetuning_args: "FinetuningArguments",  # 微调参数
    is_trainable: bool,  # 是否可训练的布尔值
    cast_trainable_params_to_fp32: bool,  # 是否将可训练参数转换为 FP32 的布尔值
) -> "PeftModel":  # 返回类型为 PeftModel
    if is_trainable:  # 如果可训练
        logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))  # 记录微调方法为 DoRA 或 LoRA

    adapter_to_resume = None  # 定义要恢复的适配器为 None

    if model_args.adapter_name_or_path is not None:  # 如果适配器名称或路径不为空
        is_mergeable = True  # 设置可合并标志为 True
        if getattr(model, "quantization_method", None):  # 如果模型有量化方法
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."  # 断言适配器数量为 1
            is_mergeable = False  # 设置可合并标志为 False

        if is_deepspeed_zero3_enabled():  # 如果启用了 DeepSpeed ZeRO-3
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."  # 断言适配器数量为 1
            is_mergeable = False  # 设置可合并标志为 False

        if model_args.use_unsloth:  # 如果使用 unsloth
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."  # 断言适配器数量为 1
            is_mergeable = False  # 设置可合并标志为 False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):  # 如果可训练且不创建新适配器，或不可合并
            adapter_to_merge = model_args.adapter_name_or_path[:-1]  # 获取要合并的适配器
            adapter_to_resume = model_args.adapter_name_or_path[-1]  # 获取要恢复的适配器
        else:  # 否则
            adapter_to_merge = model_args.adapter_name_or_path  # 直接使用适配器

        init_kwargs = {  # 初始化关键字参数
            "subfolder": model_args.adapter_folder,  # 子文件夹
            "offload_folder": model_args.offload_folder,  # 卸载文件夹
            "cache_dir": model_args.cache_dir,  # 缓存目录
            "revision": model_args.model_revision,  # 版本
            "token": model_args.hf_hub_token,  # 令牌
        }

        for adapter in adapter_to_merge:  # 遍历要合并的适配器
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)  # 从预训练模型中加载适配器
            model = model.merge_and_unload()  # 合并并卸载适配器

        if len(adapter_to_merge) > 0:  # 如果有适配器被合并
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")  # 记录合并的适配器数量

        if adapter_to_resume is not None:  # 如果要恢复的适配器不为空
            if model_args.use_unsloth:  # 如果使用 unsloth
                model = load_unsloth_peft_model(config, model_args, is_trainable=is_trainable)  # 加载 unsloth 模型
            else:  # 否则
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)  # 从预训练模型中加载适配器

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))  # 记录加载的适配器

    if is_trainable and adapter_to_resume is None:  # 如果可训练且没有要恢复的适配器
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":  # 如果 LoRA 目标为 "all"
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)  # 查找所有线性模块
        else:  # 否则
            target_modules = finetuning_args.lora_target  # 使用 LoRA 目标

        if finetuning_args.use_llama_pro:  # 如果使用 Llama Pro
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)  # 查找扩展模块

        target_modules = patch_target_modules(model.config, finetuning_args, target_modules)  # 修补目标模块

        if (  # 如果使用 DoRA 且模型的量化方法不为 BITS_AND_BYTES
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BITS_AND_BYTES
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")  # 抛出错误，DoRA 与 PTQ 量化模型不兼容

        if model_args.resize_vocab and finetuning_args.additional_target is None:  # 如果调整词汇表大小且没有额外目标
            input_embeddings = model.get_input_embeddings()  # 获取输入嵌入
            output_embeddings = model.get_output_embeddings()  # 获取输出嵌入
            module_names = set()  # 定义模块名称集合
            for name, module in model.named_modules():  # 遍历模型的命名模块
                if module in [input_embeddings, output_embeddings]:  # 如果模块是输入或输出嵌入
                    module_names.add(name.split(".")[-1])  # 添加模块名称

            finetuning_args.additional_target = module_names  # 设置额外目标
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))  # 记录词汇表已调整

        peft_kwargs = {  # 定义 LoRA 关键字参数
            "r": finetuning_args.lora_rank,  # LoRA 的秩
            "target_modules": target_modules,  # 目标模块
            "lora_alpha": finetuning_args.lora_alpha,  # LoRA 的 alpha
            "lora_dropout": finetuning_args.lora_dropout,  # LoRA 的 dropout
            "use_rslora": finetuning_args.use_rslora,  # 是否使用 RSLORA
            "use_dora": finetuning_args.use_dora,  # 是否使用 DoRA
            "modules_to_save": finetuning_args.additional_target,  # 要保存的模块
        }

        if model_args.use_unsloth:  # 如果使用 unsloth
            model = get_unsloth_peft_model(model, model_args, peft_kwargs)  # 获取 unsloth 模型
        else:  # 否则
            if finetuning_args.pissa_init:  # 如果使用 PiSSA 初始化
                if finetuning_args.pissa_iter == -1:  # 如果迭代次数为 -1
                    logger.info_rank0("Using PiSSA initialization.")  # 记录使用 PiSSA 初始化
                    peft_kwargs["init_lora_weights"] = "pissa"  # 设置初始化权重为 PiSSA
                else:  # 否则
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")  # 记录使用 PiSSA 初始化及其迭代次数
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"  # 设置初始化权重为带迭代次数的 PiSSA

            lora_config = LoraConfig(  # 创建 LoRA 配置
                task_type=TaskType.CAUSAL_LM,  # 任务类型为因果语言模型
                inference_mode=False,  # 推理模式为 False
                **peft_kwargs,  # 解包 LoRA 关键字参数
            )
            model = get_peft_model(model, lora_config)  # 获取 LoRA 模型

    if is_trainable and cast_trainable_params_to_fp32:  # 如果可训练且需要将参数转换为 FP32
        for param in filter(lambda p: p.requires_grad, model.parameters()):  # 遍历需要计算梯度的参数
            param.data = param.data.to(torch.float32)  # 将参数数据转换为 FP32

    return model  # 返回模型


def init_adapter(  # 定义初始化适配器的函数
    config: "PretrainedConfig",  # 配置参数，类型为预训练配置
    model: "PreTrainedModel",  # 模型参数，类型为预训练模型
    model_args: "ModelArguments",  # 模型参数
    finetuning_args: "FinetuningArguments",  # 微调参数
    is_trainable: bool,  # 是否可训练的布尔值
) -> "PreTrainedModel":  # 返回类型为预训练模型
    r"""
    Initializes the adapters.  # 初始化适配器。
    
    Support full-parameter, freeze and LoRA training.  # 支持全参数、冻结和 LoRA 训练。

    Note that the trainable parameters must be cast to float32.  # 请注意，可训练参数必须转换为 float32。
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:  # 如果可训练且模型有量化方法
        if finetuning_args.finetuning_type != "lora":  # 如果微调类型不是 LoRA
            raise ValueError("Quantized models can only be used for the LoRA tuning.")  # 抛出错误，量化模型只能用于 LoRA 微调

        if finetuning_args.pissa_init:  # 如果使用 PiSSA 初始化
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")  # 抛出错误，无法在量化模型上初始化 PiSSA 适配器

    # cast trainable parameters to float32 if:  # 如果满足以下条件，将可训练参数转换为 float32：
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)  # 1. 可训练且不是纯 bf16 且不是 badam 且量化位不为 None（qlora）
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 and not fsdp (zero3 or fsdp already in fp32)  # 2. 可训练且不是纯 bf16 且不是 badam 且不是 zero3 且不是 fsdp（zero3 或 fsdp 已经是 fp32）
    cast_trainable_params_to_fp32 = False  # 将可训练参数转换为 FP32 的标志初始化为 False
    if not is_trainable:  # 如果不可训练
        pass  # 不做任何操作
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:  # 如果是纯 bf16 或使用 BAdam
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")  # 记录检测到纯 bf16 / BAdam，剩余可训练参数为半精度
    elif model_args.quantization_bit is None and (is_deepspeed_zero3_enabled() or is_fsdp_enabled()):  # 如果量化位为 None 且启用了 ZeRO3 或 FSDP
        logger.info_rank0("ZeRO3 / FSDP detected, remaining trainable params in float32.")  # 记录检测到 ZeRO3 / FSDP，剩余可训练参数为 float32
    else:  # 否则
        logger.info_rank0("Upcasting trainable params to float32.")  # 记录将可训练参数提升为 float32。
        cast_trainable_params_to_fp32 = True  # 设置标志为 True

    if finetuning_args.finetuning_type == "full":  # 如果微调类型为完整微调
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)  # 设置完整微调
    elif finetuning_args.finetuning_type == "freeze":  # 如果微调类型为冻结
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)  # 设置冻结微调
    elif finetuning_args.finetuning_type == "lora":  # 如果微调类型为 LoRA
        model = _setup_lora_tuning(  # 设置 LoRA 微调
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:  # 否则
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")  # 抛出错误，未知的微调类型

    return model  # 返回模型