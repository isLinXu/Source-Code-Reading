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

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict  # 从 typing 模块导入类型检查、任意类型、字典、可选类型和类型字典

import torch  # 导入 PyTorch 库
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer  # 从 transformers 模块导入相关类
from trl import AutoModelForCausalLMWithValueHead  # 从 trl 模块导入带值头的因果语言模型

from ..extras import logging  # 从 extras 模块导入日志记录
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub  # 导入辅助函数
from .adapter import init_adapter  # 从 adapter 模块导入初始化适配器的函数
from .model_utils.liger_kernel import apply_liger_kernel  # 从 liger_kernel 模块导入应用 Liger 核心的函数
from .model_utils.misc import register_autoclass  # 从 misc 模块导入注册自动类的函数
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model  # 从 mod 模块导入相关函数
from .model_utils.unsloth import load_unsloth_pretrained_model  # 从 unsloth 模块导入加载预训练模型的函数
from .model_utils.valuehead import load_valuehead_params  # 从 valuehead 模块导入加载值头参数的函数
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model  # 从 patcher 模块导入修补相关函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin  # 导入预训练配置、模型、分词器和处理器

    from ..hparams import FinetuningArguments, ModelArguments  # 从 hparams 模块导入微调参数和模型参数


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class TokenizerModule(TypedDict):  # 定义 TokenizerModule 类型字典
    tokenizer: "PreTrainedTokenizer"  # 分词器类型
    processor: Optional["ProcessorMixin"]  # 可选的处理器类型


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:  # 定义获取初始化关键字参数的函数
    r"""
    Gets arguments to load config/tokenizer/model.  # 获取加载配置/分词器/模型的参数

    Note: including inplace operation of model_args.  # 注意：包括对 model_args 的就地操作。
    """
    skip_check_imports()  # 跳过检查导入
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)  # 尝试从其他库下载模型
    return {  # 返回初始化关键字参数字典
        "trust_remote_code": True,  # 信任远程代码
        "cache_dir": model_args.cache_dir,  # 缓存目录
        "revision": model_args.model_revision,  # 版本
        "token": model_args.hf_hub_token,  # 令牌
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":  # 定义加载分词器的函数
    r"""
    Loads pretrained tokenizer and optionally loads processor.  # 加载预训练分词器并可选地加载处理器

    Note: including inplace operation of model_args.  # 注意：包括对 model_args 的就地操作。
    """
    init_kwargs = _get_init_kwargs(model_args)  # 获取初始化关键字参数
    config = load_config(model_args)  # 加载配置
    try:  # 尝试加载分词器
        tokenizer = AutoTokenizer.from_pretrained(  # 从预训练模型中加载分词器
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,  # 使用快速分词器
            split_special_tokens=model_args.split_special_tokens,  # 拆分特殊标记
            padding_side="right",  # 填充方向为右侧
            **init_kwargs,  # 解包初始化关键字参数
        )
    except ValueError:  # 如果出现值错误，尝试快速分词器
        tokenizer = AutoTokenizer.from_pretrained(  # 从预训练模型中加载快速分词器
            model_args.model_name_or_path,
            use_fast=True,  # 强制使用快速分词器
            padding_side="right",  # 填充方向为右侧
            **init_kwargs,  # 解包初始化关键字参数
        )
    except Exception as e:  # 捕获其他异常
        raise OSError("Failed to load tokenizer.") from e  # 抛出加载分词器失败的错误

    if model_args.new_special_tokens is not None:  # 如果有新的特殊标记
        num_added_tokens = tokenizer.add_special_tokens(  # 添加特殊标记
            dict(additional_special_tokens=model_args.new_special_tokens),  # 使用新的特殊标记
            replace_additional_special_tokens=False,  # 不替换额外的特殊标记
        )
        logger.info_rank0("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))  # 记录添加的特殊标记
        if num_added_tokens > 0 and not model_args.resize_vocab:  # 如果添加了标记且未调整词汇表
            model_args.resize_vocab = True  # 设置调整词汇表为 True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")  # 记录调整词汇表的警告

    patch_tokenizer(tokenizer)  # 修补分词器
    try:  # 尝试加载处理器
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)  # 从预训练模型中加载处理器
        patch_processor(processor, config, tokenizer, model_args)  # 修补处理器
    except Exception as e:  # 捕获异常
        logger.debug(f"Processor was not found: {e}.")  # 记录处理器未找到的调试信息
        processor = None  # 设置处理器为 None

    # Avoid load tokenizer, see:  # 避免加载处理器，见：
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:  # 如果处理器不为空且类名中不包含 "Processor"
        processor = None  # 设置处理器为 None

    return {"tokenizer": tokenizer, "processor": processor}  # 返回分词器和处理器的字典


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":  # 定义加载模型配置的函数
    r"""
    Loads model config.  # 加载模型配置
    """
    init_kwargs = _get_init_kwargs(model_args)  # 获取初始化关键字参数
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)  # 从预训练模型中加载配置


def load_model(  # 定义加载模型的函数
    tokenizer: "PreTrainedTokenizer",  # 分词器参数
    model_args: "ModelArguments",  # 模型参数
    finetuning_args: "FinetuningArguments",  # 微调参数
    is_trainable: bool = False,  # 是否可训练的布尔值，默认为 False
    add_valuehead: bool = False,  # 是否添加值头的布尔值，默认为 False
) -> "PreTrainedModel":  # 返回类型为预训练模型
    r"""
    Loads pretrained model.  # 加载预训练模型
    """
    init_kwargs = _get_init_kwargs(model_args)  # 获取初始化关键字参数
    config = load_config(model_args)  # 加载配置
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)  # 修补配置
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))  # 应用 Liger 核心

    model = None  # 初始化模型为 None
    lazy_load = False  # 设置懒加载为 False
    if model_args.use_unsloth:  # 如果使用 unsloth
        if model_args.adapter_name_or_path is not None:  # 如果适配器名称或路径不为空
            lazy_load = True  # 设置懒加载为 True
        elif is_trainable:  # 如果可训练
            model = load_unsloth_pretrained_model(config, model_args)  # 加载 unsloth 预训练模型

    if model is None and not lazy_load:  # 如果模型为 None 且不懒加载
        init_kwargs["config"] = config  # 设置初始化关键字参数中的配置
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path  # 设置预训练模型名称或路径

        if model_args.mixture_of_depths == "load":  # 如果混合深度为 "load"
            model = load_mod_pretrained_model(**init_kwargs)  # 加载预训练模型
        else:  # 否则
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # 假设为内置模型
                load_class = AutoModelForVision2Seq  # 设置加载类为视觉模型
            else:
                load_class = AutoModelForCausalLM  # 否则设置为因果语言模型

            if model_args.train_from_scratch:  # 如果从头开始训练
                model = load_class.from_config(config, trust_remote_code=True)  # 从配置加载模型
            else:  # 否则
                model = load_class.from_pretrained(**init_kwargs)  # 从预训练模型加载模型

        if model_args.mixture_of_depths == "convert":  # 如果混合深度为 "convert"
            model = convert_pretrained_model_to_mod(model, config, model_args)  # 转换预训练模型

    if not lazy_load:  # 如果不懒加载
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)  # 修补模型
        register_autoclass(config, model, tokenizer)  # 注册自动类

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)  # 初始化适配器

    if add_valuehead:  # 如果添加值头
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)  # 从预训练模型加载带值头的因果语言模型
        patch_valuehead_model(model)  # 修补值头模型

        if model_args.adapter_name_or_path is not None:  # 如果适配器名称或路径不为空
            vhead_path = model_args.adapter_name_or_path[-1]  # 获取值头路径
        else:
            vhead_path = model_args.model_name_or_path  # 否则使用模型名称或路径

        vhead_params = load_valuehead_params(vhead_path, model_args)  # 加载值头参数
        if vhead_params is not None:  # 如果值头参数不为空
            model.load_state_dict(vhead_params, strict=False)  # 加载状态字典
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")  # 记录加载的值头信息

    if not is_trainable:  # 如果不可训练
        model.requires_grad_(False)  # 设置模型不需要计算梯度
        for param in model.parameters():  # 遍历模型参数
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:  # 如果参数数据类型为 float32 且计算数据类型不为 float32
                param.data = param.data.to(model_args.compute_dtype)  # 将参数数据转换为计算数据类型

        model.eval()  # 设置模型为评估模式
    else:  # 否则
        model.train()  # 设置模型为训练模式

    trainable_params, all_param = count_parameters(model)  # 计算可训练参数和所有参数
    if is_trainable:  # 如果可训练
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(  # 格式化参数统计信息
            trainable_params, all_param, 100 * trainable_params / all_param  # 计算可训练参数比例
        )
    else:  # 否则
        param_stats = f"all params: {all_param:,}"  # 记录所有参数统计信息

    logger.info_rank0(param_stats)  # 记录参数统计信息

    if model_args.print_param_status:  # 如果需要打印参数状态
        for name, param in model.named_parameters():  # 遍历模型的命名参数
            print(  # 打印参数信息
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad  # 参数名称、数据类型、设备和是否可训练
                )
            )

    return model  # 返回模型
