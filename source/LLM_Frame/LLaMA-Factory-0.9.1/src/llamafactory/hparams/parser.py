# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available
from transformers.utils.versions import require_version

from ..extras import logging
from ..extras.constants import CHECKPOINT_NAMES
from ..extras.misc import check_dependencies, get_current_device
from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments

logger = logging.get_logger(__name__)

check_dependencies()
# 检查依赖关系。

_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
# 训练参数的类列表。
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
# 训练参数的元组类型。
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
# 推理参数的类列表。
_INFER_CLS = Tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
# 推理参数的元组类型。
_EVAL_ARGS = [ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
# 评估参数的类列表。
_EVAL_CLS = Tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
# 评估参数的元组类型。

def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    # 解析命令行参数。
    if args is not None:
        return parser.parse_dict(args)
        # 如果提供了args字典，则解析并返回。

    if len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # 如果命令行参数是一个YAML文件，则解析并返回。

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
        # 如果命令行参数是一个JSON文件，则解析并返回。

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    # 解析剩余的命令行参数，返回已解析的参数和未知参数。

    if unknown_args:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")
        # 如果有未知参数，则打印帮助信息并引发错误。

    return (*parsed_args,)
    # 返回解析后的参数。

def _set_transformers_logging() -> None:
    # 设置Transformers的日志记录。
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def _verify_model_args(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    # 验证模型参数。
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")
        # 如果适配器路径不为None且微调类型不是LoRA，则引发错误。

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantization is only compatible with the LoRA method.")
            # 如果量化位数不为None且微调类型不是LoRA，则引发错误。

        if finetuning_args.pissa_init:
            raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA for a quantized model.")
            # 如果使用了PiSSA初始化，则引发错误。

        if model_args.resize_vocab:
            raise ValueError("Cannot resize embedding layers of a quantized model.")
            # 如果尝试调整量化模型的嵌入层大小，则引发错误。

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")
            # 如果尝试在量化模型上创建新适配器，则引发错误。

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")
            # 如果量化模型的适配器数量不为1，则引发错误。

    if data_args.template == "yi" and model_args.use_fast_tokenizer:
        logger.warning_rank0("We should use slow tokenizer for the Yi models. Change `use_fast_tokenizer` to False.")
        model_args.use_fast_tokenizer = False
        # 如果模板为"yi"且使用快速分词器，则发出警告并将其设置为False。

def _check_extra_dependencies(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    training_args: Optional["Seq2SeqTrainingArguments"] = None,
) -> None:
    # 检查额外依赖关系。
    if model_args.use_unsloth:
        require_version("unsloth", "Please install unsloth: https://github.com/unslothai/unsloth")
        # 如果使用unsloth，则要求安装该库。

    if model_args.enable_liger_kernel:
        require_version("liger-kernel", "To fix: pip install liger-kernel")
        # 如果启用liger内核，则要求安装该库。

    if model_args.mixture_of_depths is not None:
        require_version("mixture-of-depth>=1.1.6", "To fix: pip install mixture-of-depth>=1.1.6")
        # 如果使用深度混合，则要求安装该库。

    if model_args.infer_backend == "vllm":
        require_version("vllm>=0.4.3,<0.6.4", "To fix: pip install vllm>=0.4.3,<0.6.4")
        # 如果推断后端为vLLM，则要求安装该库。

    if finetuning_args.use_galore:
        require_version("galore_torch", "To fix: pip install galore_torch")
        # 如果使用GaLore，则要求安装该库。

    if finetuning_args.use_badam:
        require_version("badam>=1.2.1", "To fix: pip install badam>=1.2.1")
        # 如果使用Badam，则要求安装该库。

    if finetuning_args.use_adam_mini:
        require_version("adam-mini", "To fix: pip install adam-mini")
        # 如果使用Adam Mini，则要求安装该库。

    if finetuning_args.plot_loss:
        require_version("matplotlib", "To fix: pip install matplotlib")
        # 如果绘制损失，则要求安装该库。

    if training_args is not None and training_args.predict_with_generate:
        require_version("jieba", "To fix: pip install jieba")
        require_version("nltk", "To fix: pip install nltk")
        require_version("rouge_chinese", "To fix: pip install rouge-chinese")
        # 如果预测时使用生成，则要求安装相关库。

def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args)
    # 解析训练参数。

def _parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return _parse_args(parser, args)
    # 解析推理参数。

def _parse_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    parser = HfArgumentParser(_EVAL_ARGS)
    return _parse_args(parser, args)
    # 解析评估参数。

def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()
        # 如果需要记录，则设置日志。

    # Check arguments
    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")
        # 如果微调阶段不是"pt"且模板为None，则引发错误。

    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")
            # 如果在非SFT阶段设置了预测生成，则引发错误。

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")
            # 如果在非SFT阶段设置了整洁打包，则引发错误。

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt` or `mask_history` cannot be set as True except SFT.")
            # 如果在非SFT阶段设置了训练提示或掩码历史，则引发错误。

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")
        # 如果在SFT阶段进行预测但未启用预测生成，则引发错误。

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")
        # 如果在RM或PPO阶段设置了在最后加载最佳模型，则引发错误。

    if finetuning_args.stage == "ppo":
        if not training_args.do_train:
            raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")
            # 如果在PPO阶段未进行训练，则引发错误。

        if model_args.shift_attn:
            raise ValueError("PPO training is incompatible with S^2-Attn.")
            # 如果在PPO阶段启用了短注意力，则引发错误。

        if finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
            raise ValueError("Unsloth does not support lora reward model.")
            # 如果在PPO阶段使用了lora奖励模型且启用了unsloth，则引发错误。

        if training_args.report_to and training_args.report_to[0] not in ["wandb", "tensorboard"]:
            raise ValueError("PPO only accepts wandb or tensorboard logger.")
            # 如果在PPO阶段的报告目标不在允许的日志记录器中，则引发错误。

    if training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")
        # 如果并行模式不是分布式，则引发错误。

    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")
        # 如果启用了DeepSpeed但并行模式不是分布式，则引发错误。

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")
        # 如果在流模式下未指定最大步骤，则引发错误。

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")
        # 如果在训练时未指定数据集，则引发错误。

    if (training_args.do_eval or training_args.do_predict) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please specify dataset for evaluation.")
        # 如果在评估或预测时未指定评估数据集，则引发错误。

    if training_args.predict_with_generate:
        if is_deepspeed_zero3_enabled():
            raise ValueError("`predict_with_generate` is incompatible with DeepSpeed ZeRO-3.")
            # 如果启用了DeepSpeed ZeRO-3且设置了预测生成，则引发错误。

        if data_args.eval_dataset is None:
            raise ValueError("Cannot use `predict_with_generate` if `eval_dataset` is None.")
            # 如果评估数据集为None，则引发错误。

        if finetuning_args.compute_accuracy:
            raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")
            # 如果同时设置了预测生成和计算准确性，则引发错误。

    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")
        # 如果在训练时对量化模型使用设备映射，则引发错误。

    if finetuning_args.pissa_init and is_deepspeed_zero3_enabled():
        raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA in DeepSpeed ZeRO-3.")
        # 如果在DeepSpeed ZeRO-3中使用了PiSSA初始化，则引发错误。

    if finetuning_args.pure_bf16:
        if not (is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())):
            raise ValueError("This device does not support `pure_bf16`.")
            # 如果设备不支持纯bf16，则引发错误。

        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")
            # 如果在DeepSpeed ZeRO-3中使用了纯bf16，则引发错误。

    if (
        finetuning_args.use_galore
        and finetuning_args.galore_layerwise
        and training_args.parallel_mode == ParallelMode.DISTRIBUTED
    ):
        raise ValueError("Distributed training does not support layer-wise GaLore.")
        # 如果在分布式训练中使用了层级GaLore，则引发错误。

    if finetuning_args.use_badam and training_args.parallel_mode == ParallelMode.DISTRIBUTED:
        if finetuning_args.badam_mode == "ratio":
            raise ValueError("Radio-based BAdam does not yet support distributed training, use layer-wise BAdam.")
            # 如果在分布式训练中使用了基于比例的Badam，则引发错误。
        elif not is_deepspeed_zero3_enabled():
            raise ValueError("Layer-wise BAdam only supports DeepSpeed ZeRO-3 training.")
            # 如果在分布式训练中使用了层级Badam但未启用DeepSpeed，则引发错误。

    if finetuning_args.use_galore and training_args.deepspeed is not None:
        raise ValueError("GaLore is incompatible with DeepSpeed yet.")
        # 如果在DeepSpeed中使用了GaLore，则引发错误。

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")
        # 如果推断后端为vLLM，则引发错误。

    if model_args.use_unsloth and is_deepspeed_zero3_enabled():
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")
        # 如果在DeepSpeed ZeRO-3中使用了unsloth，则引发错误。

    if data_args.neat_packing and not data_args.packing:
        logger.warning_rank0("`neat_packing` requires `packing` is True. Change `packing` to True.")
        data_args.packing = True
        # 如果使用整洁打包但未设置打包，则发出警告并将其设置为True。

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)

    if (
        training_args.do_train
        and finetuning_args.finetuning_type == "lora"
        and model_args.quantization_bit is None
        and model_args.resize_vocab
        and finetuning_args.additional_target is None
    ):
        logger.warning_rank0(
            "Remember to add embedding layers to `additional_target` to make the added tokens trainable."
        )
        # 如果在训练时使用LoRA且未设置量化位数、调整词汇表和附加目标，则发出警告。

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning_rank0("We recommend enable `upcast_layernorm` in quantized training.")
        # 如果在量化训练中未启用上升层归一化，则发出警告。

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning_rank0("We recommend enable mixed precision training.")
        # 如果在训练中未启用混合精度，则发出警告。

    if training_args.do_train and finetuning_args.use_galore and not finetuning_args.pure_bf16:
        logger.warning_rank0(
            "Using GaLore with mixed precision training may significantly increases GPU memory usage."
        )
        # 如果在混合精度训练中使用GaLore，则发出警告。

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning_rank0("Evaluating model in 4/8-bit mode may cause lower scores.")
        # 如果在非训练模式下评估量化模型，则发出警告。

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning_rank0("Specify `ref_model` for computing rewards at evaluation.")
        # 如果在评估阶段未指定参考模型，则发出警告。

    # Post-process training arguments
    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning_rank0("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False
        # 如果在分布式训练中使用LoRA且未设置未使用参数，则发出警告并将其设置为False。

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning_rank0("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
            # 如果在当前阶段无法从检查点恢复，则发出警告并将其设置为None。
    else:
        can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
        and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(
            os.path.isfile(os.path.join(training_args.output_dir, name)) for name in CHECKPOINT_NAMES
        ):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")
            # 如果输出目录已存在且不为空，则引发错误。

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info_rank0(f"Resuming training from {training_args.resume_from_checkpoint}.")
            logger.info_rank0("Change `output_dir` or use `overwrite_output_dir` to avoid.")
            # 如果存在最后的检查点，则设置为恢复训练。

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning_rank0(
            "Add {} to `adapter_name_or_path` to resume training from checkpoint.".format(
                training_args.resume_from_checkpoint
            )
        )
        # 如果在RM或PPO阶段恢复训练，则发出警告。

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
        # 如果启用了bf16或纯bf16，则设置计算数据类型为bfloat16。
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16
        # 如果启用了fp16，则设置计算数据类型为float16。

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"
    # 设置模型参数和数据参数。

    # Log on each process the small summary
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode == ParallelMode.DISTRIBUTED,
            str(model_args.compute_dtype),
        )
    )
    # 在每个进程中记录小摘要。

    transformers.set_seed(training_args.seed)
    # 设置随机种子。

    return model_args, data_args, training_args, finetuning_args, generating_args
    # 返回模型参数、数据参数、训练参数、微调参数和生成参数。

def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(args)

    _set_transformers_logging()
    # 设置Transformers的日志记录。

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")
        # 如果模板为None，则引发错误。

    if model_args.infer_backend == "vllm":
        if finetuning_args.stage != "sft":
            raise ValueError("vLLM engine only supports auto-regressive models.")
            # 如果vLLM引擎的阶段不是SFT，则引发错误。

        if model_args.quantization_bit is not None:
            raise ValueError("vLLM engine does not support bnb quantization (GPTQ and AWQ are supported).")
            # 如果vLLM引擎不支持bnb量化，则引发错误。

        if model_args.rope_scaling is not None:
            raise ValueError("vLLM engine does not support RoPE scaling.")
            # 如果vLLM引擎不支持RoPE缩放，则引发错误。

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("vLLM only accepts a single adapter. Merge them first.")
            # 如果vLLM引擎的适配器数量不为1，则引发错误。

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    if model_args.export_dir is not None and model_args.export_device == "cpu":
        model_args.device_map = {"": torch.device("cpu")}
        model_args.model_max_length = data_args.cutoff_len
    else:
        model_args.device_map = "auto"
        # 设置设备映射。

    return model_args, data_args, finetuning_args, generating_args
    # 返回模型参数、数据参数、微调参数和生成参数。

def get_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = _parse_eval_args(args)

    _set_transformers_logging()
    # 设置Transformers的日志记录。

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")
        # 如果模板为None，则引发错误。

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")
        # 如果vLLM后端仅适用于API、CLI和Web，则引发错误。

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = "auto"
    # 设置设备映射为自动。

    transformers.set_seed(eval_args.seed)
    # 设置随机种子。

    return model_args, data_args, eval_args, finetuning_args
    # 返回模型参数、数据参数、评估参数和微调参数。