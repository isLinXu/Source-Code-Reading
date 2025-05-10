import os
from enum import Enum
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from maestro.trainer.common.utils.device import parse_device_spec

DEFAULT_FLORENCE2_MODEL_ID = "microsoft/Florence-2-base-ft"
DEFAULT_FLORENCE2_MODEL_REVISION = "refs/pr/20"


class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""
    """优化策略的枚举类。"""

    LORA = "lora"  # LoRA优化策略
    FREEZE = "freeze"  # 冻结部分参数的优化策略
    NONE = "none"  # 无优化策略


def load_model(
    model_id_or_path: str = DEFAULT_FLORENCE2_MODEL_ID,
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.LORA,
    cache_dir: Optional[str] = None,
) -> tuple[AutoProcessor, AutoModelForCausalLM]:
    """Loads a Florence 2 model and its associated processor.
    加载Florence 2模型及其关联的处理器。

    Args:
        model_id_or_path (str): The identifier or path of the Florence 2 model to load.
            要加载的Florence 2模型的标识符或路径。
        revision (str): The specific model revision to use.
            要使用的特定模型版本。
        device (torch.device): The device to load the model onto.
            加载模型的设备。
        optimization_strategy (OptimizationStrategy): The optimization strategy to apply to the model.
            应用于模型的优化策略。
        cache_dir (Optional[str]): Directory to cache the downloaded model files.
            缓存下载模型文件的目录。

    Returns:
        tuple(AutoProcessor, AutoModelForCausalLM):
            A tuple containing the loaded processor and model.
            包含加载的处理器和模型的元组。

    Raises:
        ValueError: If the model or processor cannot be loaded.
            如果无法加载模型或处理器，则抛出ValueError。
    """
    # 解析设备规格
    device = parse_device_spec(device)
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True, revision=revision)

    # 根据优化策略加载模型
    if optimization_strategy == OptimizationStrategy.LORA:
        # 配置LoRA参数
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
            task_type="CAUSAL_LM",
        )
        # 加载模型并应用LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = get_peft_model(model, config).to(device)
        # 打印可训练参数
        model.print_trainable_parameters()
    else:
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        ).to(device)

        # 如果选择冻结策略，冻结视觉塔的参数
        if optimization_strategy == OptimizationStrategy.FREEZE:
            for param in model.vision_tower.parameters():
                param.is_trainable = False

    return processor, model


def save_model(
    target_dir: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
) -> None:
    """
    Saves the processor and model to the specified directory.
    将处理器和模型保存到指定目录。

    Args:
        target_dir (str): The directory to save the model and processor.
            保存模型和处理器的目录。
        processor (AutoProcessor): The processor to save.
            要保存的处理器。
        model (AutoModelForCausalLM): The model to save.
            要保存的模型。
    """
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    # 保存处理器
    processor.save_pretrained(target_dir)
    # 保存模型
    model.save_pretrained(target_dir)
