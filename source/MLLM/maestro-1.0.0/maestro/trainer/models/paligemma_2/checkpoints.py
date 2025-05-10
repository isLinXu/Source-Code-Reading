import os  # 导入操作系统接口模块
from enum import Enum  # 从enum模块导入Enum，用于创建枚举类型
from typing import Optional  # 从typing模块导入Optional，用于类型注解

import torch  # 导入PyTorch库
from peft import LoraConfig, get_peft_model  # 从peft库导入LoRA配置和获取peft模型的函数
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor  # 从transformers库导入相关类和配置

from maestro.trainer.common.utils.device import parse_device_spec  # 从maestro.trainer.common.utils.device导入设备解析函数

DEFAULT_PALIGEMMA2_MODEL_ID = "google/paligemma2-3b-pt-224"  # 默认的PaliGemma2模型ID
DEFAULT_PALIGEMMA2_MODEL_REVISION = "refs/heads/main"  # 默认的PaliGemma2模型修订版本


class OptimizationStrategy(Enum):  # 定义优化策略枚举类
    """Enumeration for optimization strategies."""
    """优化策略的枚举。"""

    LORA = "lora"  # LoRA优化策略
    QLORA = "qlora"  # QLoRA优化策略
    FREEZE = "freeze"  # 冻结优化策略
    NONE = "none"  # 无优化策略


def load_model(  # 定义加载模型的函数
    model_id_or_path: str = DEFAULT_PALIGEMMA2_MODEL_ID,  # 模型ID或路径，默认为预定义的常量
    revision: str = DEFAULT_PALIGEMMA2_MODEL_REVISION,  # 模型修订版本，默认为预定义的常量
    device: str | torch.device = "auto",  # 设备，默认为 "auto"
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.LORA,  # 优化策略，默认为LoRA
    cache_dir: Optional[str] = None,  # 缓存目录，默认为None
) -> tuple[PaliGemmaProcessor, PaliGemmaForConditionalGeneration]:  # 返回类型为处理器和模型的元组
    """Loads a PaliGemma 2 model and its associated processor.
    加载PaliGemma 2模型及其关联的处理器。

    Args:
        model_id_or_path (str): The identifier or path of the model to load.
                               要加载的模型的标识符或路径。
        revision (str): The specific model revision to use.
                       要使用的特定模型修订版本。
        device (torch.device): The device to load the model onto.
                              加载模型到的设备。
        optimization_strategy (OptimizationStrategy): The optimization strategy to apply to the model.
                                                     要应用于模型的优化策略。
        cache_dir (Optional[str]): Directory to cache the downloaded model files.
                                  缓存下载的模型文件的目录。

    Returns:
        (PaliGemmaProcessor, PaliGemmaForConditionalGeneration):
            A tuple containing the loaded processor and model.
            包含加载的处理器和模型的元组。

    Raises:
        ValueError: If the model or processor cannot be loaded.
                   如果无法加载模型或处理器。
    """
    device = parse_device_spec(device)  # 解析设备规范字符串
    processor = PaliGemmaProcessor.from_pretrained(  # 从预训练模型加载处理器
        model_id_or_path,  # 模型ID或路径
        trust_remote_code=True,  # 信任远程代码
        revision=revision,  # 模型修订版本
    )

    if optimization_strategy in {OptimizationStrategy.LORA, OptimizationStrategy.QLORA}:  # 如果优化策略是LoRA或QLoRA
        lora_config = LoraConfig(  # 创建LoRA配置
            r=8,  # LoRA的秩
            lora_alpha=16,  # LoRA的alpha值
            lora_dropout=0.05,  # LoRA的dropout率
            bias="none",  # 不使用偏置
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
            task_type="CAUSAL_LM",  # 任务类型为因果语言模型
        )
        bnb_config = (  # 创建BitsAndBytes配置
            BitsAndBytesConfig(  # 如果是QLoRA策略
                load_in_4bit=True,  # 使用4位量化
                bnb_4bit_quant_type="nf4",  # 使用nf4量化类型
                bnb_4bit_compute_type=torch.bfloat16,  # 使用bfloat16计算类型
            )
            if optimization_strategy == OptimizationStrategy.QLORA  # 如果是QLoRA策略
            else None  # 否则为None
        )

        model = PaliGemmaForConditionalGeneration.from_pretrained(  # 从预训练模型加载模型
            pretrained_model_name_or_path=model_id_or_path,  # 预训练模型名称或路径
            revision=revision,  # 模型修订版本
            trust_remote_code=True,  # 信任远程代码
            device_map="auto",  # 自动设备映射
            quantization_config=bnb_config,  # 量化配置
            torch_dtype=torch.bfloat16,  # 使用bfloat16数据类型
            cache_dir=cache_dir,  # 缓存目录
        )
        model = get_peft_model(model, lora_config)  # 获取peft模型
        model.print_trainable_parameters()  # 打印可训练参数
    else:  # 如果优化策略不是LoRA或QLoRA
        model = PaliGemmaForConditionalGeneration.from_pretrained(  # 从预训练模型加载模型
            pretrained_model_name_or_path=model_id_or_path,  # 预训练模型名称或路径
            revision=revision,  # 模型修订版本
            trust_remote_code=True,  # 信任远程代码
            device_map="auto",  # 自动设备映射
            cache_dir=cache_dir,  # 缓存目录
        ).to(device)  # 将模型移动到指定设备

        if optimization_strategy == OptimizationStrategy.FREEZE:  # 如果优化策略是冻结
            for param in model.vision_tower.parameters():  # 遍历视觉塔的参数
                param.requires_grad = False  # 冻结参数

            for param in model.multi_modal_projector.parameters():  # 遍历多模态投影器的参数
                param.requires_grad = False  # 冻结参数

    return processor, model  # 返回处理器和模型


def save_model(  # 定义保存模型的函数
    target_dir: str,  # 目标目录
    processor: PaliGemmaProcessor,  # PaliGemma处理器
    model: PaliGemmaForConditionalGeneration,  # PaliGemma模型
) -> None:  # 返回类型为None
    """
    Save a PaliGemma 2 model and its processor to disk.
    将PaliGemma 2模型及其处理器保存到磁盘。

    Args:
        target_dir: Directory path where the model and processor will be saved.
                   模型和处理器将被保存的目录路径。
            Will be created if it doesn't exist.
            如果不存在将被创建。
        processor: The PaliGemma 2 processor to save.
                  要保存的PaliGemma 2处理器。
        model: The PaliGemma 2model to save.
               要保存的PaliGemma 2模型。
    """
    os.makedirs(target_dir, exist_ok=True)  # 创建目标目录，如果已存在则忽略
    processor.save_pretrained(target_dir)  # 保存处理器
    model.save_pretrained(target_dir)  # 保存模型
