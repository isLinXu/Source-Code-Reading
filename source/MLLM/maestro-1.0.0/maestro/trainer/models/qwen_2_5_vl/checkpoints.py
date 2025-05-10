# 导入操作系统模块
import os
# 枚举类型支持
from enum import Enum
# 类型提示相关
from typing import Optional

# PyTorch深度学习框架
import torch
# PEFT（参数高效微调）相关模块
from peft import LoraConfig, get_peft_model
#  transformers库中的模型和处理器
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# 设备解析工具函数
from maestro.trainer.common.utils.device import parse_device_spec

# 默认模型ID（HuggingFace模型仓库路径）
DEFAULT_QWEN2_5_VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# 默认模型版本（Git分支）
DEFAULT_QWEN2_5_VL_MODEL_REVISION = "refs/heads/main"


class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies. 优化策略枚举"""
    LORA = "lora"    # LoRA低秩适应微调
    QLORA = "qlora"  # 量化LoRA（4-bit量化版）
    NONE = "none"    # 不使用优化策略


def load_model(
    model_id_or_path: str = DEFAULT_QWEN2_5_VL_MODEL_ID,  # 模型标识或本地路径
    revision: str = DEFAULT_QWEN2_5_VL_MODEL_REVISION,     # 模型版本（Git分支）
    device: str | torch.device = "auto",                   # 目标设备（自动检测）
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.LORA,  # 优化策略选择
    cache_dir: Optional[str] = None,                       # 模型缓存目录
    min_pixels: int = 256 * 28 * 28,  # 图像最小像素数（256x28x28=200,704）
    max_pixels: int = 1280 * 28 * 28, # 图像最大像素数（1280x28x28=1,003,520）
) -> tuple[Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration]:
    """
    Loads a Qwen2.5-VL model and its associated processor with optional LoRA or QLoRA.

    Args:
        model_id_or_path (str): The model name or path.
        revision (str): The model revision to load.
        device (str | torch.device): The device to load the model onto.
        optimization_strategy (OptimizationStrategy): LORA, QLORA, or NONE.
        cache_dir (Optional[str]): Directory to cache downloaded model files.
        min_pixels (int): Minimum number of pixels allowed in the resized image.
        max_pixels (int): Maximum number of pixels allowed in the resized image.

    Returns:
        (Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration):
            A tuple containing the loaded processor and model.
    """
    # 解析设备规格（如"cuda:0"或"cpu"）
    device = parse_device_spec(device)
    
    # 初始化处理器（包含tokenizer和image processor）
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_id_or_path,
        revision=revision,
        trust_remote_code=True,  # 信任远程代码（模型自定义代码）
        cache_dir=cache_dir,     # 指定缓存目录
        min_pixels=min_pixels,   # 设置图像处理的最小像素限制
        max_pixels=max_pixels,   # 设置图像处理的最大像素限制
    )

    # 判断是否使用LoRA类优化策略
    if optimization_strategy in {OptimizationStrategy.LORA, OptimizationStrategy.QLORA}:
        # LoRA配置参数
        lora_config = LoraConfig(
            r=8,                  # 低秩矩阵的秩
            lora_alpha=16,        # 缩放系数
            lora_dropout=0.05,    # Dropout概率
            bias="none",          # 不使用偏置项
            target_modules=["q_proj", "v_proj"],  # 要适配的注意力模块
            task_type="CAUSAL_LM",# 因果语言模型任务
        )

        # 仅当使用QLoRA时配置4-bit量化参数
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,            # 启用4-bit量化加载
                bnb_4bit_use_double_quant=True,  # 使用双重量化减少存储
                bnb_4bit_quant_type="nf4",    # 使用4-bit NormalFloat量化
                bnb_4bit_compute_type=torch.bfloat16,  # 计算时使用bfloat16
            )
            if optimization_strategy == OptimizationStrategy.QLORA
            else None
        )

        # 加载基础模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,  # 信任模型自定义代码
            device_map="auto",       # 自动分配设备（多GPU支持）
            quantization_config=bnb_config,  # 应用量化配置（QLoRA时生效）
            torch_dtype=torch.bfloat16,  # 使用bfloat16精度
            cache_dir=cache_dir,     # 模型缓存目录
        )
        # 应用LoRA适配器
        model = get_peft_model(model, lora_config)
        # 打印可训练参数信息
        model.print_trainable_parameters()
    else:
        # 加载原始模型（不使用PEFT）
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",       # 自动分配设备
            torch_dtype=torch.bfloat16,  # 使用bfloat16精度
            cache_dir=cache_dir,
        )
        # 将模型移动到指定设备
        model.to(device)

    return processor, model


def save_model(
    target_dir: str,  # 目标保存目录
    processor: Qwen2_5_VLProcessor,  # 要保存的处理器
    model: Qwen2_5_VLForConditionalGeneration,  # 要保存的模型
) -> None:
    """
    Save a Qwen2.5-VL model and its processor to disk.

    Args:
        target_dir: Directory path where the model and processor will be saved.
            Will be created if it doesn't exist.
        processor: The Qwen2.5-VL processor to save.
        model: The Qwen2.5-VL model to save.
    """
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    # 保存处理器（包含tokenizer和image processor配置）
    processor.save_pretrained(target_dir)
    # 保存模型权重和配置
    model.save_pretrained(target_dir)
