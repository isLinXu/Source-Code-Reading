from typing import TYPE_CHECKING

from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available
from transformers.utils.versions import require_version

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)  # 获取日志记录器


def configure_attn_implementation(
    config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool
) -> None:
    # 如果模型类型为 "gemma2" 且可训练
    if getattr(config, "model_type", None) == "gemma2" and is_trainable:
        # 如果 flash_attn 为 "auto" 或 "fa2"
        if model_args.flash_attn == "auto" or model_args.flash_attn == "fa2":
            if is_flash_attn_2_available():  # 检查 FlashAttention-2 是否可用
                require_version("transformers>=4.42.4", "To fix: pip install transformers>=4.42.4")  # 要求 transformers 版本
                require_version("flash_attn>=2.6.3", "To fix: pip install flash_attn>=2.6.3")  # 要求 flash_attn 版本
                if model_args.flash_attn != "fa2":  # 如果 flash_attn 不是 "fa2"
                    logger.warning_rank0("Gemma-2 should use flash attention 2, change `flash_attn` to fa2.")  # 记录警告
                    model_args.flash_attn = "fa2"  # 将 flash_attn 设置为 "fa2"
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")  # 记录警告
                model_args.flash_attn = "disabled"  # 禁用 flash_attn
        elif model_args.flash_attn == "sdpa":  # 如果 flash_attn 为 "sdpa"
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )  # 记录警告

    if model_args.flash_attn == "auto":  # 如果 flash_attn 为 "auto"
        return  # 直接返回

    elif model_args.flash_attn == "disabled":  # 如果 flash_attn 被禁用
        requested_attn_implementation = "eager"  # 请求的注意力实现为 "eager"

    elif model_args.flash_attn == "sdpa":  # 如果 flash_attn 为 "sdpa"
        if not is_torch_sdpa_available():  # 检查 torch 是否可用
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")  # 记录警告
            return  # 直接返回

        requested_attn_implementation = "sdpa"  # 请求的注意力实现为 "sdpa"
    elif model_args.flash_attn == "fa2":  # 如果 flash_attn 为 "fa2"
        if not is_flash_attn_2_available():  # 检查 FlashAttention-2 是否可用
            logger.warning_rank0("FlashAttention-2 is not installed.")  # 记录警告
            return  # 直接返回

        requested_attn_implementation = "flash_attention_2"  # 请求的注意力实现为 "flash_attention_2"
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")  # 抛出未实现的错误

    # 特殊情况处理自定义模型
    if getattr(config, "model_type", None) == "internlm2":
        setattr(config, "attn_implementation", requested_attn_implementation)  # 设置注意力实现
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)  # 设置注意力实现


def print_attn_implementation(config: "PretrainedConfig") -> None:
    # 特殊情况处理自定义模型
    if getattr(config, "model_type", None) == "internlm2":
        attn_implementation = getattr(config, "attn_implementation", None)  # 获取注意力实现
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)  # 获取注意力实现

    if attn_implementation == "flash_attention_2":  # 如果使用 FlashAttention-2
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")  # 记录信息
    elif attn_implementation == "sdpa":  # 如果使用 SDPA
        logger.info_rank0("Using torch SDPA for faster training and inference.")  # 记录信息
    else:
        logger.info_rank0("Using vanilla attention implementation.")  # 记录信息