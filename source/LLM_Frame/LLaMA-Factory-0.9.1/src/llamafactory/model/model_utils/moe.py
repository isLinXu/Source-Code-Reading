from typing import TYPE_CHECKING, Sequence  # 导入类型检查和序列类型

import torch  # 导入 PyTorch 库
from transformers.integrations import is_deepspeed_zero3_enabled  # 从 transformers 导入检查是否启用 DeepSpeed Zero3 的函数
from transformers.utils.versions import require_version  # 从 transformers 导入版本检查函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PretrainedConfig, PreTrainedModel  # 导入预训练配置和模型的类型

    from ...hparams import ModelArguments  # 导入模型参数的类型


def _set_z3_leaf_modules(model: "PreTrainedModel", leaf_modules: Sequence["torch.nn.Module"]) -> None:
    require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")  # 检查 DeepSpeed 版本
    from deepspeed.utils import set_z3_leaf_modules  # 从 deepspeed 导入设置 Z3 叶模块的函数

    set_z3_leaf_modules(model, leaf_modules)  # 设置模型的 Z3 叶模块


def add_z3_leaf_module(model: "PreTrainedModel") -> None:
    r"""
    Sets module as a leaf module to skip partitioning in deepspeed zero3.  # 将模块设置为叶模块，以跳过 DeepSpeed Zero3 中的分区
    """
    if not is_deepspeed_zero3_enabled():  # 如果未启用 DeepSpeed Zero3
        return  # 直接返回

    model_type = getattr(model.config, "model_type", None)  # 获取模型类型
    if model_type == "dbrx":  # 如果模型类型为 dbrx
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN  # 从 transformers 导入 DbrxFFN 模块

        _set_z3_leaf_modules(model, [DbrxFFN])  # 设置 DbrxFFN 为叶模块

    if model_type == "jamba":  # 如果模型类型为 jamba
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock  # 从 transformers 导入 JambaSparseMoeBlock 模块

        _set_z3_leaf_modules(model, [JambaSparseMoeBlock])  # 设置 JambaSparseMoeBlock 为叶模块

    if model_type == "jetmoe":  # 如果模型类型为 jetmoe
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE  # 从 transformers 导入 JetMoeMoA 和 JetMoeMoE 模块

        _set_z3_leaf_modules(model, [JetMoeMoA, JetMoeMoE])  # 设置 JetMoeMoA 和 JetMoeMoE 为叶模块

    if model_type == "mixtral":  # 如果模型类型为 mixtral
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # 从 transformers 导入 MixtralSparseMoeBlock 模块

        _set_z3_leaf_modules(model, [MixtralSparseMoeBlock])  # 设置 MixtralSparseMoeBlock 为叶模块

    if model_type == "qwen2moe":  # 如果模型类型为 qwen2moe
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock  # 从 transformers 导入 Qwen2MoeSparseMoeBlock 模块

        _set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])  # 设置 Qwen2MoeSparseMoeBlock 为叶模块


def configure_moe(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    model_type = getattr(config, "model_type", None)  # 获取模型类型
    if model_args.moe_aux_loss_coef is not None:  # 如果模型参数中有辅助损失系数
        if model_type in ["jamba", "mixtral", "qwen2_moe"]:  # 如果模型类型在指定列表中
            setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)  # 设置路由辅助损失系数

        elif model_type == "deepseek":  # 如果模型类型为 deepseek
            setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)  # 设置辅助损失的 alpha 值

        elif model_type == "jetmoe":  # 如果模型类型为 jetmoe
            setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)  # 设置辅助损失系数

    if model_type in ["dbrx", "jamba", "jetmoe", "mixtral", "qwen2_moe"]:  # 如果模型类型在指定列表中
        setattr(config, "output_router_logits", is_trainable)  # 设置输出路由 logits 的可训练性